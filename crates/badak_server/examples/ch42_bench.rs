//! 42장: 서버 여러 대를 띄워 놓고 잰다.
//!
//! 실행: `cargo run --release -p badak_server --example ch42_bench -- \
//!        --part 이름 [--model 이름] [--policy 방법] [--backends 대] \
//!        [--threads 개] [--keep 개] [--warm yes|no]`
//!
//! 먼저 `cargo build --release -p badak_cli`로 실행 파일 `badak`을 만들어
//! 둔다. 이 프로그램이 그것을 자식 프로세스로 띄운다.
//!
//! `--part`:
//! - `route`: 서버 `--backends`대 앞에 라우터를 두고, 가게 여덟 곳의 대화
//!   열여섯 개를 세 차례씩 보낸다. `--policy`로 보내는 방법을 바꾼다.
//! - `cold`: 서버를 새로 띄워, 요청을 받을 수 있게 되기까지와 첫 요청의
//!   첫 글까지를 잰다.
//! - `split`: 글을 읽는 서버와 답을 쓰는 서버를 따로 두고, 읽은 KV 캐시를
//!   넘겨준다.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use badak_formats::json::{self, Value};
use badak_server::client::Client;
use badak_server::router::{Policy, Router};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const SHOPS: [&str; 8] = [
    "Harbor Books",
    "Juniper Pages",
    "The Lantern Shelf",
    "Marigold Books",
    "North Gate Books",
    "Owl and Inkwell",
    "Paper Lighthouse",
    "Quill Corner",
];
const QUESTIONS: [&str; 6] = [
    "Hello, can I return a book I bought last week?",
    "How long does shipping usually take?",
    "Do you sell gift cards?",
    "When are you open on Sundays?",
    "Can my school order thirty copies?",
    "Is there a newsletter I can sign up for?",
];

/// 가게마다의 지시문. 가게의 이름이 맨 앞에 와서, 가게가 다르면 글의
/// 처음부터 다르다.
fn manual(shop: &str, rules: usize) -> String {
    let mut text = format!(
        "{shop}. You are the support assistant of this bookshop. Follow every rule below.\n"
    );
    let topics = [
        "returns",
        "shipping",
        "gift cards",
        "opening hours",
        "book clubs",
        "rare books",
        "school orders",
        "newsletters",
    ];
    for rule in 1..=rules {
        let topic = topics[rule % topics.len()];
        text.push_str(&format!(
            "Rule {rule}: When a customer asks about {topic}, answer in one short sentence, \
             mention policy number {} and never promise a date.\n",
            1000 + rule * 7
        ));
    }
    text
}

fn body(messages: &[(String, String)], limit: usize, reuse: bool) -> String {
    let messages: Vec<String> = messages
        .iter()
        .map(|(role, content)| {
            format!(
                "{{\"role\":\"{role}\",\"content\":{}}}",
                json::quote(content)
            )
        })
        .collect();
    format!(
        "{{\"messages\":[{}],\"max_tokens\":{limit},\"temperature\":0,\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"cache_prompt\":{reuse}}}",
        messages.join(",")
    )
}

/// 받는 쪽에서 잰 응답 하나(초).
#[derive(Debug, Clone, Default)]
struct Measured {
    text: String,
    first: f64,
    total: f64,
    gaps: Vec<f64>,
    prompt: usize,
    cached: usize,
}

fn ask(address: &str, body: &str) -> Outcome<Measured> {
    let response = Client::connect(address)?.send("POST", "/v1/chat/completions", body)?;
    if response.status != 200 {
        return Err(format!("상태 {}: {}", response.status, response.text()).into());
    }
    let mut measured = Measured {
        total: response.total,
        ..Measured::default()
    };
    let mut last = None;
    for (event, arrived) in response.events() {
        if event == "[DONE]" {
            break;
        }
        let root = json::parse(event)?;
        let piece = root
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !piece.is_empty() {
            match last {
                None => measured.first = arrived,
                Some(before) => measured.gaps.push(arrived - before),
            }
            last = Some(arrived);
            measured.text.push_str(piece);
        }
        if let Some(usage) = root.get("usage") {
            let count = |value: Option<&Value>| value.and_then(Value::as_usize).unwrap_or(0);
            measured.prompt = count(usage.get("prompt_tokens"));
            measured.cached = count(
                usage
                    .get("prompt_tokens_details")
                    .and_then(|details| details.get("cached_tokens")),
            );
        }
    }
    Ok(measured)
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// 띄워 둔 서버 하나. 버려지면 끈다.
struct Served {
    child: Child,
    address: String,
    /// 프로세스를 띄운 때부터 요청을 받을 수 있게 되기까지(초).
    ready: f64,
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 빈 포트 하나를 얻는다.
fn free_port() -> Outcome<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// `badak serve`를 띄우고 요청을 받을 때까지 기다린다.
fn serve(model: &str, threads: usize, extra: &[&str]) -> Outcome<Served> {
    let program: PathBuf = std::env::current_exe()?
        .parent()
        .and_then(|examples| examples.parent())
        .ok_or("실행 파일의 자리를 알 수 없습니다")?
        .join("badak");
    let port = free_port()?;
    let address = format!("127.0.0.1:{port}");
    let start = Instant::now();
    let child = Command::new(&program)
        .args(["serve", "--model", model, "--port", &port.to_string()])
        .args(extra)
        .env("BADAK_THREADS", threads.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("{}을 띄우지 못했습니다: {error}", program.display()))?;
    let mut served = Served {
        child,
        address,
        ready: 0.0,
    };
    loop {
        let healthy = Client::connect(&served.address)
            .and_then(|mut client| client.send("GET", "/health", ""))
            .is_ok_and(|response| response.status == 200);
        if healthy {
            served.ready = start.elapsed().as_secs_f64();
            return Ok(served);
        }
        if start.elapsed() > Duration::from_secs(120) {
            return Err("서버가 뜨지 않았습니다".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// 서버 여러 대 앞에 라우터를 두고 가게 여덟 곳의 대화를 보낸다.
fn route(
    model: &str,
    policy: Policy,
    name: &str,
    backends: usize,
    threads: usize,
    keep: usize,
) -> Outcome<()> {
    let keep = keep.to_string();
    let servers: Vec<Served> = (0..backends)
        .map(|_| {
            serve(
                model,
                threads,
                &["--parallel", "2", "--keep", &keep, "--context", "8192"],
            )
        })
        .collect::<Outcome<_>>()?;
    let addresses: Vec<String> = servers
        .iter()
        .map(|server| server.address.clone())
        .collect();
    let router = Arc::new(Router::new(&addresses, policy));
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?.to_string();
    let background = Arc::clone(&router);
    std::thread::spawn(move || background.serve(&listener));

    // 가게마다 손님 둘. 대화 열여섯 개가 세 차례씩 묻는다.
    let talks: Vec<Mutex<Vec<(String, String)>>> = (0..SHOPS.len() * 2)
        .map(|index| Mutex::new(vec![("system".to_owned(), manual(SHOPS[index / 2], 24))]))
        .collect();
    let talks = Arc::new(talks);
    let start = Instant::now();
    let mut turns: Vec<Vec<Measured>> = Vec::new();
    for turn in 0..3 {
        // 대화의 순서를 차례마다 섞는다(늘 같은 순서로 섞인다).
        let mut order: Vec<usize> = (0..talks.len()).collect();
        order.sort_by_key(|&index| (index * 7 + turn * 5) % talks.len());
        let queue = Arc::new(Mutex::new(order));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let (queue, talks, address) =
                    (Arc::clone(&queue), Arc::clone(&talks), address.clone());
                std::thread::spawn(move || -> Result<Vec<Measured>, String> {
                    let mut done = Vec::new();
                    loop {
                        let Some(index) = queue.lock().map_err(|_| "줄이 깨졌습니다")?.pop()
                        else {
                            return Ok(done);
                        };
                        let mut talk = talks[index].lock().map_err(|_| "대화가 깨졌습니다")?;
                        talk.push((
                            "user".to_owned(),
                            QUESTIONS[(index + turn * 3) % QUESTIONS.len()].to_owned(),
                        ));
                        let measured = ask(&address, &body(&talk, 32, true))
                            .map_err(|error| error.to_string())?;
                        talk.push(("assistant".to_owned(), measured.text.clone()));
                        done.push(measured);
                    }
                })
            })
            .collect();
        let mut runs = Vec::new();
        for worker in workers {
            runs.extend(worker.join().map_err(|_| "스레드가 공황에 빠졌습니다")??);
        }
        turns.push(runs);
    }
    let wall = start.elapsed().as_secs_f64();
    for (turn, runs) in turns.iter().enumerate() {
        let firsts: Vec<f64> = runs.iter().map(|run| run.first).collect();
        let (read, cached): (usize, usize) = runs
            .iter()
            .fold((0, 0), |sum, run| (sum.0 + run.prompt, sum.1 + run.cached));
        println!(
            "route {name} 차례 {} 첫글 {:.1} {:.1} 읽음 {} 캐시 {} 비율 {:.3}",
            turn + 1,
            quantile(&firsts, 0.5) * 1e3,
            quantile(&firsts, 0.95) * 1e3,
            read,
            cached,
            cached as f64 / read as f64,
        );
    }
    let all: Vec<f64> = turns.iter().flatten().map(|run| run.first).collect();
    let served: Vec<String> = router.served().iter().map(usize::to_string).collect();
    println!(
        "route {name} 전체 첫글 {:.1} {:.1} 걸린시간 {wall:.2} 서버마다 {}",
        quantile(&all, 0.5) * 1e3,
        quantile(&all, 0.95) * 1e3,
        served.join("/"),
    );
    Ok(())
}

/// 서버를 새로 띄워 첫 요청을 보낸다.
fn cold(model: &str, warm: bool) -> Outcome<()> {
    let extra: &[&str] = if warm { &[] } else { &["--no-warm"] };
    let server = serve(model, 12, extra)?;
    let messages = [(
        "user".to_owned(),
        "Explain in a few sentences why the sky is blue.".to_owned(),
    )];
    let one = ask(&server.address, &body(&messages, 8, false))?;
    let two = ask(&server.address, &body(&messages, 8, false))?;
    println!(
        "cold {} 준비 {:.1} 첫요청 {:.1} 둘째요청 {:.1} 띄우고첫글까지 {:.1}",
        if warm { "데움" } else { "안데움" },
        server.ready * 1e3,
        one.first * 1e3,
        two.first * 1e3,
        (server.ready + one.first) * 1e3,
    );
    Ok(())
}

/// 서버에 캐시를 쓰거나 되살리게 하고, 응답의 `name` 수를 꺼낸다.
fn cache(address: &str, action: &str, key: &str) -> Outcome<(f64, f64)> {
    let start = Instant::now();
    let response = Client::connect(address)?.send(
        "POST",
        &format!("/cache/{action}"),
        "{\"name\":\"handoff\"}",
    )?;
    let seconds = start.elapsed().as_secs_f64();
    let root = json::parse(&response.text())?;
    let value = root
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| response.text())?;
    Ok((seconds, value))
}

/// 글을 읽는 서버와 답을 쓰는 서버를 따로 둔다.
fn split(model: &str) -> Outcome<()> {
    let directory = std::env::temp_dir().join(format!("badak_ch42_{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let path = directory.to_string_lossy().into_owned();
    let long = vec![
        ("system".to_owned(), manual(SHOPS[0], 96)),
        ("user".to_owned(), QUESTIONS[0].to_owned()),
    ];
    let story = [(
        "user".to_owned(),
        "Tell me a short story about a lighthouse keeper.".to_owned(),
    )];

    // 1. 한 서버가 다 한다(스레드 12개).
    {
        let whole = serve(model, 12, &["--context", "8192"])?;
        ask(&whole.address, &body(&story, 4, false))?;
        let alone = ask(&whole.address, &body(&long, 64, false))?;
        println!(
            "split 한서버 읽음 {} 첫글 {:.0} 전체 {:.0}",
            alone.prompt,
            alone.first * 1e3,
            alone.total * 1e3
        );
    }

    // 2. 읽는 서버(스레드 8개)와 쓰는 서버(스레드 4개).
    let reader = serve(model, 8, &["--context", "8192", "--cache-dir", &path])?;
    let writer = serve(model, 4, &["--context", "8192", "--cache-dir", &path])?;
    ask(&reader.address, &body(&story, 4, false))?;
    ask(&writer.address, &body(&story, 4, false))?;
    // 쓰는 서버가 혼자 쓸 때의 토큰 사이.
    let calm = ask(&writer.address, &body(&story, 192, false))?;

    // 쓰는 서버가 다른 답을 쓰는 동안, 읽는 서버가 긴 글을 읽는다.
    let busy = {
        let (address, story) = (writer.address.clone(), body(&story, 192, false));
        std::thread::spawn(move || ask(&address, &story).map_err(|error| error.to_string()))
    };
    std::thread::sleep(Duration::from_millis(300));
    let start = Instant::now();
    let read = ask(&reader.address, &body(&long, 1, false))?;
    let reading = start.elapsed().as_secs_f64();
    let (saving, bytes) = cache(&reader.address, "save", "bytes")?;
    let busy = busy.join().map_err(|_| "스레드가 공황에 빠졌습니다")??;

    // 캐시를 넘겨받아 답을 쓴다.
    let handoff = Instant::now();
    let (restoring, _) = cache(&writer.address, "restore", "tokens")?;
    let answer = ask(&writer.address, &body(&long, 64, true))?;
    let waited = reading + saving + restoring + answer.first;
    println!(
        "split 나눔 읽음 {} 읽기 {:.0} 저장 {:.1} 파일 {:.0} 복원 {:.1} 캐시 {} 넘긴뒤첫글 {:.1} 합친첫글 {:.0} 넘긴뒤전체 {:.0}",
        read.prompt,
        reading * 1e3,
        saving * 1e3,
        bytes,
        restoring * 1e3,
        answer.cached,
        answer.first * 1e3,
        waited * 1e3,
        handoff.elapsed().as_secs_f64() * 1e3,
    );
    for (name, run) in [("혼자쓸때", &calm), ("옆에서읽을때", &busy)] {
        println!(
            "split 쓰는서버 {name} 토큰사이 {:.2} {:.2} {:.1} 전체 {:.0}",
            quantile(&run.gaps, 0.5) * 1e3,
            quantile(&run.gaps, 0.95) * 1e3,
            quantile(&run.gaps, 1.0) * 1e3,
            run.total * 1e3,
        );
    }
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

fn main() -> Outcome<()> {
    let (mut part, mut model, mut policy) = (
        String::new(),
        "smollm2_135m".to_owned(),
        "round_robin".to_owned(),
    );
    let (mut backends, mut threads, mut keep, mut warm) = (4usize, 3usize, 4usize, true);
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--part" => part = value()?,
            "--model" => model = value()?,
            "--policy" => policy = value()?,
            "--backends" => backends = value()?.parse()?,
            "--threads" => threads = value()?.parse()?,
            "--keep" => keep = value()?.parse()?,
            "--warm" => warm = value()? == "yes",
            _ => return Err(format!("모르는 인자입니다: {flag}").into()),
        }
    }
    match part.as_str() {
        "route" => {
            let chosen = match policy.as_str() {
                "round_robin" => Policy::RoundRobin,
                "least_busy" => Policy::LeastBusy,
                "prefix_hash" => Policy::PrefixHash { bytes: 256 },
                "longest_prefix" => Policy::LongestPrefix { min: 256 },
                other => return Err(format!("모르는 방법입니다: {other}").into()),
            };
            route(&model, chosen, &policy, backends, threads, keep)
        }
        "cold" => cold(&model, warm),
        "split" => split(&model),
        _ => Err("--part에 route, cold, split 가운데 하나를 주십시오".into()),
    }
}
