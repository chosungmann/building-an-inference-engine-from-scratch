//! 40장: 요청이 여럿일 때 서버가 얼마나 해내는지 잰다.
//!
//! 실행: `cargo run --release -p badak_server --example ch40_bench -- \
//!        --part 이름 [--model 이름] [--mode 방법] [--chunk 토큰 수] \
//!        [--clients 개] [--requests 개] [--rate 초당 요청] [--address 주소:포트]`
//!
//! `--mode`는 `serial`(39장의 엔진), `continuous:8`(연속 배칭, 여덟까지),
//! `static:4:100`(정적 배칭, 넷씩, 100밀리초까지 기다림) 가운데 하나다.
//! `--address`를 주면 그 주소의 서버를 부르고 `--mode`는 보지 않는다.
//!
//! `--part`:
//! - `closed`: 클라이언트 `--clients`개가 쉬지 않고 요청을 보낸다.
//! - `open`: 요청이 초당 `--rate`개꼴로 제멋대로 도착한다.
//! - `long`: 한 요청이 답을 쓰는 동안 긴 글을 든 요청이 끼어든다.
//! - `shared`: 같은 지시문으로 시작하는 대화 여덟이 번갈아 온다.
//! - `deep`: 요청 하나가 긴 글을 읽고 답을 쓴다. 캐시의 모양만 다른 두
//!   방법(이어진 캐시, 페이지 캐시)을 견주는 데 쓴다.

use std::net::TcpListener;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use badak_formats::json::{self, Value};
use badak_models::models_dir;
use badak_server::client::Client;
use badak_server::engine::Engine;
use badak_server::scheduler::{Policy, Settings, pages_for};
use badak_server::server::Server;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const LIMIT: usize = 64;
const TOPICS: [&str; 16] = [
    "a lighthouse keeper",
    "how bread is made",
    "the moon",
    "a lost umbrella",
    "why leaves fall",
    "a quiet library",
    "how bridges stand",
    "a street musician",
    "the first snow",
    "a paper boat",
    "how clocks work",
    "a mountain village",
    "an old bicycle",
    "the smell of rain",
    "a night train",
    "how bees find flowers",
];

/// 39장의 벤치마크와 같은 긴 지시문.
fn manual(rules: usize) -> String {
    let mut text = String::from(
        "You are the support assistant of a small bookshop. Follow every rule below.\n",
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

fn body(messages: &[(&str, &str)], limit: usize, reuse: bool) -> String {
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
    /// 요청을 보낸 때부터 첫 글 조각까지.
    first: f64,
    /// 요청을 보낸 때부터 끝까지.
    total: f64,
    /// 글 조각과 다음 글 조각 사이의 시간들.
    gaps: Vec<f64>,
    written: usize,
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
        let count = |value: Option<&Value>| value.and_then(Value::as_usize);
        if let Some(usage) = root.get("usage") {
            measured.written = count(usage.get("completion_tokens")).unwrap_or(0);
            measured.prompt = count(usage.get("prompt_tokens")).unwrap_or(0);
            measured.cached = count(
                usage
                    .get("prompt_tokens_details")
                    .and_then(|details| details.get("cached_tokens")),
            )
            .unwrap_or(0);
        }
        // llama.cpp의 서버는 토큰 수를 여기에 적는다.
        if let Some(timings) = root.get("timings")
            && let (Some(cached), Some(read), Some(written)) = (
                count(timings.get("cache_n")),
                count(timings.get("prompt_n")),
                count(timings.get("predicted_n")),
            )
        {
            (measured.prompt, measured.cached, measured.written) = (cached + read, cached, written);
        }
    }
    Ok(measured)
}

/// 작은 것부터 늘어놓았을 때 `q`(0~1) 자리의 값.
fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// 응답들을 한 줄로 요약한다. 시간은 밀리초다.
fn report(name: &str, wall: f64, runs: &[Measured], state: &str) {
    let ms = |values: Vec<f64>, q: f64| quantile(&values, q) * 1e3;
    let firsts = || runs.iter().map(|run| run.first).collect::<Vec<_>>();
    let totals = || runs.iter().map(|run| run.total).collect::<Vec<_>>();
    let gaps = || {
        runs.iter()
            .flat_map(|run| run.gaps.iter().copied())
            .collect::<Vec<_>>()
    };
    let written: usize = runs.iter().map(|run| run.written).sum();
    println!(
        "{name} 요청 {} 걸린시간 {:.2} 처리량 {:.1} 첫글 {:.1} {:.1} 토큰사이 {:.2} {:.2} {:.1} 전체 {:.0} {:.0} {state}",
        runs.len(),
        wall,
        written as f64 / wall,
        ms(firsts(), 0.5),
        ms(firsts(), 0.95),
        ms(gaps(), 0.5),
        ms(gaps(), 0.95),
        ms(gaps(), 1.0),
        ms(totals(), 0.5),
        ms(totals(), 0.95),
    );
}

/// 서버가 알려 주는 숫자들을 한 줄로.
fn state(address: &str, server: bool) -> String {
    if !server {
        return String::new();
    }
    let text = Client::connect(address)
        .and_then(|mut client| client.send("GET", "/cache", ""))
        .map(|response| response.text())
        .unwrap_or_default();
    let Ok(root) = json::parse(&text) else {
        return String::new();
    };
    let number = |name: &str| root.get(name).and_then(Value::as_f64).unwrap_or(0.0);
    if number("steps") == 0.0 {
        return String::new();
    }
    format!(
        "묶음 {:.2} 페이지 {:.0} 가장많을때 {:.0} 페이지크기 {:.0}",
        number("rows") / number("steps"),
        number("pages"),
        number("peak_pages"),
        number("page_bytes")
    )
}

fn question(index: usize) -> String {
    format!(
        "Tell me a short story about {}.",
        TOPICS[index % TOPICS.len()]
    )
}

/// 클라이언트 `clients`개가 저마다 요청 `each`개를 쉬지 않고 보낸다.
fn closed(address: &str, clients: usize, each: usize, scheduled: bool) -> Outcome<()> {
    // 한 번 돌려 데운다.
    ask(address, &body(&[("user", "Hello.")], 4, false))?;
    let barrier = Arc::new(Barrier::new(clients + 1));
    let handles: Vec<_> = (0..clients)
        .map(|client| {
            let (barrier, address) = (Arc::clone(&barrier), address.to_owned());
            std::thread::spawn(move || {
                barrier.wait();
                (0..each)
                    .map(|round| {
                        let text = question(client + round * clients);
                        ask(&address, &body(&[("user", &text)], LIMIT, false))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
        })
        .collect();
    barrier.wait();
    let start = Instant::now();
    let mut runs = Vec::new();
    for handle in handles {
        runs.extend(handle.join().map_err(|_| "스레드가 공황에 빠졌습니다")??);
    }
    report(
        &format!("closed 클라이언트 {clients}"),
        start.elapsed().as_secs_f64(),
        &runs,
        &state(address, scheduled),
    );
    Ok(())
}

/// 요청이 초당 `rate`개꼴로, 서로 상관없이 도착한다(푸아송 도착).
fn open(address: &str, rate: f64, count: usize, scheduled: bool) -> Outcome<()> {
    ask(address, &body(&[("user", "Hello.")], 4, false))?;
    // 도착 시각을 미리 정한다. 시드가 같으면 늘 같은 시각이다.
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut uniform = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut at = 0.0;
    let arrivals: Vec<f64> = (0..count)
        .map(|_| {
            at += -(1.0 - uniform()).ln() / rate;
            at
        })
        .collect();
    let start = Instant::now();
    let handles: Vec<_> = arrivals
        .iter()
        .enumerate()
        .map(|(index, &arrival)| {
            let address = address.to_owned();
            std::thread::spawn(move || {
                let wait = Duration::from_secs_f64(arrival).saturating_sub(start.elapsed());
                std::thread::sleep(wait);
                ask(&address, &body(&[("user", &question(index))], LIMIT, false))
                    .map_err(|error| error.to_string())
            })
        })
        .collect();
    let mut runs = Vec::new();
    for handle in handles {
        runs.push(handle.join().map_err(|_| "스레드가 공황에 빠졌습니다")??);
    }
    report(
        &format!("open 초당 {rate}"),
        start.elapsed().as_secs_f64(),
        &runs,
        &state(address, scheduled),
    );
    Ok(())
}

/// 한 요청이 답을 쓰는 동안, 긴 글을 든 요청이 끼어든다.
fn long(address: &str, scheduled: bool) -> Outcome<()> {
    ask(address, &body(&[("user", "Hello.")], 4, false))?;
    let writer = {
        let address = address.to_owned();
        std::thread::spawn(move || {
            ask(&address, &body(&[("user", &question(0))], 192, false))
                .map_err(|error| error.to_string())
        })
    };
    // 답을 쓰기 시작한 뒤에 긴 글을 보낸다.
    std::thread::sleep(Duration::from_millis(300));
    let document = manual(96);
    let reader = ask(
        address,
        &body(
            &[
                ("system", &document),
                ("user", "Hello, can I return a book?"),
            ],
            8,
            false,
        ),
    )?;
    let writer = writer.join().map_err(|_| "스레드가 공황에 빠졌습니다")??;
    println!(
        "long 쓰던요청 토큰사이 {:.2} {:.2} {:.1} 전체 {:.0} 씀 {} 긴요청 읽음 {} 첫글 {:.0} {}",
        quantile(&writer.gaps, 0.5) * 1e3,
        quantile(&writer.gaps, 0.95) * 1e3,
        quantile(&writer.gaps, 1.0) * 1e3,
        writer.total * 1e3,
        writer.written,
        reader.prompt,
        reader.first * 1e3,
        state(address, scheduled),
    );
    Ok(())
}

/// 같은 지시문으로 시작하는 대화 여덟이 세 차례씩, 번갈아 온다.
fn shared(address: &str, scheduled: bool) -> Outcome<()> {
    let system = manual(24);
    let mut talks: Vec<Vec<(String, String)>> = (0..8)
        .map(|_| vec![("system".to_owned(), system.clone())])
        .collect();
    let start = Instant::now();
    for turn in 0..3 {
        let mut runs = Vec::new();
        for (who, talk) in talks.iter_mut().enumerate() {
            talk.push(("user".to_owned(), question(who + turn * 8)));
            let messages: Vec<(&str, &str)> = talk
                .iter()
                .map(|(role, content)| (role.as_str(), content.as_str()))
                .collect();
            let measured = ask(address, &body(&messages, 32, true))?;
            talk.push(("assistant".to_owned(), measured.text.clone()));
            runs.push(measured);
        }
        let mean = |pick: fn(&Measured) -> usize| {
            runs.iter().map(pick).sum::<usize>() as f64 / runs.len() as f64
        };
        println!(
            "shared 차례 {} 읽음 {:.0} 캐시 {:.0} 첫글 {:.1} {}",
            turn + 1,
            mean(|run| run.prompt),
            mean(|run| run.cached),
            quantile(&runs.iter().map(|run| run.first).collect::<Vec<_>>(), 0.5) * 1e3,
            state(address, scheduled),
        );
    }
    println!("shared 걸린시간 {:.2}", start.elapsed().as_secs_f64());
    Ok(())
}

/// 요청 하나가 긴 글(지시문 96줄)을 읽고 64토큰을 쓴다. 세 번 잰 가운데.
fn deep(address: &str, scheduled: bool) -> Outcome<()> {
    ask(address, &body(&[("user", "Hello.")], 4, false))?;
    let document = manual(96);
    let mut runs = Vec::new();
    for round in 0..3 {
        let text = question(round);
        runs.push(ask(
            address,
            &body(&[("system", &document), ("user", &text)], LIMIT, false),
        )?);
    }
    let gaps: Vec<f64> = runs.iter().map(|run| quantile(&run.gaps, 0.5)).collect();
    let firsts: Vec<f64> = runs.iter().map(|run| run.first).collect();
    println!(
        "deep 읽음 {} 첫글 {:.0} 토큰사이 {:.2} {}",
        runs[0].prompt,
        quantile(&firsts, 0.5) * 1e3,
        quantile(&gaps, 0.5) * 1e3,
        state(address, scheduled),
    );
    Ok(())
}

fn main() -> Outcome<()> {
    let (mut model, mut mode, mut address) = ("smollm2_135m".to_owned(), "serial".to_owned(), None);
    let (mut part, mut chunk, mut clients, mut requests, mut rate) =
        (String::new(), 64usize, 1usize, 4usize, 1.0f64);
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--model" => model = value()?,
            "--mode" => mode = value()?,
            "--address" => address = Some(value()?),
            "--part" => part = value()?,
            "--chunk" => chunk = value()?.parse()?,
            "--clients" => clients = value()?.parse()?,
            "--requests" => requests = value()?.parse()?,
            "--rate" => rate = value()?.parse()?,
            _ => return Err(format!("모르는 인자입니다: {flag}").into()),
        }
    }

    // 묶는 방법을 읽는다.
    let fields: Vec<&str> = mode.split(':').collect();
    let policy = match fields.as_slice() {
        ["serial"] => None,
        ["continuous", max] => Some(Policy::Continuous { max: max.parse()? }),
        ["static", size, wait] => Some(Policy::Static {
            size: size.parse()?,
            wait: Duration::from_millis(wait.parse()?),
        }),
        _ => return Err(format!("모르는 방법입니다: {mode}").into()),
    };
    let external = address.is_some();
    let address = match address {
        Some(address) => address,
        None => {
            let mut engine = Engine::load(models_dir().join(&model))?;
            engine.context = 8192;
            let mut server = Server::new(engine, None);
            if let Some(policy) = policy {
                let slots = match policy {
                    Policy::Continuous { max } => max,
                    Policy::Static { size, .. } => size,
                };
                server = server.with_scheduler(Settings {
                    policy,
                    chunk,
                    pages: pages_for(8192, slots.max(8) + 2),
                    keep: 8,
                });
            }
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?.to_string();
            let server = Arc::new(server);
            std::thread::spawn(move || server.serve(&listener));
            address
        }
    };
    let scheduled = policy.is_some() && !external;
    match part.as_str() {
        "closed" => closed(&address, clients, requests, scheduled),
        "open" => open(&address, rate, requests, scheduled),
        "long" => long(&address, scheduled),
        "shared" => shared(&address, scheduled),
        "deep" => deep(&address, scheduled),
        _ => Err("--part에 closed, open, long, shared, deep 가운데 하나를 주십시오".into()),
    }
}
