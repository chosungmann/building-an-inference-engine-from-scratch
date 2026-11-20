//! 39장: 서버를 거치면 무엇이 얼마나 달라지는지 잰다.
//!
//! 실행: `cargo run --release -p badak_server --example ch39_bench -- \
//!        [--model 이름] [--address 주소:포트] [--llama] [--repeats 번] [--part 이름]`
//!
//! `--address`를 주지 않으면 서버를 이 프로그램 안에서 띄우고 진짜 TCP
//! 연결로 부른다. 주면 그 주소의 서버(이 책의 `badak serve`나 llama.cpp의
//! `llama-server`)를 부른다. 그때는 엔진을 직접 부르는 측정을 건너뛴다.
//! `--llama`는 그 서버가 llama.cpp의 것임을 알린다. 캐시를 파일에 쓰는
//! 길의 이름과 꼴이 다르기 때문이다.
//!
//! `--part`는 `overhead`, `turns`, `interleave`, `persist`, `queue` 가운데 하나다. 주지
//! 않으면 모두 잰다. `--part first`는 따로 불러야 한다. 방금 띄운 서버에
//! 보내는 첫 요청과 둘째 요청을 잰다.

use std::net::TcpListener;
use std::sync::{Arc, Barrier};

use badak_formats::json::{self, Value};
use badak_models::models_dir;
use badak_models::tokenizer::chat::Message;
use badak_server::client::{Client, Response};
use badak_server::engine::{Engine, Options};
use badak_server::server::Server;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const LIMIT: usize = 64;

/// 긴 지시문. 대화의 맨 앞에 놓여 차례마다 되풀이된다.
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

const QUESTIONS: [&str; 5] = [
    "Hello, can I return a book I bought last week?",
    "How long does shipping usually take?",
    "Do you sell gift cards?",
    "When are you open on Sundays?",
    "Can my school order thirty copies?",
];

fn body(messages: &[Message], limit: usize, stream: bool, reuse: bool) -> String {
    let messages: Vec<String> = messages
        .iter()
        .map(|message| {
            format!(
                "{{\"role\":{},\"content\":{}}}",
                json::quote(&message.role),
                json::quote(&message.content)
            )
        })
        .collect();
    // 조각으로 받을 때는 끝에 토큰의 수를 적은 조각도 달라고 한다.
    let options = if stream {
        ",\"stream_options\":{\"include_usage\":true}"
    } else {
        ""
    };
    format!(
        "{{\"model\":\"badak\",\"messages\":[{}],\"max_tokens\":{limit},\"temperature\":0,\"seed\":0,\"stream\":{stream}{options},\"cache_prompt\":{reuse}}}",
        messages.join(",")
    )
}

/// 받는 쪽에서 잰 응답 하나.
#[derive(Debug, Clone, Default)]
struct Measured {
    text: String,
    /// 요청을 보낸 때부터 첫 글 조각이 온 때까지(밀리초).
    first: f64,
    /// 요청을 보낸 때부터 응답이 끝난 때까지(밀리초).
    total: f64,
    prompt: usize,
    cached: Option<usize>,
    written: usize,
    /// 서버가 적어 보낸, 앞 요청을 기다린 시간(밀리초).
    queued: Option<f64>,
    /// 서버가 적어 보낸, 대화를 토큰으로 바꾼 시간(밀리초).
    tokenize: Option<f64>,
}

impl Measured {
    /// 첫 토큰 뒤의 속도(토큰/초).
    fn rate(&self) -> f64 {
        (self.written.max(2) - 1) as f64 / ((self.total - self.first).max(1e-6) / 1e3)
    }
}

fn usage(root: &Value, measured: &mut Measured) {
    let count = |value: Option<&Value>| value.and_then(Value::as_usize);
    if let Some(usage) = root.get("usage") {
        measured.prompt = count(usage.get("prompt_tokens")).unwrap_or(0);
        measured.written = count(usage.get("completion_tokens")).unwrap_or(0);
        measured.cached = count(
            usage
                .get("prompt_tokens_details")
                .and_then(|details| details.get("cached_tokens")),
        );
    }
    if let Some(timings) = root.get("timings") {
        measured.queued = timings.get("queued_ms").and_then(Value::as_f64);
        measured.tokenize = timings.get("tokenize_ms").and_then(Value::as_f64);
        // llama.cpp의 서버는 조각으로 보낼 때 토큰 수를 여기에만 적는다.
        // 캐시에서 온 것(cache_n), 새로 읽은 것(prompt_n), 쓴 것(predicted_n).
        if let (Some(cached), Some(read), Some(written)) = (
            count(timings.get("cache_n")),
            count(timings.get("prompt_n")),
            count(timings.get("predicted_n")),
        ) {
            measured.prompt = cached + read;
            measured.cached = Some(cached);
            measured.written = written;
        }
    }
}

fn measure(response: &Response, stream: bool) -> Outcome<Measured> {
    if response.status != 200 {
        return Err(format!("상태 {}: {}", response.status, response.text()).into());
    }
    let mut measured = Measured {
        total: response.total * 1e3,
        ..Measured::default()
    };
    if !stream {
        let root = json::parse(&response.text())?;
        measured.text = root
            .get("choices")
            .and_then(Value::as_array)
            .ok_or("choices")?[0]
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or("content")?
            .to_owned();
        measured.first = measured.total;
        usage(&root, &mut measured);
        return Ok(measured);
    }
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
        if !piece.is_empty() && measured.text.is_empty() {
            measured.first = arrived * 1e3;
        }
        measured.text.push_str(piece);
        usage(&root, &mut measured);
    }
    Ok(measured)
}

fn ask(
    address: &str,
    messages: &[Message],
    limit: usize,
    stream: bool,
    reuse: bool,
) -> Outcome<Measured> {
    let response = Client::connect(address)?.send(
        "POST",
        "/v1/chat/completions",
        &body(messages, limit, stream, reuse),
    )?;
    measure(&response, stream)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// 같은 요청을 엔진에 직접, 서버에 한 번에, 서버에 조각으로 보낸다.
fn overhead(address: &str, server: Option<&Server>, repeats: usize) -> Outcome<()> {
    let messages = [
        Message::new("system", "You are a helpful assistant."),
        Message::new("user", "Explain in a few sentences why the sky is blue."),
    ];
    let options = Options {
        max_tokens: LIMIT,
        temperature: 0.0,
        reuse: false,
        ..Options::default()
    };
    let (mut direct, mut whole, mut pieces) = (Vec::new(), Vec::new(), Vec::new());
    // 첫 실행은 버린다(22.12절).
    for round in 0..=repeats {
        if let Some(server) = server {
            // 엔진을 직접 부를 때도 부르는 쪽의 시계로 잰다. 토큰으로 바꾸는
            // 시간까지 든다.
            let start = std::time::Instant::now();
            let mut first = None;
            let done = server.engine.complete(&messages, &options, &mut |_| {
                first.get_or_insert_with(|| start.elapsed().as_secs_f64());
                true
            })?;
            let total = start.elapsed().as_secs_f64();
            if round > 0 {
                direct.push(Measured {
                    text: done.text,
                    first: first.unwrap_or(total) * 1e3,
                    total: total * 1e3,
                    prompt: done.prompt_tokens,
                    written: done.completion_tokens,
                    ..Measured::default()
                });
            }
        }
        let one = ask(address, &messages, LIMIT, false, false)?;
        let two = ask(address, &messages, LIMIT, true, false)?;
        if round > 0 {
            whole.push(one);
            pieces.push(two);
        }
    }
    for (name, runs) in [("직접", &direct), ("한번에", &whole), ("조각", &pieces)] {
        let Some(last) = runs.last() else { continue };
        // 한 번에 받으면 첫 글이 곧 전체다. 토큰마다의 속도를 알 수 없다.
        let rate = if name == "한번에" {
            "-".to_owned()
        } else {
            format!("{:.1}", median(runs.iter().map(Measured::rate).collect()))
        };
        println!(
            "overhead {name} 읽음 {} 씀 {} 첫글 {:.2} 전체 {:.2} 속도 {rate}",
            last.prompt,
            last.written,
            median(runs.iter().map(|run| run.first).collect()),
            median(runs.iter().map(|run| run.total).collect()),
        );
    }
    let same = direct
        .iter()
        .chain(&whole)
        .chain(&pieces)
        .all(|run| run.text == pieces[0].text);
    println!("overhead 답이모두같음 {same}");
    Ok(())
}

/// 긴 지시문 뒤에 대화를 다섯 차례 잇는다. 차례마다 접두사 캐시를 쓴
/// 것과 쓰지 않은 것을 잰다.
fn turns(address: &str, repeats: usize) -> Outcome<()> {
    let mut same = true;
    for round in 0..=repeats {
        // 지시문의 맨 앞을 차례마다 달리해 앞 차례의 캐시와 겹치지 않게 한다.
        let system = format!("Session {round}.\n{}", manual(24));
        let mut answers: Vec<Vec<String>> = Vec::new();
        for reuse in [true, false] {
            let mut messages = vec![Message::new("system", &system)];
            // 캐시를 비운다: 아무 상관 없는 요청 하나.
            ask(address, &[Message::new("user", "Reset.")], 1, true, false)?;
            let mut written = Vec::new();
            for (turn, question) in QUESTIONS.iter().enumerate() {
                messages.push(Message::new("user", question));
                let measured = ask(address, &messages, 32, true, reuse)?;
                if round > 0 {
                    println!(
                        "turns {} 차례 {} 읽음 {} 캐시 {} 씀 {} 첫글 {:.2} 전체 {:.2} 토큰화 {}",
                        if reuse { "캐시씀" } else { "캐시안씀" },
                        turn + 1,
                        measured.prompt,
                        measured
                            .cached
                            .map_or("-".to_owned(), |cached| cached.to_string()),
                        measured.written,
                        measured.first,
                        measured.total,
                        measured
                            .tokenize
                            .map_or("-".to_owned(), |time| format!("{time:.2}")),
                    );
                }
                messages.push(Message::new("assistant", &measured.text));
                written.push(measured.text);
            }
            answers.push(written);
        }
        same &= answers[0] == answers[1];
    }
    // 캐시를 쓰든 안 쓰든 같은 글을 읽었으니 같은 답을 써야 한다.
    println!("turns 답이모두같음 {same}");
    Ok(())
}

/// 같은 지시문으로 시작하는 두 대화가 번갈아 온다. 캐시는 하나뿐이다.
fn interleave(address: &str, repeats: usize) -> Outcome<()> {
    let other = [
        "Hi, do you have any rare books about ships?",
        "Is there a newsletter I can sign up for?",
        "Can I join a book club this month?",
    ];
    for round in 0..=repeats {
        let system = format!("Session {round}.\n{}", manual(24));
        let mut talks = [
            vec![Message::new("system", &system)],
            vec![Message::new("system", &system)],
        ];
        ask(address, &[Message::new("user", "Reset.")], 1, true, false)?;
        for turn in 0..3 {
            for (who, questions) in [&QUESTIONS[..3], &other[..]].iter().enumerate() {
                talks[who].push(Message::new("user", questions[turn]));
                let measured = ask(address, &talks[who], 32, true, true)?;
                if round > 0 {
                    println!(
                        "interleave 대화 {} 차례 {} 읽음 {} 캐시 {} 첫글 {:.2}",
                        ["가", "나"][who],
                        turn + 1,
                        measured.prompt,
                        measured
                            .cached
                            .map_or("-".to_owned(), |cached| cached.to_string()),
                        measured.first,
                    );
                }
                talks[who].push(Message::new("assistant", &measured.text));
            }
        }
    }
    Ok(())
}

/// 긴 글을 읽은 캐시를 파일에 썼다가 되살린다.
fn persist(address: &str, repeats: usize, llama: bool) -> Outcome<()> {
    // 길과 몸통, 그리고 응답에서 바이트 수와 걸린 시간이 적힌 자리.
    let (save, restore, bytes, save_ms, restore_ms) = if llama {
        (
            "/slots/0?action=save",
            "/slots/0?action=restore",
            "n_written",
            "save_ms",
            "restore_ms",
        )
    } else {
        ("/cache/save", "/cache/restore", "bytes", "ms", "ms")
    };
    for rules in [24, 96] {
        let messages = [
            Message::new("system", &manual(rules)),
            Message::new("user", QUESTIONS[0]),
        ];
        for round in 0..=repeats {
            let name = if llama {
                format!("{{\"filename\":\"bench_{rules}.bin\"}}")
            } else {
                format!("{{\"name\":\"bench_{rules}\"}}")
            };
            let cold = ask(address, &messages, 1, true, false)?;
            let saved = Client::connect(address)?.send("POST", save, &name)?;
            ask(address, &[Message::new("user", "Reset.")], 1, true, false)?;
            let restored = Client::connect(address)?.send("POST", restore, &name)?;
            let warm = ask(address, &messages, 1, true, true)?;
            // llama.cpp는 시간을 `timings` 아래에 적는다.
            let field = |response: &Response, key: &str| -> Outcome<f64> {
                let root = json::parse(&response.text())?;
                Ok(root
                    .get(key)
                    .or_else(|| root.get("timings").and_then(|timings| timings.get(key)))
                    .and_then(Value::as_f64)
                    .ok_or(format!("{key}: {}", response.text()))?)
            };
            if round > 0 {
                println!(
                    "persist 읽음 {} 처음부터 {:.2} 파일 {:.0} 저장 {:.2} 복원 {:.2} 복원요청 {:.2} 되살린뒤 {:.2} 캐시 {}",
                    cold.prompt,
                    cold.first,
                    field(&saved, bytes)?,
                    field(&saved, save_ms)?,
                    field(&restored, restore_ms)?,
                    restored.total * 1e3,
                    warm.first,
                    warm.cached.unwrap_or(0)
                );
            }
        }
    }
    Ok(())
}

/// 방금 띄운 서버의 첫 요청과, 같은 요청을 다시 보낸 것.
fn first(address: &str) -> Outcome<()> {
    let messages = [
        Message::new("system", "You are a helpful assistant."),
        Message::new("user", "Explain in a few sentences why the sky is blue."),
    ];
    let one = ask(address, &messages, 8, true, false)?;
    let two = ask(address, &messages, 8, true, false)?;
    println!("first 첫요청 {:.2} 둘째요청 {:.2}", one.first, two.first);
    Ok(())
}

/// 두 클라이언트가 같은 때에 요청을 보낸다.
fn queue(address: &str, repeats: usize) -> Outcome<()> {
    let prompts = [
        "Tell me a short story about a lighthouse keeper.",
        "Describe how bread is made, step by step.",
    ];
    for round in 0..=repeats {
        let alone = ask(
            address,
            &[Message::new("user", prompts[0])],
            LIMIT,
            true,
            false,
        )?;
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                let (barrier, address) = (Arc::clone(&barrier), address.to_owned());
                let messages = [Message::new("user", prompt)];
                std::thread::spawn(move || {
                    barrier.wait();
                    ask(&address, &messages, LIMIT, true, false).map_err(|error| error.to_string())
                })
            })
            .collect();
        let mut pair = Vec::new();
        for handle in handles {
            pair.push(handle.join().map_err(|_| "스레드가 공황에 빠졌습니다")??);
        }
        // 먼저 끝난 쪽을 앞에 둔다.
        pair.sort_by(|a, b| a.total.total_cmp(&b.total));
        if round > 0 {
            println!(
                "queue 혼자 첫글 {:.2} 전체 {:.2} 씀 {}",
                alone.first, alone.total, alone.written
            );
            for (name, one) in ["먼저", "나중"].iter().zip(&pair) {
                println!(
                    "queue {name} 기다림 {} 첫글 {:.2} 전체 {:.2} 씀 {}",
                    one.queued
                        .map_or("-".to_owned(), |queued| format!("{queued:.2}")),
                    one.first,
                    one.total,
                    one.written
                );
            }
        }
    }
    Ok(())
}

fn main() -> Outcome<()> {
    let (mut model, mut address, mut repeats, mut part) =
        ("smollm2_135m".to_owned(), None, 5usize, None);
    let mut llama = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().ok_or(format!("{flag} 뒤에 값이 없습니다"));
        match flag.as_str() {
            "--model" => model = value()?,
            "--address" => address = Some(value()?),
            "--llama" => llama = true,
            "--repeats" => repeats = value()?.parse()?,
            "--part" => part = Some(value()?),
            _ => return Err(format!("모르는 인자입니다: {flag}").into()),
        }
    }
    let wanted = |name: &str| part.as_deref().is_none_or(|part| part == name);

    // 서버를 안에서 띄우거나, 밖의 서버를 쓴다.
    let (address, server) = match address {
        Some(address) => (address, None),
        None => {
            let directory = models_dir().join(&model);
            let cache_dir = std::env::temp_dir().join(format!("badak_ch39_{}", std::process::id()));
            std::fs::create_dir_all(&cache_dir)?;
            let mut engine = Engine::load(&directory)?;
            engine.context = 16_384;
            let server = Arc::new(Server::new(engine, Some(cache_dir)));
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?.to_string();
            let background = Arc::clone(&server);
            std::thread::spawn(move || background.serve(&listener));
            (address, Some(server))
        }
    };
    if part.as_deref() == Some("first") {
        return first(&address);
    }
    if wanted("overhead") {
        overhead(&address, server.as_deref(), repeats)?;
    }
    if wanted("turns") {
        turns(&address, repeats)?;
    }
    if wanted("interleave") {
        interleave(&address, repeats)?;
    }
    if wanted("persist") {
        persist(&address, repeats, llama)?;
    }
    if wanted("queue") {
        queue(&address, repeats)?;
    }
    if let Some(server) = &server
        && let Some(directory) = &server.cache_dir
    {
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
