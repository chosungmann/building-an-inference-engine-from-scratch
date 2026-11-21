//! 40장: 요청을 묶어 처리해도 답이 한 번에 하나씩 처리한 것과 같은지 본다.
//!
//! 실행: `cargo run --release -p badak_server --example ch40_verify`
//!
//! `models/smollm2_135m`이 있어야 한다.

use std::net::TcpListener;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use badak_formats::json::{self, Value};
use badak_models::models_dir;
use badak_models::tokenizer::chat::Message;
use badak_server::client::request;
use badak_server::engine::{Engine, Options};
use badak_server::scheduler::{Policy, Settings};
use badak_server::server::Server;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const SYSTEM: &str = "You are a helpful assistant. Answer briefly and plainly. Never guess. \
                      When a question has more than one answer, give the most common one.";
const QUESTIONS: [(&str, usize); 6] = [
    ("What is the capital of France?", 16),
    ("Name three colors of the rainbow.", 24),
    ("Write one sentence about the sea.", 40),
    ("What is two plus two?", 8),
    ("Explain what a river is.", 48),
    ("Say hello in Spanish.", 12),
];

fn body(question: &str, limit: usize) -> String {
    format!(
        "{{\"messages\":[{{\"role\":\"system\",\"content\":{}}},{{\"role\":\"user\",\"content\":{}}}],\"max_tokens\":{limit},\"temperature\":0}}",
        json::quote(SYSTEM),
        json::quote(question)
    )
}

/// 응답에서 답과 캐시에서 온 토큰의 수를 꺼낸다.
fn answer(text: &str) -> Outcome<(String, usize)> {
    let root = json::parse(text)?;
    let content = root
        .get("choices")
        .and_then(Value::as_array)
        .ok_or("choices")?[0]
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or("content")?
        .to_owned();
    let cached = root
        .get("usage")
        .and_then(|usage| usage.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_usize)
        .ok_or("cached_tokens")?;
    Ok((content, cached))
}

fn number(text: &str, name: &str) -> Outcome<usize> {
    Ok(json::parse(text)?
        .get(name)
        .and_then(Value::as_usize)
        .ok_or(name.to_owned())?)
}

fn check(name: &str, passed: bool) -> Outcome<()> {
    println!("{} {name}", if passed { "같음  " } else { "다름  " });
    if passed {
        Ok(())
    } else {
        Err(format!("{name}: 어긋났습니다").into())
    }
}

/// 서버를 띄우고 주소를 돌려준다.
fn serve(directory: &std::path::Path, settings: Settings) -> Outcome<String> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?.to_string();
    let server = Arc::new(Server::new(Engine::load(directory)?, None).with_scheduler(settings));
    std::thread::spawn(move || server.serve(&listener));
    Ok(address)
}

/// 여섯 요청을 같은 순간에 보내고 답들을 받는다.
fn together(address: &str) -> Outcome<Vec<(String, usize)>> {
    let barrier = Arc::new(Barrier::new(QUESTIONS.len()));
    let handles: Vec<_> = QUESTIONS
        .iter()
        .map(|&(question, limit)| {
            let (barrier, address) = (Arc::clone(&barrier), address.to_owned());
            std::thread::spawn(move || {
                barrier.wait();
                request(
                    &address,
                    "POST",
                    "/v1/chat/completions",
                    &body(question, limit),
                )
                .map(|response| response.text())
                .map_err(|error| error.to_string())
            })
        })
        .collect();
    let mut answers = Vec::new();
    for handle in handles {
        answers.push(answer(
            &handle.join().map_err(|_| "스레드가 공황에 빠졌습니다")??,
        )?);
    }
    Ok(answers)
}

fn main() -> Outcome<()> {
    let directory = models_dir().join("smollm2_135m");
    if !directory.join("model.safetensors").exists() {
        println!("건너뜀: {}에 모델이 없습니다", directory.display());
        return Ok(());
    }

    // 정답: 39장의 엔진으로 하나씩, 캐시 없이 쓴 답.
    let engine = Engine::load(&directory)?;
    let mut expected = Vec::new();
    for (question, limit) in QUESTIONS {
        let messages = [
            Message::new("system", SYSTEM),
            Message::new("user", question),
        ];
        let options = Options {
            max_tokens: limit,
            temperature: 0.0,
            reuse: false,
            ..Options::default()
        };
        expected.push(engine.complete(&messages, &options, &mut |_| true)?.text);
    }
    for (index, text) in expected.iter().enumerate() {
        println!("{}. {text:?}", index + 1);
    }
    println!();

    // 1. 묶는 방법과 크기를 바꿔 가며, 여섯 요청을 같은 순간에 보낸다.
    let wait = Duration::from_millis(50);
    let cases = [
        (
            "연속 배칭, 넷까지, 한 스텝에 32토큰",
            Policy::Continuous { max: 4 },
            32,
        ),
        (
            "연속 배칭, 여섯까지, 한 스텝에 1토큰",
            Policy::Continuous { max: 6 },
            1,
        ),
        (
            "연속 배칭, 둘까지, 끊지 않고",
            Policy::Continuous { max: 2 },
            usize::MAX,
        ),
        ("정적 배칭, 넷씩", Policy::Static { size: 4, wait }, 64),
    ];
    for (name, policy, chunk) in cases {
        let address = serve(
            &directory,
            Settings {
                policy,
                chunk,
                pages: 512,
                keep: 8,
            },
        )?;
        let answers = together(&address)?;
        let same = answers
            .iter()
            .zip(&expected)
            .all(|((text, _), wanted)| text == wanted);
        check(&format!("{name}: 답 여섯이 하나씩 쓴 것과 같다"), same)?;
        let state = request(&address, "GET", "/cache", "")?.text();
        let (steps, rows) = (number(&state, "steps")?, number(&state, "rows")?);
        println!(
            "        순전파 {steps}번에 토큰 {rows}개 (한 번에 평균 {:.1}개)",
            rows as f64 / steps as f64
        );
    }

    // 2. 같은 지시문으로 시작하는 요청들은 그 페이지를 함께 쓴다.
    let settings = Settings {
        policy: Policy::Continuous { max: 4 },
        chunk: 64,
        pages: 512,
        keep: 8,
    };
    let address = serve(&directory, settings)?;
    let (question, limit) = QUESTIONS[0];
    let (_, cached) = answer(
        &request(
            &address,
            "POST",
            "/v1/chat/completions",
            &body(question, limit),
        )?
        .text(),
    )?;
    check("첫 요청은 캐시에서 온 것이 없다", cached == 0)?;
    let alone = number(&request(&address, "GET", "/cache", "")?.text(), "pages")?;
    let mut shared = Vec::new();
    for (question, limit) in &QUESTIONS[1..] {
        let response = request(
            &address,
            "POST",
            "/v1/chat/completions",
            &body(question, *limit),
        )?;
        shared.push(answer(&response.text())?);
    }
    let same = shared
        .iter()
        .zip(&expected[1..])
        .all(|((text, _), wanted)| text == wanted);
    check("앞부분을 빌려 쓴 답 다섯이 하나씩 쓴 것과 같다", same)?;
    let least = shared.iter().map(|(_, cached)| *cached).min().unwrap_or(0);
    check("다섯 모두 지시문만큼을 캐시에서 얻었다", least >= 30)?;
    let state = request(&address, "GET", "/cache", "")?.text();
    let (pages, idle) = (number(&state, "pages")?, number(&state, "idle")?);
    println!(
        "        남겨 둔 캐시 {idle}개가 페이지 {pages}장을 쓴다. 첫 요청 하나가 {alone}장이었다"
    );
    check(
        "여섯 캐시가 여섯 배보다 적은 페이지를 쓴다",
        idle == 6 && pages < 6 * alone,
    )?;

    // 3. 번갈아 오는 두 대화가 서로의 캐시를 밀어내지 않는다(39.7절).
    let mut talks = [
        vec![("system", SYSTEM.to_owned())],
        vec![("system", SYSTEM.to_owned())],
    ];
    // 차례마다 두 대화가 하나씩 묻는다.
    let turns = [
        ["Tell me about cats.", "Tell me about tea."],
        ["And dogs?", "And coffee?"],
    ];
    let mut reused = Vec::new();
    for questions in turns {
        for (talk, question) in talks.iter_mut().zip(questions) {
            talk.push(("user", question.to_owned()));
            let messages: Vec<String> = talk
                .iter()
                .map(|(role, content)| {
                    format!(
                        "{{\"role\":\"{role}\",\"content\":{}}}",
                        json::quote(content)
                    )
                })
                .collect();
            let text = format!(
                "{{\"messages\":[{}],\"max_tokens\":24,\"temperature\":0}}",
                messages.join(",")
            );
            let root = request(&address, "POST", "/v1/chat/completions", &text)?.text();
            let (content, cached) = answer(&root)?;
            let prompt = json::parse(&root)?
                .get("usage")
                .and_then(|usage| usage.get("prompt_tokens"))
                .and_then(Value::as_usize)
                .ok_or("prompt_tokens")?;
            reused.push((prompt, cached));
            talk.push(("assistant", content));
        }
    }
    println!("        (읽은 토큰, 캐시에서 온 토큰): {reused:?}");
    let second_turns = &reused[2..];
    check(
        "둘째 차례에는 두 대화 모두 제 앞 차례 전체를 캐시에서 얻었다",
        second_turns
            .iter()
            .zip(&reused[..2])
            .all(|((_, cached), (before, _))| cached >= before),
    )?;

    // 4. 페이지가 모자라면 남겨 둔 캐시를 버리고 계속 돈다.
    let tight = Settings {
        policy: Policy::Continuous { max: 2 },
        chunk: 64,
        pages: 14,
        keep: 8,
    };
    let address = serve(&directory, tight)?;
    let mut answers = Vec::new();
    for (question, limit) in QUESTIONS {
        let response = request(
            &address,
            "POST",
            "/v1/chat/completions",
            &body(question, limit),
        )?;
        answers.push(answer(&response.text())?);
    }
    let same = answers
        .iter()
        .zip(&expected)
        .all(|((text, _), wanted)| text == wanted);
    let state = request(&address, "GET", "/cache", "")?.text();
    println!(
        "        페이지 한도 14장, 가장 많이 쓴 때 {}장, 남겨 둔 캐시 {}개",
        number(&state, "peak_pages")?,
        number(&state, "idle")?
    );
    check(
        "페이지가 모자라도 답이 같다",
        same && number(&state, "peak_pages")? <= 14,
    )?;

    println!("\n모두 맞았습니다.");
    Ok(())
}
