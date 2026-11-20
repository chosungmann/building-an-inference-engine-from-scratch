//! 47장: 기기의 작은 모델이 먼저 답하고, 자신 없을 때만 큰 모델에 넘긴다.
//!
//! 실행: `cargo run --release -p badak_models --example ch47_hybrid -- \
//!        [기기 모델] [서버 모델] [왕복 시간 ms]`
//!
//! 기본값은 `smollm2_360m`, `smollm2_1_7b`, 100이다. 질문은
//! `reference/ch47_questions.tsv`에 있다. 두 모델 모두 이 기계에서 돈다.
//! "서버"는 흉내다. 서버 모델이 답하는 시간은 잰 것이고, 거기에 더하는
//! 네트워크 왕복 시간은 **지어낸 숫자**다.

use std::time::Instant;

use badak_core::prob::softmax;
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;
use badak_models::tokenizer::chat::{ChatFormat, Message};

/// 답 하나에 쓸 토큰의 한도.
const LIMIT: usize = 24;
/// SmolLM2가 학습할 때 쓴 기본 지시문. 모델의 대화 틀(`tokenizer_config.json`의
/// `chat_template`)에 적혀 있고, 대화에 시스템의 말이 없으면 이것을 넣는다.
const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// 질문 하나에 모델이 낸 것.
struct Answer {
    text: String,
    correct: bool,
    /// 쓴 토큰들에 모델이 준 확률의 로그의 평균. 0에 가까울수록 자신 있다.
    mean_log: f64,
    /// 그 가운데 가장 낮았던 확률.
    lowest: f64,
    /// 답을 다 쓰기까지의 시간(ms).
    total: f64,
}

/// 답에 맞는 말이 들어 있는지. 한 낱말짜리 답은 낱말 단위로 견준다. "O"가
/// "of"에 걸리지 않게 하려는 것이다.
fn matches(text: &str, accepted: &[String]) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    accepted.iter().any(|answer| {
        let answer = answer.to_lowercase();
        if answer.contains(' ') || answer.contains('-') {
            lower.contains(&answer)
        } else {
            words.contains(&answer.as_str())
        }
    })
}

// ANCHOR: confidence
/// 질문에 답하면서, 고른 토큰마다 모델이 준 확률을 모은다.
fn answer(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    question: &str,
    accepted: &[String],
) -> Result<Answer, Box<dyn std::error::Error>> {
    let content = format!("{question} Answer in a few words.");
    let mut messages = vec![Message::new("user", &content)];
    // 환경 변수 CH47_SYSTEM=0 이면 지시문을 뺀다. 빼면 어떻게 되는지 보려는 것이다.
    if std::env::var("CH47_SYSTEM").as_deref() != Ok("0") {
        messages.insert(0, Message::new("system", SYSTEM));
    }
    let prompt = ChatFormat::ChatMl.encode(tokenizer, &messages, true);
    let started = Instant::now();
    let mut cache = KvCache::new(&model.config);
    let mut input = prompt;
    let (mut ids, mut logs) = (Vec::new(), Vec::new());
    while ids.len() < LIMIT {
        let logits = model.forward_cached(&input, &mut cache)?;
        let probabilities = softmax(logits.as_slice::<f32>()?);
        let mut next = 0;
        for (index, &probability) in probabilities.iter().enumerate() {
            if probability > probabilities[next] {
                next = index;
            }
        }
        if model.config.eos_token_ids.contains(&(next as u32)) {
            break;
        }
        // 고른 토큰의 확률. 모델이 그 토큰을 얼마나 믿었는가.
        logs.push(f64::from(probabilities[next]).ln());
        ids.push(next as u32);
        input = vec![next as u32];
    }
    let text = tokenizer.decode(&ids);
    Ok(Answer {
        correct: matches(&text, accepted),
        mean_log: logs.iter().sum::<f64>() / logs.len().max(1) as f64,
        lowest: logs.iter().copied().fold(0.0, f64::min).exp(),
        total: started.elapsed().as_secs_f64() * 1e3,
        text: text.replace('\n', " "),
    })
}
// ANCHOR_END: confidence

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let device_name = arguments.next().unwrap_or_else(|| "smollm2_360m".into());
    let server_name = arguments.next().unwrap_or_else(|| "smollm2_1_7b".into());
    let round_trip: f64 = arguments.next().map_or(Ok(100.0), |text| text.parse())?;
    let file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/ch47_questions.tsv");
    let questions: Vec<(String, Vec<String>)> = std::fs::read_to_string(file)?
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains('\t'))
        .map(|line| {
            let (question, answers) = line.split_once('\t').expect("탭이 있는 줄만 골랐다");
            (
                question.to_owned(),
                answers.split('|').map(str::to_owned).collect(),
            )
        })
        .collect();

    let mut all = Vec::new();
    for name in [&device_name, &server_name] {
        let directory = models_dir().join(name);
        let format = match std::env::var("CH47_WEIGHTS").as_deref() {
            Ok("stored") => WeightFormat::Stored,
            Ok("q8_0") => WeightFormat::Q8_0,
            _ => WeightFormat::Q4_0,
        };
        let model = LlamaModel::load_as(&directory, format)?;
        let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
        let mut answers = Vec::new();
        for (index, (question, accepted)) in questions.iter().enumerate() {
            let one = answer(&model, &tokenizer, question, accepted)?;
            println!(
                "answer {name} {index} 맞음 {} 평균로그 {:.3} 가장낮은확률 {:.3} 전체 {:.0} 답 {}",
                u8::from(one.correct),
                one.mean_log,
                one.lowest,
                one.total,
                one.text
            );
            answers.push(one);
        }
        all.push(answers);
    }
    let (device, server) = (&all[0], &all[1]);
    let count = questions.len() as f64;
    let share =
        |answers: &[Answer]| answers.iter().filter(|one| one.correct).count() as f64 / count;
    println!(
        "summary {device_name} 맞음 {:.3} 평균시간 {:.0} 맞은답의평균로그 {:.3} 틀린답의평균로그 {:.3}",
        share(device),
        mean(device.iter().map(|one| one.total)),
        mean(
            device
                .iter()
                .filter(|one| one.correct)
                .map(|one| one.mean_log)
        ),
        mean(
            device
                .iter()
                .filter(|one| !one.correct)
                .map(|one| one.mean_log)
        )
    );
    println!(
        "summary {server_name} 맞음 {:.3} 평균시간 {:.0} (왕복 {round_trip:.0}ms를 더하면 {:.0})",
        share(server),
        mean(server.iter().map(|one| one.total)),
        mean(server.iter().map(|one| one.total)) + round_trip
    );

    // ANCHOR: cascade
    // 기기의 답을 자신 없는 것부터 줄 세우고, 앞에서부터 몇 %를 서버에 넘길지를
    // 바꿔 가며 본다. 넘긴 질문은 기기가 쓴 시간에 왕복과 서버의 시간이 더해진다.
    for (signal, name) in [(0, "평균로그"), (1, "가장낮은확률")] {
        let mut order: Vec<usize> = (0..device.len()).collect();
        let key = |index: usize| {
            if signal == 0 {
                device[index].mean_log
            } else {
                device[index].lowest
            }
        };
        order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
        for percent in (0..=100).step_by(10) {
            let sent = device.len() * percent / 100;
            let (mut correct, mut time) = (0, 0.0);
            for (rank, &index) in order.iter().enumerate() {
                if rank < sent {
                    correct += usize::from(server[index].correct);
                    time += device[index].total + round_trip + server[index].total;
                } else {
                    correct += usize::from(device[index].correct);
                    time += device[index].total;
                }
            }
            // 넘긴 것 가운데 기기가 틀렸던 것. 넘길 만했던 질문이다.
            let needed = order[..sent]
                .iter()
                .filter(|&&index| !device[index].correct)
                .count();
            println!(
                "cascade {name} 넘긴비율 {percent} 맞음 {:.3} 평균시간 {:.0} 넘긴것 {sent} 그가운데기기가틀린것 {needed}",
                correct as f64 / count,
                time / count
            );
        }
    }
    // ANCHOR_END: cascade
    Ok(())
}
