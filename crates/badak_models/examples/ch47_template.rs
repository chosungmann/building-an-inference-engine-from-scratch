//! 47장: 대화의 틀에 적힌 줄을 넣을 때와 뺄 때, 답이 얼마나 달라지는가.
//!
//! 실행: `cargo run --release -p badak_models --example ch47_template -- [모델]`
//!
//! 기본값은 `llama32_1b`다. Llama 3.2의 틀은 시스템의 말이 있든 없든 맨
//! 앞에 시스템의 자리를 만들고 날짜 두 줄을 적는다. 우리의
//! `ChatFormat::Llama3`은 그 줄들을 적지 않는다. `reference/ch47_questions.tsv`의
//! 질문을 두 가지로 물어 맞은 수를 센다. 가중치는 파일에 든 그대로 쓴다.

use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::chat::{ChatFormat, Message};
use badak_models::tokenizer::{TokenId, Tokenizer};

/// 답 하나에 쓸 토큰의 한도.
const LIMIT: usize = 24;

// ANCHOR: dated
/// Llama 3.2의 틀이 내는 글. 틀에 적힌 그대로다(`tokenizer_config.json`의
/// `chat_template`). 틀은 "Today Date" 뒤에 **그날의 날짜**를 적는다. 여기서는
/// 틀이 날짜를 받지 못했을 때 쓰는 기본값(26 Jul 2024)으로 고정했다.
fn dated(question: &str) -> String {
    format!(
        "<|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n",
        question.trim()
    )
}
// ANCHOR_END: dated

/// 답에 맞는 말이 들어 있는지(`ch47_hybrid`와 같은 규칙).
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

/// 점수가 가장 높은 토큰을 골라 가며 답을 쓴다.
fn write(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: Vec<TokenId>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut cache = KvCache::new(&model.config);
    let (mut input, mut ids) = (prompt, Vec::new());
    while ids.len() < LIMIT {
        let logits = model.forward_cached(&input, &mut cache)?;
        let scores = logits.as_slice::<f32>()?;
        let mut next = 0;
        for (index, &score) in scores.iter().enumerate() {
            if score > scores[next] {
                next = index;
            }
        }
        if model.config.eos_token_ids.contains(&(next as u32)) {
            break;
        }
        ids.push(next as u32);
        input = vec![next as u32];
    }
    Ok(tokenizer.decode(&ids).replace('\n', " "))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".into());
    let directory = models_dir().join(&name);
    if !directory.join("model.safetensors").exists() {
        println!("건너뜀: {}에 모델이 없습니다", directory.display());
        return Ok(());
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let questions: Vec<(String, Vec<String>)> =
        std::fs::read_to_string(root.join("reference/ch47_questions.tsv"))?
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
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let encode = |text: &str| {
        let mut ids = tokenizer.prefix().to_vec();
        ids.extend(tokenizer.encode(text, true));
        ids
    };

    // 날짜를 넣은 글이 transformers가 틀을 실행해 낸 글과 같은지 먼저 본다.
    let answers = root.join("bench/ch47/chat_template_llama32.txt");
    if let Ok(expected) = std::fs::read_to_string(&answers) {
        let (text, ids) = expected
            .split_once('\t')
            .ok_or("정답 파일의 꼴이 다릅니다")?;
        let text = text.replace("\\n", "\n");
        let ids: Vec<TokenId> = ids
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let same = dated("Hi") == text.strip_prefix("<|begin_of_text|>").unwrap_or(&text)
            && encode(&dated("Hi")) == ids;
        println!(
            "check 날짜를 넣은 글과 토큰이 transformers의 것과 {}",
            if same { "같음" } else { "다름" }
        );
        if !same {
            return Err("틀을 옮긴 글이 정답과 다릅니다".into());
        }
    }

    let (mut bare_right, mut dated_right, mut same_text) = (0, 0, 0);
    for (index, (question, accepted)) in questions.iter().enumerate() {
        let content = format!("{question} Answer in a few words.");
        let bare = ChatFormat::Llama3.render(&[Message::new("user", &content)], true);
        let without = write(&model, &tokenizer, encode(&bare))?;
        let with = write(&model, &tokenizer, encode(&dated(&content)))?;
        let (a, b) = (matches(&without, accepted), matches(&with, accepted));
        bare_right += usize::from(a);
        dated_right += usize::from(b);
        same_text += usize::from(without == with);
        let mark = |right| if right { "O" } else { "X" };
        println!(
            "question {:>2} 뼈대만 {} 날짜와 {} | {question} | {without} | {with}",
            index + 1,
            mark(a),
            mark(b)
        );
    }
    let count = questions.len();
    println!(
        "summary {name} 질문 {count} 뼈대만 {bare_right} ({:.1}%) 날짜 두 줄과 {dated_right} ({:.1}%) 글이 같은 답 {same_text}",
        100.0 * bare_right as f64 / count as f64,
        100.0 * dated_right as f64 / count as f64
    );
    Ok(())
}
