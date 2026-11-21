//! 11장: 글을 이어 쓴다.
//!
//! 실행: `cargo run --release -p badak_models --example ch11_generate -- [모델 디렉터리] [글] [새 토큰 수]`
//!
//! 글 앞에 `chat:`을 붙이면 그 뒤를 사용자의 말로 보고 대화 형식으로 감싼다.
//! 예: `... -- llama32_1b "chat:프랑스의 수도는 어디인가요?" 40`

use std::io::Write;
use std::time::Instant;

use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::chat::{ChatFormat, Message};
use badak_models::tokenizer::{StreamDecoder, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_new_tokens: usize = arguments.next().map_or(Ok(40), |count| count.parse())?;

    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;

    // ANCHOR: prompt
    // 대화라면 모델이 학습한 형식으로 감싼다. 형식은 단어 사전에 어떤 특수
    // 토큰이 있는지로 알아본다.
    let text = match prompt.strip_prefix("chat:") {
        Some(question) => {
            // 틀의 뼈대만 쓴다. SmolLM2의 틀에 있는 기본 지시문은 넣지 않는다.
            // 그것까지 넣는 것이 `ChatTemplate`이다(47.3절).
            let format = ChatFormat::detect(&tokenizer);
            format.render(&[Message::new("user", question)], true)
        }
        None => prompt.clone(),
    };
    let mut ids = tokenizer.prefix().to_vec();
    ids.extend(tokenizer.encode(&text, true));
    // ANCHOR_END: prompt

    println!("모델 {name}, 프롬프트의 토큰 {}개", ids.len());
    println!("{prompt}");
    println!("---");

    // ANCHOR: stream
    let mut cache = KvCache::new(&model.config);
    let mut decoder = StreamDecoder::default();
    let start = Instant::now();
    let (mut first_token, mut last_token) = (None, start.elapsed());
    let generated = generate(&model, &ids, max_new_tokens, &mut cache, |token| {
        // 첫 토큰과 마지막 토큰이 나온 때를 적어 둔다.
        last_token = start.elapsed();
        first_token.get_or_insert(last_token);
        // 토큰이 나오는 대로 글로 바꿔 내보낸다.
        print!("{}", decoder.push(&tokenizer, token));
        std::io::stdout().flush().ok();
    })?;
    println!("{}", decoder.finish());
    // ANCHOR_END: stream

    let total = start.elapsed().as_secs_f64();
    let first = first_token.map_or(total, |time| time.as_secs_f64());
    println!("---");
    println!("{:<30} {:>8} 개", "새로 쓴 토큰", generated.len());
    println!("{:<30} {:>8.2} 초", "첫 토큰까지", first);
    if generated.len() > 1 {
        let rest = (last_token.as_secs_f64() - first) / (generated.len() - 1) as f64;
        println!("{:<30} {:>8.2} 초", "그 뒤 토큰 하나마다", rest);
        println!("{:<30} {:>8.2} 개", "그 뒤 1초에 쓰는 토큰", 1.0 / rest);
    }
    println!("{:<30} {:>8.2} 초", "전체", total);
    println!(
        "{:<30} {:>8.2} MB (토큰 {}개)",
        "KV 캐시",
        cache.size_in_bytes() as f64 / 1e6,
        cache.len()
    );
    Ok(())
}
