//! 43장: 언어 모델 하나를 읽어 토큰을 쓰는 가장 작은 프로그램.
//!
//! 실행: `cargo run --release -p badak_models --example ch43_llm -- \
//!        [모델] [--tokens 개] [--seconds 초] [--hold 초]`
//!
//! 가중치를 메모리에 두는 방식은 20장의 `BADAK_WEIGHTS`(`stored`, `f32`,
//! `q8_0`, `q4_0`)를 따른다. `--seconds`를 주면 그 시간 동안 쉬지 않고
//! 쓰면서 10초마다 속도를 적는다. 오래 돌릴 때 속도가 떨어지는지를 본다.

use std::time::{Duration, Instant};

use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::generate;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let since = || start.elapsed().as_secs_f64() * 1e3;
    let (mut name, mut tokens, mut seconds, mut hold) =
        ("smollm2_135m".to_owned(), 64usize, 0.0, 0.0);
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or(format!("{argument} 뒤에 값이 없습니다"))
        };
        match argument.as_str() {
            "--tokens" => tokens = value()?.parse()?,
            "--seconds" => seconds = value()?.parse()?,
            "--hold" => hold = value()?.parse()?,
            _ => name = argument,
        }
    }
    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let loaded = since();
    let mut prompt = tokenizer.prefix().to_vec();
    prompt.extend(tokenizer.encode("Once upon a time, in a small village by the sea,", false));

    let mut first = None;
    let mut cache = KvCache::new(&model.config);
    let begin = Instant::now();
    let written = generate(&model, &prompt, tokens, &mut cache, |_| {
        first.get_or_insert_with(since);
    })?;
    let rate = (written.len().max(2) - 1) as f64
        / (begin.elapsed().as_secs_f64() - (first.unwrap_or(0.0) - loaded) / 1e3).max(1e-9);
    let weights = std::env::var("BADAK_WEIGHTS").unwrap_or_else(|_| "stored".into());
    println!(
        "llm {name} {weights} 읽기 {loaded:.1} 첫토큰 {:.1} 토큰 {} 속도 {rate:.1}",
        first.unwrap_or(0.0),
        written.len()
    );

    // 오래 돌리기. 글이 한도 없이 길어지지 않게 256토큰마다 처음부터 다시 쓴다.
    let run = Instant::now();
    let mut window = 0;
    while run.elapsed().as_secs_f64() < seconds {
        let (mut count, begin) = (0usize, Instant::now());
        while begin.elapsed() < Duration::from_secs(10) {
            let mut cache = KvCache::new(&model.config);
            count += generate(&model, &prompt, 256, &mut cache, |_| {})?.len();
        }
        window += 1;
        println!(
            "sustained {name} {weights} {:>4.0}초 속도 {:.1}",
            f64::from(window) * 10.0,
            count as f64 / begin.elapsed().as_secs_f64()
        );
    }
    if hold > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(hold));
    }
    Ok(())
}
