//! 13장: badak의 속도를 llama-bench와 같은 규칙으로 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch13_bench -- [모델 디렉터리] [되풀이 횟수]`
//!
//! 프롬프트 256토큰을 한꺼번에 처리하는 속도(pp256)와, 시작 토큰 하나에서
//! 출발해 토큰 32개를 하나씩 처리하는 속도(tg32)를 잰다. 토큰은 llama-bench처럼
//! 아무 번호나 쓴다. 어떤 토큰인지는 속도에 영향을 주지 않는다.

use std::time::Instant;

use badak_core::random::Pcg64;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

const PROMPT_TOKENS: usize = 256;
const NEW_TOKENS: usize = 32;

// ANCHOR: statistics
/// 평균과 표준편차. 표준편차는 llama-bench처럼 표본 표준편차(n - 1로 나눔)다.
fn mean_and_deviation(samples: &[f64]) -> (f64, f64) {
    let count = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / count;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, variance.sqrt())
}
// ANCHOR_END: statistics

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let repetitions: usize = arguments.next().map_or(Ok(3), |count| count.parse())?;
    let directory = models_dir().join(&name);

    let start = Instant::now();
    let model = LlamaModel::load(&directory)?;
    let load = start.elapsed().as_secs_f64();
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;

    // 시작 토큰 하나와 아무 토큰들.
    let mut rng = Pcg64::new(13);
    let first = tokenizer.prefix().first().copied().unwrap_or(0);
    let vocab = model.config.vocab_size as u64;
    let tokens: Vec<TokenId> = std::iter::once(first)
        .chain(std::iter::repeat_with(|| {
            (rng.next_u64() % vocab) as TokenId
        }))
        .take(PROMPT_TOKENS)
        .collect();

    // ANCHOR: measure
    let (mut prompt_speeds, mut generation_speeds) = (Vec::new(), Vec::new());
    for _ in 0..repetitions {
        // pp: 토큰 256개를 빈 캐시에 한꺼번에 넣는다.
        let mut cache = KvCache::new(&model.config);
        let start = Instant::now();
        model.forward_cached(&tokens, &mut cache)?;
        prompt_speeds.push(PROMPT_TOKENS as f64 / start.elapsed().as_secs_f64());

        // tg: 빈 캐시에서 토큰을 하나씩 32번 넣는다. 매번 점수까지 계산한다.
        let mut cache = KvCache::new(&model.config);
        let start = Instant::now();
        for token in &tokens[..NEW_TOKENS] {
            model.forward_cached(std::slice::from_ref(token), &mut cache)?;
        }
        generation_speeds.push(NEW_TOKENS as f64 / start.elapsed().as_secs_f64());
    }
    // ANCHOR_END: measure

    println!(
        "badak, 모델 {name}, 스레드 {}개, 되풀이 {repetitions}번",
        badak_core::threads::global().pool().threads()
    );
    println!("{:<10} {:>18}", "시험", "토큰/초");
    for (label, speeds) in [("pp256", &prompt_speeds), ("tg32", &generation_speeds)] {
        let (mean, deviation) = mean_and_deviation(speeds);
        println!("{:<10} {:>10.2} ± {:.2}", label, mean, deviation);
    }
    println!();
    println!("{:<28} {:>8.2} 초", "가중치를 읽는 시간", load);

    // 토큰화의 비중: 프롬프트 256토큰만큼의 글을 토큰으로 바꾸는 시간.
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(26);
    let start = Instant::now();
    let rounds = 200;
    let mut count = 0;
    for _ in 0..rounds {
        count = std::hint::black_box(tokenizer.encode(&text, false)).len();
    }
    let tokenize = start.elapsed().as_secs_f64() / f64::from(rounds);
    println!(
        "{:<28} {:>8.3} 밀리초 (토큰 {count}개)",
        "글을 토큰으로 바꾸는 시간",
        tokenize * 1e3
    );
    Ok(())
}
