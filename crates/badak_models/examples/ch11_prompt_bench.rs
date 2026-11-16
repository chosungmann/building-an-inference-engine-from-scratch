//! 11장: 프롬프트를 처리하는 시간과 토큰 하나를 만드는 시간을 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch11_prompt_bench -- [모델 디렉터리]`
//!
//! `reference/ch11_bench_pytorch.py`가 같은 토큰으로 같은 것을 잰다.

use std::time::Instant;

use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

/// 프롬프트를 만들 때 되풀이하는 문장.
const SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";
/// 재어 볼 프롬프트의 길이. 토큰 수다.
const LENGTHS: [usize; 3] = [16, 64, 256];
/// 프롬프트 뒤에 만들어 볼 토큰의 수.
const NEW_TOKENS: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    let model = LlamaModel::load(&directory)?;
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let mut ids = tokenizer.prefix().to_vec();
    ids.extend(tokenizer.encode(&SENTENCE.repeat(40), false));

    println!("모델 {name}");
    println!(
        "{:>10} {:>14} {:>14} {:>14} {:>8}",
        "프롬프트", "프롬프트 처리", "토큰당", "새 토큰 하나", "비율"
    );
    for length in LENGTHS {
        let mut cache = KvCache::new(&model.config);
        // 프롬프트 처리: 토큰 `length`개를 한꺼번에 넣는다.
        let start = Instant::now();
        model.forward_cached(&ids[..length], &mut cache)?;
        let prompt = start.elapsed().as_secs_f64();

        // 생성: 토큰을 하나씩 넣는다. 어떤 토큰인지는 시간에 영향이 없다.
        let start = Instant::now();
        for step in 0..NEW_TOKENS {
            model.forward_cached(&ids[length + step..length + step + 1], &mut cache)?;
        }
        let step = start.elapsed().as_secs_f64() / NEW_TOKENS as f64;

        let per_token = prompt / length as f64;
        println!(
            "{:>8}개 {:>11.3} 초 {:>11.4} 초 {:>11.4} 초 {:>7.1}배",
            length,
            prompt,
            per_token,
            step,
            step / per_token
        );
    }
    Ok(())
}
