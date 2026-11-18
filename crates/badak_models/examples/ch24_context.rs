//! 24장: 문맥이 길어질 때 토큰 하나를 만드는 시간이 어떻게 늘어나는지 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch24_context -- [모델] [가장 긴 문맥]`
//!
//! 문맥을 0, 256, 1024, ... 토큰으로 채워 두고, 거기에 토큰을 하나씩 더
//! 넣으면서 시간을 잰다. 14.7절의 프로파일러로 어텐션의 몫도 함께 본다.
//! KV 캐시의 정밀도는 `BADAK_KV`로, 가중치의 방식은 `BADAK_WEIGHTS`로 고른다.

use std::time::Instant;

use badak_bench::stats::Summary;
use badak_core::profile;
use badak_core::random::Pcg64;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::TokenId;

const NEW_TOKENS: usize = 20;
const WARM_UP: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let longest: usize = arguments.next().map_or(Ok(4096), |value| value.parse())?;
    let model = LlamaModel::load(models_dir().join(&name))?;
    let mut rng = Pcg64::new(24);
    let vocab = model.config.vocab_size as u64;
    let tokens: Vec<TokenId> = (0..longest + NEW_TOKENS)
        .map(|_| (rng.next_u64() % vocab) as TokenId)
        .collect();

    println!(
        "모델 {name}, 문맥을 채운 뒤 토큰을 하나씩 넣는다 ({}개의 중앙값)",
        NEW_TOKENS - WARM_UP
    );
    println!(
        "{:>10} {:>16} {:>14} {:>14} {:>12} {:>14}",
        "문맥", "채우는 시간 (초)", "토큰 하나 (ms)", "토큰/초", "어텐션 (ms)", "어텐션의 몫"
    );
    let mut cache = KvCache::new(&model.config);
    cache.reserve(longest + NEW_TOKENS);
    let mut contexts = vec![0];
    let mut context = 256;
    while context <= longest {
        contexts.push(context);
        context *= 2;
    }
    let mut filled = 0;
    let mut fill_seconds = 0.0;
    for context in contexts {
        // 문맥을 `context`토큰까지 채운다. 256토큰씩 끊어 넣는다.
        let start = Instant::now();
        while filled < context {
            let end = (filled + 256).min(context);
            model.forward_cached(&tokens[filled..end], &mut cache)?;
            filled = end;
        }
        fill_seconds += start.elapsed().as_secs_f64();

        // 토큰을 하나씩 넣는다. 처음 넷은 준비 운동이다.
        for token in &tokens[context..context + WARM_UP] {
            model.forward_cached(std::slice::from_ref(token), &mut cache)?;
        }
        let mut times = Vec::new();
        profile::reset();
        profile::enable(true);
        for token in &tokens[context + WARM_UP..context + NEW_TOKENS] {
            let start = Instant::now();
            model.forward_cached(std::slice::from_ref(token), &mut cache)?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        profile::enable(false);
        let attention: f64 = profile::report()
            .iter()
            .filter(|(path, _)| path.rsplit('/').next() == Some("attention"))
            .map(|(_, record)| record.own.as_secs_f64() * 1e3)
            .sum::<f64>()
            / times.len() as f64;
        let per_token = Summary::new(&times).median;
        println!(
            "{:>10} {:>16.2} {:>14.2} {:>14.1} {:>12.2} {:>13.0}%",
            context,
            fill_seconds,
            per_token,
            1e3 / per_token,
            attention,
            100.0 * attention / per_token
        );
        // 다음 문맥을 채우기 전에 방금 넣은 토큰들을 캐시에서 뺀다.
        cache.truncate(context);
        filled = context;
    }
    println!(
        "KV 캐시의 크기 {:.1}MB (토큰 {}개)",
        cache.size_in_bytes() as f64 / 1e6,
        cache.len()
    );
    Ok(())
}
