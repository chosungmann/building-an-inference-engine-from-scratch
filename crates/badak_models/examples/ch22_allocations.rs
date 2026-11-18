//! 22장: 토큰을 처리하는 동안 엔진이 메모리를 몇 번, 얼마나 할당하는지 센다.
//!
//! 실행: `cargo run --release -p badak_models --example ch22_allocations -- [모델]`
//!
//! 가중치를 두는 방식은 `BADAK_WEIGHTS`로 고른다. 14장의 기준 측정과 같은 두
//! 가지 일을 한다. 프롬프트 256토큰을 한꺼번에 처리하는 일과, 토큰을
//! 하나씩 처리하는 일이다.

use std::time::Instant;

use badak_bench::alloc::{Counting, SIZE_CLASSES, Snapshot, reset_peak};
use badak_core::random::Pcg64;
use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::TokenId;

// ANCHOR: register
#[global_allocator]
static ALLOCATOR: Counting = Counting;
// ANCHOR_END: register

const PROMPT_TOKENS: usize = 256;
const NEW_TOKENS: usize = 32;

fn megabytes(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let format = WeightFormat::from_environment();
    let before = Snapshot::now();
    let start = Instant::now();
    let model = LlamaModel::load_as(models_dir().join(&name), format)?;
    let loaded = Snapshot::now().since(before);
    println!(
        "모델 {name}, 가중치 {format:?}: 읽는 데 {:.2}초, 할당 {}번, 읽은 뒤 쥐고 있는 힙 {:.1}MB",
        start.elapsed().as_secs_f64(),
        loaded.count,
        megabytes(loaded.current)
    );

    let mut rng = Pcg64::new(22);
    let vocab = model.config.vocab_size as u64;
    let tokens: Vec<TokenId> = (0..PROMPT_TOKENS + NEW_TOKENS)
        .map(|_| (rng.next_u64() % vocab) as TokenId)
        .collect();

    println!();
    println!(
        "{:<26} {:>8} {:>12} {:>14} {:>9} {:>9} {:>9} {:>9}",
        "일", "할당", "합 (MB)", "최대치 (MB)", "1KB 미만", "64KB 미만", "1MB 미만", "1MB 이상"
    );
    let report = |label: &str, used: Snapshot, per: usize| {
        println!(
            "{:<26} {:>8} {:>12.2} {:>14.2} {:>9} {:>9} {:>9} {:>9}",
            label,
            used.count / per,
            megabytes(used.bytes) / per as f64,
            megabytes(used.peak),
            used.by_size[0] / per,
            used.by_size[1] / per,
            used.by_size[2] / per,
            used.by_size[3] / per
        );
    };
    let _ = SIZE_CLASSES;

    // ANCHOR: measure
    // 한 번 실행해 처음에만 생기는 할당(스레드 풀 등)을 치른 뒤에 센다.
    let mut cache = KvCache::new(&model.config);
    model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;

    let mut cache = KvCache::new(&model.config);
    reset_peak();
    let before = Snapshot::now();
    model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;
    report("프롬프트 256토큰", Snapshot::now().since(before), 1);
    // ANCHOR_END: measure
    // 프롬프트를 처리하느라 키운 아레나의 덩어리를 놓고 나서 센다. 그러지
    // 않으면 토큰을 하나씩 처리하는 도중에 아레나가 덩어리를 작은 것으로
    // 바꾸면서, 그만큼이 "쥐고 있는 양"에서 빠져 최대치가 작게 보인다.
    let prompt_arena = badak_core::arena::statistics().capacity;
    badak_core::arena::release();
    // ANCHOR: measure

    reset_peak();
    let before = Snapshot::now();
    for token in &tokens[PROMPT_TOKENS..] {
        model.forward_cached(std::slice::from_ref(token), &mut cache)?;
    }
    report(
        "새 토큰 하나 (32개의 평균)",
        Snapshot::now().since(before),
        NEW_TOKENS,
    );
    // ANCHOR_END: measure

    // RESERVE: 캐시의 자리를 미리 잡아 두고 같은 일을 한다.
    let mut cache = KvCache::new(&model.config);
    cache.reserve(PROMPT_TOKENS + NEW_TOKENS);
    model.forward_cached(&tokens[..PROMPT_TOKENS], &mut cache)?;
    badak_core::arena::release();
    reset_peak();
    let before = Snapshot::now();
    for token in &tokens[PROMPT_TOKENS..] {
        model.forward_cached(std::slice::from_ref(token), &mut cache)?;
    }
    report(
        "  캐시의 자리를 미리 잡으면",
        Snapshot::now().since(before),
        NEW_TOKENS,
    );
    // RESERVE_END

    // 아레나가 쥐고 있던 덩어리. 프롬프트의 줄에서는 세기 전에 얻어 둔 것이라
    // 위의 표에 보이지 않는다.
    let arena = badak_core::arena::statistics();
    println!();
    println!(
        "아레나가 쥔 덩어리: 프롬프트를 처리할 때 {:.1}MB, 토큰을 하나씩 처리할 때 {:.2}MB",
        megabytes(prompt_arena * 4),
        megabytes(arena.capacity * 4)
    );
    println!(
        "아레나가 내준 자리 {}번, 자리가 모자라 힙으로 보낸 요청 {}번",
        arena.taken, arena.overflowed
    );
    Ok(())
}
