//! 36장: 사진 한 장을 보고 답을 쓰는 일의 단계마다의 시간.
//!
//! 실행: `cargo run --release -p badak_models --example ch36_bench [fast]`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다. 입력은 정답 파일에 든
//! 고양이 사진이다.

use std::hint::black_box;
use std::time::Instant;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::smolvlm::SmolVlm;
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Outcome<()> {
    let fast = std::env::args().nth(1).as_deref() == Some("fast");
    let directory = models_dir().join("smolvlm");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let model = SmolVlm::load(&directory)?;
    let model = if fast { model.fast()? } else { model };
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let stop = tokenizer
        .token_id("<end_of_utterance>")
        .ok_or("끝 토큰이 없습니다")?;
    let workers = threads::global().pool().threads();
    let path = if fast { "빠른 길" } else { "기본   " };
    let pixels = reference.tensor("tabby_cat.pixels")?.to_vec::<f32>()?;
    let ids: Vec<TokenId> = reference
        .tensor("tabby_cat.ids")?
        .to_vec::<i32>()?
        .iter()
        .map(|&id| id as TokenId)
        .collect();
    let plan = Plan {
        warmup: 2,
        repetitions: 11,
    };

    // 1. 사진 보기: 비전 인코더와 커넥터.
    let samples = measure(plan, || {
        black_box(model.vision.encode(&pixels).expect("실행"));
    });
    println!(
        "vision  {path} 스레드 {workers:>2} 중앙값 {:>8.2}ms",
        Summary::new(&samples).median * 1e3
    );
    let patches = model.vision.encode(&pixels)?;
    let samples = measure(plan, || {
        black_box(model.connect(&patches).expect("실행"));
    });
    println!(
        "connect {path} 스레드 {workers:>2} 중앙값 {:>8.2}ms",
        Summary::new(&samples).median * 1e3
    );
    let seen = model.connect(&patches)?;

    // 2. 글 읽기(사진의 벡터 64개가 든 토큰 79개)와 쓰기(48토큰).
    let (mut reads, mut writes) = (Vec::new(), Vec::new());
    for _ in 0..7 {
        let start = Instant::now();
        let mut first = None;
        let (written, _) = model.generate(&ids, &seen, stop, 48, |_| {
            first.get_or_insert_with(|| start.elapsed());
        })?;
        let total = start.elapsed();
        let first = first.unwrap_or(total);
        reads.push(first.as_secs_f64());
        writes.push((total - first).as_secs_f64() / (written.len().max(2) - 1) as f64);
    }
    println!(
        "read    {path} 스레드 {workers:>2} 중앙값 {:>8.2}ms",
        Summary::new(&reads).median * 1e3
    );
    println!(
        "token   {path} 스레드 {workers:>2} 중앙값 {:>8.2}ms",
        Summary::new(&writes).median * 1e3
    );
    Ok(())
}
