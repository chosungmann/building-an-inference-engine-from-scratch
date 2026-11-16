//! 8장: 이 책의 원고를 토큰으로 바꿔 통계와 속도를 본다.
//!
//! 실행: `cargo run --release -p badak_models --example ch08_stats`

use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use badak_formats::gguf::Gguf;
use badak_models::models_dir;
use badak_models::tokenizer::Tokenizer;

/// 여러 번 실행해 가장 빠른 시간을 초 단위로 돌려준다.
fn best_of<R>(repeats: usize, mut f: impl FnMut() -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let vocab = models_dir().join("llama_cpp_vocab/ggml-vocab-llama-bpe.gguf");
    let (tokenizer, load) = {
        let start = Instant::now();
        let tokenizer = Tokenizer::from_gguf(&Gguf::open(vocab)?)?;
        (tokenizer, start.elapsed().as_secs_f64())
    };
    println!("토크나이저 준비: {:.1} ms", load * 1e3);

    println!();
    println!("== 언어에 따른 토큰 수 (Llama 3) ==");
    println!(
        "{:<26} {:>8} {:>9} {:>8} {:>12} {:>12}",
        "글", "글자", "바이트", "토큰", "글자/토큰", "바이트/토큰"
    );
    let samples = [
        ("영어 (README.md)", root.join("README.md")),
        (
            "한국어 (1장 원고)",
            root.join("book/src/part1/ch01_landscape.md"),
        ),
        (
            "Rust (tensor.rs)",
            root.join("crates/badak_core/src/tensor.rs"),
        ),
    ];
    let mut corpus = String::new();
    for (label, path) in samples {
        let text = std::fs::read_to_string(path)?;
        let tokens = tokenizer.encode(&text, false).len();
        println!(
            "{:<26} {:>8} {:>9} {:>8} {:>12.2} {:>12.2}",
            label,
            text.chars().count(),
            text.len(),
            tokens,
            text.chars().count() as f64 / tokens as f64,
            text.len() as f64 / tokens as f64
        );
        corpus.push_str(&text);
    }

    println!();
    println!(
        "== 속도: 위 세 파일을 이어 붙인 글 ({} 바이트) ==",
        corpus.len()
    );
    let megabytes = corpus.len() as f64 / 1e6;
    let naive = best_of(5, || tokenizer.encode(&corpus, false));
    let cold = best_of(5, || {
        tokenizer.encode_cached(&corpus, false, &mut HashMap::new())
    });
    let mut cache = HashMap::new();
    let ids = tokenizer.encode_cached(&corpus, false, &mut cache);
    let warm = best_of(5, || tokenizer.encode_cached(&corpus, false, &mut cache));
    assert_eq!(ids, tokenizer.encode(&corpus, false));
    println!(
        "{:<34} {:>9.2} ms {:>9.2} MB/s",
        "조각마다 병합을 새로 계산",
        naive * 1e3,
        megabytes / naive
    );
    println!(
        "{:<34} {:>9.2} ms {:>9.2} MB/s",
        "캐시 사용, 빈 캐시에서 시작",
        cold * 1e3,
        megabytes / cold
    );
    println!(
        "{:<34} {:>9.2} ms {:>9.2} MB/s",
        "캐시 사용, 채워진 캐시",
        warm * 1e3,
        megabytes / warm
    );
    println!("토큰 {}개, 서로 다른 조각 {}개", ids.len(), cache.len());
    Ok(())
}
