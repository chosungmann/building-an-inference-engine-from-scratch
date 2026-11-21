//! 22장: 가중치를 한 번 양자화해 파일로 적어 둔다.
//!
//! 실행: `cargo run --release -p badak_models --example ch22_pack -- [모델]`
//!
//! `models/<모델>/model_q8_0.safetensors`와 `model_q4_0.safetensors`가 생긴다.
//! 그 뒤로 `BADAK_WEIGHTS=q8_0`이나 `q4_0`으로 모델을 읽으면 이 파일을 매핑해
//! 쓴다.

use std::time::Instant;

use badak_models::llama::{WeightFormat, pack};
use badak_models::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let directory = models_dir().join(&name);
    for format in [WeightFormat::Q8_0, WeightFormat::Q4_0] {
        let start = Instant::now();
        let path = pack(&directory, format)?;
        let size = std::fs::metadata(&path)?.len();
        println!(
            "{:<28} {:>6.2}GB, 만드는 데 {:.1}초",
            path.file_name().unwrap_or_default().to_string_lossy(),
            size as f64 / 1e9,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
