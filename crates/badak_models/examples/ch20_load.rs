//! 20장: 가중치를 메모리에 두는 방식마다 읽는 시간과 크기를 잰다.
//!
//! 실행: `BADAK_WEIGHTS=<f32|stored|q8_0|q4_0> cargo run --release -p badak_models --example ch20_load -- [모델 디렉터리]`
//!
//! 방식 하나를 읽어 한 줄을 찍는다. 메모리 사용량은 운영체제에 물어야 하므로
//! `/usr/bin/time -l`로 감싸 실행한다(`scripts/measure_ch20_memory.sh`).

use std::time::Instant;

use badak_models::llama::WeightFormat;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let format = WeightFormat::from_environment();
    let start = Instant::now();
    let model = LlamaModel::load_as(models_dir().join(&name), format)?;
    let seconds = start.elapsed().as_secs_f64();
    let bytes: usize = model
        .matrices()
        .iter()
        .map(|(_, _, matrix)| matrix.size_in_bytes())
        .sum();
    // 가중치를 실제로 메모리에 불러오도록 토큰 몇 개를 처리한다.
    let mut cache = KvCache::new(&model.config);
    model.forward_cached(&[1, 2, 3, 4], &mut cache)?;
    println!(
        "{:<8} 읽는 시간 {:>6.2}초, 행렬 가중치 {:>5.2}GB",
        format!("{format:?}"),
        seconds,
        bytes as f64 / 1e9
    );
    Ok(())
}
