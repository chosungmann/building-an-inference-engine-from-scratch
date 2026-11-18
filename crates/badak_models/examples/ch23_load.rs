//! 23장: 가중치를 GPU에 올렸을 때 프로세스의 메모리를 보려고 쓰는 작은 프로그램.
//!
//! 실행: `sh scripts/measure_ch23_memory.sh`
//!
//! 첫 인자가 `gpu`이면 블록의 선형 층들을 GPU에 올리고, 아니면 CPU에 둔다.
//! 토큰 32개를 한꺼번에 처리하고 끝난다.

use std::sync::Arc;
use std::time::Instant;

use badak_gpu::Gpu;
use badak_gpu::offload::Offload;
use badak_models::llama::cache::KvCache;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::TokenId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let place = arguments.next().unwrap_or_else(|| "cpu".to_string());
    let name = arguments.next().unwrap_or_else(|| "llama32_1b".to_string());
    let start = Instant::now();
    let mut model = LlamaModel::load(models_dir().join(&name))?;
    let mut uploaded = 0;
    if place == "gpu" {
        let gpu = Arc::new(Gpu::new().ok_or("이 기계에는 쓸 수 있는 GPU가 없습니다")?);
        let offload = Offload::new(gpu);
        model.replace_matrices(|kind, _, matrix| {
            if kind == "embedding" {
                return Ok(matrix.clone());
            }
            uploaded += matrix.len() * size_of::<f32>();
            Ok(offload.matrix(matrix.clone())?)
        })?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let tokens: Vec<TokenId> = (0..32).map(|i| 1000 + i * 37).collect();
    let mut cache = KvCache::new(&model.config);
    let logits = model.forward_cached(&tokens, &mut cache)?;
    std::hint::black_box(logits);
    println!(
        "{:<4} 읽는 시간 {:>5.2}초, GPU에 올린 가중치 {:>5.2}GB",
        place.to_uppercase(),
        seconds,
        uploaded as f64 / 1e9
    );
    Ok(())
}
