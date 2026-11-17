//! 12장: 소프트맥스의 합이 얼마나 정확한지, 샘플러가 얼마나 걸리는지 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch12_bench -- [모델 디렉터리]`
//!
//! 모델을 실행하지 않는다. 9장의 정답 파일에 든 마지막 줄의 점수를 쓴다.

use std::time::Instant;

use badak_core::linalg::{sum_naive, sum_pairwise};
use badak_core::prob::argmax;
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "llama32_1b".to_string());
    let reference = SafeTensors::open(models_dir().join(&name).join("reference.safetensors"))?;
    let logits = reference.tensor("logits")?;
    let rows = logits.shape()[0];
    let logits = logits.slice(0, rows - 1..rows)?.to_vec::<f32>()?;
    println!("모델 {name}, 점수 {}개", logits.len());

    // ANCHOR: sum
    // 소프트맥스의 분모: exp(점수 - 최댓값)을 모두 더한 값. 세 가지로 더한다.
    println!();
    println!("소프트맥스의 분모를 더하는 방법에 따른 오차");
    println!(
        "{:>6} {:>14} {:>14} {:>14}",
        "온도", "차례로 (f32)", "쌍별로 (f32)", "기준 (f64)"
    );
    for temperature in [1.0_f32, 0.7, 0.5] {
        let scaled: Vec<f32> = logits.iter().map(|logit| logit / temperature).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|s| (s - max).exp()).collect();
        let exact: f64 = exps.iter().map(|&e| f64::from(e)).sum();
        let error = |sum: f32| (f64::from(sum) - exact) / exact;
        println!(
            "{:>6} {:>+14.1e} {:>+14.1e} {:>14.6}",
            temperature,
            error(sum_naive(&exps)),
            error(sum_pairwise(&exps)),
            exact
        );
    }
    // ANCHOR_END: sum

    // 샘플러가 토큰 하나를 뽑는 데 걸리는 시간.
    println!();
    println!("토큰 하나를 뽑는 시간");
    let repeats = 200;
    let start = Instant::now();
    let mut checksum = 0;
    for _ in 0..repeats {
        checksum += argmax(std::hint::black_box(&logits)).unwrap_or(0);
    }
    let greedy = start.elapsed().as_secs_f64() / f64::from(repeats);
    println!(
        "{:<34} {:>9.3} ms",
        "가장 높은 것 고르기 (argmax)",
        greedy * 1e3
    );

    let plain = SamplerConfig::default();
    let settings = [
        ("손질 없음 (전체를 정렬)", plain),
        ("top-k 40", SamplerConfig { top_k: 40, ..plain }),
        (
            "top-p 0.9 (전체를 정렬)",
            SamplerConfig {
                top_p: 0.9,
                ..plain
            },
        ),
        (
            "top-k 40, top-p 0.9",
            SamplerConfig {
                top_k: 40,
                top_p: 0.9,
                ..plain
            },
        ),
        (
            "반복 벌점 1.1, 온도 0.8, top-k 40, top-p 0.9",
            SamplerConfig {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
                ..plain
            },
        ),
    ];
    for (label, config) in settings {
        let mut sampler = Sampler::new(config);
        let start = Instant::now();
        for _ in 0..repeats {
            checksum += sampler.sample(std::hint::black_box(&logits), &[1, 2, 3]) as usize;
        }
        let seconds = start.elapsed().as_secs_f64() / f64::from(repeats);
        println!("{:<34} {:>9.3} ms", label, seconds * 1e3);
    }
    std::hint::black_box(checksum);
    Ok(())
}
