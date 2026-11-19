//! 35장: U-Net 한 번과 디코더 한 번의 시간, 그리고 그 시간이 어디에 드는지.
//!
//! 실행: `cargo run --release -p badak_models --example ch35_bench [fast] [profile]`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다.

use std::hint::black_box;
use std::time::Instant;

use badak_core::{profile, threads};
use badak_formats::safetensors::SafeTensors;
use badak_models::diffusion::{Decoder, Map, Unet};
use badak_models::models_dir;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 한 번 데우고 `rounds`번 재서 가운데 값을 돌려준다(초).
fn median(rounds: usize, mut work: impl FnMut() -> Outcome<()>) -> Outcome<f64> {
    work()?;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        work()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn main() -> Outcome<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let has = |word: &str| arguments.iter().any(|argument| argument == word);
    let fast = has("fast");
    let directory = models_dir().join("stable_diffusion");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let workers = threads::global().pool().threads();
    let path = if fast { "빠른 길" } else { "기본   " };

    let unet = Unet::load(directory.join("unet.safetensors"), 8)?;
    let unet = if fast { unet.fast()? } else { unet };
    let latent = Map::from_planes(
        64,
        64,
        4,
        &reference.tensor("latent.start")?.to_vec::<f32>()?,
    );
    let context = reference.tensor("context.0")?.to_vec::<f32>()?;
    let rounds = if fast || workers > 1 { 5 } else { 3 };
    let seconds = median(rounds, || {
        black_box(unet.forward(&latent, 951, &context)?);
        Ok(())
    })?;
    println!("unet {path} 스레드 {workers:>2} 중앙값 {seconds:>7.3}초");
    if has("profile") {
        profile::reset();
        profile::enable(true);
        black_box(unet.forward(&latent, 951, &context)?);
        profile::enable(false);
        for (name, record) in profile::report() {
            println!(
                "  unet {name:<12} {:>5}번 {:>8.1}ms",
                record.calls,
                record.total.as_secs_f64() * 1e3
            );
        }
    }
    drop(unet);

    let decoder = Decoder::load(directory.join("vae.safetensors"))?;
    let decoder = if fast { decoder.fast()? } else { decoder };
    let latent = Map::from_planes(64, 64, 4, &reference.tensor("latent.end")?.to_vec::<f32>()?);
    let seconds = median(rounds, || {
        black_box(decoder.decode(&latent)?);
        Ok(())
    })?;
    println!("vae  {path} 스레드 {workers:>2} 중앙값 {seconds:>7.3}초");
    if has("profile") {
        profile::reset();
        profile::enable(true);
        black_box(decoder.decode(&latent)?);
        profile::enable(false);
        for (name, record) in profile::report() {
            println!(
                "  vae  {name:<12} {:>5}번 {:>8.1}ms",
                record.calls,
                record.total.as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}
