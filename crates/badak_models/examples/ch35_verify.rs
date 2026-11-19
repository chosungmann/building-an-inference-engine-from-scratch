//! 35장: Stable Diffusion 1.5의 부품을 하나씩 diffusers의 정답과 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch35_verify [fast]`

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_models::diffusion::{Decoder, Map, TextEncoder, Unet};
use badak_models::models_dir;
use badak_models::tokenizer::clip::ClipTokenizer;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const PROMPTS: [&str; 3] = [
    "a watercolor painting of a lighthouse on a cliff at sunset",
    "",
    "A photo of 2 cats, isn't it?! Très bien.",
];

fn gap(ours: &[f32], theirs: &[f32]) -> (f32, f32) {
    assert_eq!(ours.len(), theirs.len());
    let largest = ours
        .iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = theirs
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    (largest, scale)
}

fn main() -> Outcome<()> {
    let fast = std::env::args().nth(1).as_deref() == Some("fast");
    let directory = models_dir().join("stable_diffusion");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let floats = |key: &str| -> Outcome<Vec<f32>> { Ok(reference.tensor(key)?.to_vec()?) };
    println!(
        "Stable Diffusion 1.5{}",
        if fast { " (빠른 길)" } else { "" }
    );

    // 1. 토크나이저와 텍스트 인코더.
    let tokenizer = ClipTokenizer::load(&directory)?;
    let start = Instant::now();
    let encoder = TextEncoder::load(directory.join("text_encoder.safetensors"), 12)?;
    println!(
        "\n텍스트 인코더: 조각 {}가지, 토큰 {}개, 폭 {} (읽는 데 {:.0}ms)",
        tokenizer.vocab_size(),
        encoder.length(),
        encoder.width(),
        start.elapsed().as_secs_f64() * 1e3
    );
    println!(
        "  {:<6} {:<8} {:>10} {:>9} {:>9}",
        "글", "토큰", "차이", "값의 크기", "시간"
    );
    let mut contexts = Vec::new();
    let mut worst = 0.0_f32;
    let mut all_same = true;
    for (index, prompt) in PROMPTS.iter().enumerate() {
        let ids = tokenizer.encode(prompt, encoder.length());
        let expected = reference.tensor(&format!("ids.{index}"))?.to_vec::<i32>()?;
        let same = ids.iter().map(|&id| id as i32).eq(expected.iter().copied());
        all_same &= same;
        let start = Instant::now();
        let context = encoder.encode(&ids)?;
        let elapsed = start.elapsed();
        let (largest, scale) = gap(&context, &floats(&format!("context.{index}"))?);
        worst = worst.max(largest);
        println!(
            "  {index:<6} {:<8} {largest:>10.2e} {scale:>9.1} {:>7.0}ms",
            if same { "같음" } else { "다름" },
            elapsed.as_secs_f64() * 1e3
        );
        contexts.push(context);
    }

    // 2. U-Net 한 번.
    let start = Instant::now();
    let unet = Unet::load(directory.join("unet.safetensors"), 8)?;
    let mut unet = if fast { unet.fast()? } else { unet };
    println!(
        "\nU-Net: 합성곱과 선형 층의 가중치 {:.0}만 개 (읽는 데 {:.0}ms)",
        unet.parameters() as f64 / 1e4,
        start.elapsed().as_secs_f64() * 1e3
    );
    let latent = Map::from_planes(64, 64, 4, &floats("latent.start")?);
    let step = reference.tensor("timesteps")?.to_vec::<i32>()?[0] as usize;
    let start = Instant::now();
    let noise = unet.forward(&latent, step, &contexts[0])?;
    let elapsed = start.elapsed();
    let (unet_gap, scale) = gap(&noise.to_planes(), &floats("noise.first")?);
    println!(
        "  단계 {step}에서 짐작한 잡음: 차이 {unet_gap:.2e} (값의 크기 {scale:.1}), {:.2}초",
        elapsed.as_secs_f64()
    );

    // 3. 디코더. 정답의 마지막 잠재 그림을 푼다.
    let start = Instant::now();
    let decoder = Decoder::load(directory.join("vae.safetensors"))?;
    let decoder = if fast { decoder.fast()? } else { decoder };
    let load_time = start.elapsed();
    let latent = Map::from_planes(64, 64, 4, &floats("latent.end")?);
    let start = Instant::now();
    let image = decoder.decode(&latent)?;
    let elapsed = start.elapsed();
    let (image_gap, scale) = gap(&image.to_planes(), &floats("image")?);
    println!(
        "\nVAE 디코더 (읽는 데 {:.0}ms)\n  {} × {} 그림: 차이 {image_gap:.2e} (값의 크기 {scale:.1}), {:.2}초",
        load_time.as_secs_f64() * 1e3,
        image.width,
        image.height,
        elapsed.as_secs_f64()
    );

    if !all_same || worst > 1e-3 || unet_gap > 1e-2 || image_gap > 2e-2 {
        return Err("정답과 다릅니다".into());
    }
    Ok(())
}
