//! 35장: 글에서 그림을 만든다.
//!
//! 실행: `cargo run --release -p badak_models --example ch35_generate -- \
//!        [--prompt 글] [--steps 20] [--guidance 7.5] [--seed 번호] [--basic] [--out 파일]
//!        [--watch 걸음,걸음,...]`
//!
//! `--seed`를 주지 않으면 정답 파일에 든 잡음에서 시작하고, 끝난 뒤에
//! diffusers의 결과와 견준다. 주면 그 번호로 잡음을 새로 만든다. `--watch`에
//! 적은 걸음이 끝날 때마다 그때의 잠재 그림을 풀어 `파일.걸음.png`로 남긴다.

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_image::Image;
use badak_models::diffusion::{
    Decoder, Map, Settings, TextEncoder, Unet, denoise, noise, to_bytes,
};
use badak_models::models_dir;
use badak_models::tokenizer::clip::ClipTokenizer;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Outcome<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|at| arguments.get(at + 1))
    };
    let prompt = option("--prompt").map_or(
        "a watercolor painting of a lighthouse on a cliff at sunset",
        String::as_str,
    );
    let settings = Settings {
        steps: option("--steps").map_or(Ok(20), |text| text.parse())?,
        guidance: option("--guidance").map_or(Ok(7.5), |text| text.parse())?,
    };
    let seed: Option<u64> = option("--seed").map(|text| text.parse()).transpose()?;
    let fast = !arguments.iter().any(|argument| argument == "--basic");
    let directory = models_dir().join("stable_diffusion");
    let out =
        option("--out").map_or_else(|| directory.join("generated.png"), std::path::PathBuf::from);

    // 1. 부품을 읽는다.
    let start = Instant::now();
    let tokenizer = ClipTokenizer::load(&directory)?;
    let encoder = TextEncoder::load(directory.join("text_encoder.safetensors"), 12)?;
    let unet = Unet::load(directory.join("unet.safetensors"), 8)?;
    let decoder = Decoder::load(directory.join("vae.safetensors"))?;
    let (unet, decoder) = if fast {
        (unet.fast()?, decoder.fast()?)
    } else {
        (unet, decoder)
    };
    let load_time = start.elapsed();

    // 2. 글과 빈 글을 벡터로.
    let start = Instant::now();
    let context = encoder.encode(&tokenizer.encode(prompt, encoder.length()))?;
    let empty = encoder.encode(&tokenizer.encode("", encoder.length()))?;
    let text_time = start.elapsed();

    // 3. 잡음에서 시작해 걷어 낸다.
    let reference = SafeTensors::open(directory.join("reference.safetensors")).ok();
    let latent = match (seed, &reference) {
        (Some(seed), _) => Map::new(64, 64, 4, noise(seed, 64 * 64 * 4)),
        (None, Some(reference)) => Map::from_planes(
            64,
            64,
            4,
            &reference.tensor("latent.start")?.to_vec::<f32>()?,
        ),
        (None, None) => return Err("정답 파일이 없습니다. --seed를 주십시오".into()),
    };
    println!(
        "글: {prompt:?}\n걸음 {}, 가이던스 {}, {}",
        settings.steps,
        settings.guidance,
        if fast { "빠른 길" } else { "기본 길" }
    );
    let start = Instant::now();
    let mut last = Instant::now();
    let watched: Vec<usize> = option("--watch")
        .map(|text| {
            text.split(',')
                .filter_map(|part| part.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    let latent = denoise(
        &unet,
        latent,
        &context,
        &empty,
        settings,
        |index, step, latent| {
            println!(
                "  걸음 {:>2} (단계 {step:>3}) {:.2}초",
                index + 1,
                last.elapsed().as_secs_f64()
            );
            if watched.contains(&(index + 1))
                && let Ok(image) = decoder.decode(latent)
            {
                let path = out.with_extension(format!("{}.png", index + 1));
                let saved = Image {
                    width: image.width,
                    height: image.height,
                    pixels: to_bytes(&image),
                }
                .save_png(path);
                if let Err(error) = saved {
                    eprintln!("중간 그림을 쓰지 못했습니다: {error}");
                }
            }
            last = Instant::now();
        },
    )?;
    let denoise_time = start.elapsed();

    // 4. 픽셀로 푼다.
    let start = Instant::now();
    let image = decoder.decode(&latent)?;
    let decode_time = start.elapsed();
    Image {
        width: image.width,
        height: image.height,
        pixels: to_bytes(&image),
    }
    .save_png(&out)?;

    let total = text_time + denoise_time + decode_time;
    println!(
        "읽기 {:.2}초 | 글 {:.2}초, 걷어 내기 {:.2}초(걸음마다 {:.2}초), 풀기 {:.2}초, 합 {:.2}초",
        load_time.as_secs_f64(),
        text_time.as_secs_f64(),
        denoise_time.as_secs_f64(),
        denoise_time.as_secs_f64() / settings.steps as f64,
        decode_time.as_secs_f64(),
        total.as_secs_f64()
    );
    println!("{} 기록", out.display());

    // 정답과 같은 설정이면 견준다.
    if let (None, Some(reference)) = (seed, &reference)
        && settings.steps == 20
        && (settings.guidance - 7.5).abs() < 1e-6
        && option("--prompt").is_none()
    {
        let gap = |ours: &[f32], key: &str| -> Outcome<f32> {
            let theirs = reference.tensor(key)?.to_vec::<f32>()?;
            Ok(ours
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max))
        };
        let latent_gap = gap(&latent.to_planes(), "latent.end")?;
        let image_gap = gap(&image.to_planes(), "image")?;
        let theirs: Vec<u8> = reference
            .tensor("image")?
            .to_vec::<f32>()?
            .iter()
            .map(|&value| ((value / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let ours = Map::new(image.height, image.width, 3, image.data.clone()).to_planes();
        let ours: Vec<u8> = ours
            .iter()
            .map(|&value| ((value / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let different = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        let largest = ours
            .iter()
            .zip(&theirs)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        println!(
            "diffusers와의 차이: 마지막 잠재 그림 {latent_gap:.2e}, 그림 {image_gap:.2e}, 다른 바이트 {different}개({:.2}%, 가장 큰 차이 {largest})",
            100.0 * different as f64 / ours.len() as f64
        );
    }
    Ok(())
}
