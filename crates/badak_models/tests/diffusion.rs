//! Stable Diffusion 1.5의 부품을 diffusers의 정답과 견준다 (35장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch35.sh`로 받는다(4.3GB).
//! 토크나이저와 텍스트 인코더는 `sh scripts/download_ch35.sh text`로 받은
//! 것만으로 시험한다.

use badak_formats::safetensors::SafeTensors;
use badak_models::diffusion::{Decoder, Map, Settings, TextEncoder, Unet, denoise};
use badak_models::models_dir;
use badak_models::tokenizer::clip::ClipTokenizer;

const PROMPTS: [&str; 3] = [
    "a watercolor painting of a lighthouse on a cliff at sunset",
    "",
    "A photo of 2 cats, isn't it?! Très bien.",
];

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn the_tokenizer_and_text_encoder_match_the_reference() {
    let directory = models_dir().join("stable_diffusion");
    if !directory.join("text_encoder.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let tokenizer = ClipTokenizer::load(&directory).unwrap();
    assert_eq!(tokenizer.vocab_size(), 49408);
    assert_eq!(tokenizer.end(), 49407);
    let encoder = TextEncoder::load(directory.join("text_encoder.safetensors"), 12).unwrap();
    assert_eq!((encoder.length(), encoder.width()), (77, 768));
    for (index, prompt) in PROMPTS.iter().enumerate() {
        let ids = tokenizer.encode(prompt, 77);
        let expected = reference
            .tensor(&format!("ids.{index}"))
            .unwrap()
            .to_vec::<i32>()
            .unwrap();
        assert!(
            ids.iter().map(|&id| id as i32).eq(expected.iter().copied()),
            "{index}번 글의 토큰이 다릅니다"
        );
        let context = encoder.encode(&ids).unwrap();
        let expected = reference
            .tensor(&format!("context.{index}"))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        // 값의 크기가 33까지 간다.
        assert!(largest_gap(&context, &expected) < 1e-3);
    }
    // 길이가 77이 아니면 거절한다.
    assert!(encoder.encode(&[49406, 49407]).is_err());
    assert!(encoder.encode(&[60_000; 77]).is_err());
}

#[test]
fn the_unet_and_decoder_match_the_reference_on_both_paths() {
    let directory = models_dir().join("stable_diffusion");
    if !directory.join("unet.safetensors").exists() {
        eprintln!("건너뜀: {}에 U-Net이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let floats = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let context = floats("context.0");
    let start = Map::from_planes(64, 64, 4, &floats("latent.start"));
    let end = Map::from_planes(64, 64, 4, &floats("latent.end"));

    // 빠른 길만 끝까지 돌린다. 기본 길은 U-Net 한 번만.
    let basic = Unet::load(directory.join("unet.safetensors"), 8).unwrap();
    let noise = basic.forward(&start, 951, &context).unwrap();
    assert_eq!((noise.height, noise.width, noise.channels), (64, 64, 4));
    assert!(largest_gap(&noise.to_planes(), &floats("noise.first")) < 1e-3);
    // 채널 수가 다른 그림은 거절한다.
    assert!(
        basic
            .forward(&Map::new(8, 8, 3, vec![0.0; 192]), 951, &context)
            .is_err()
    );

    let fast = basic.fast().unwrap();
    let noise = fast.forward(&start, 951, &context).unwrap();
    assert!(largest_gap(&noise.to_planes(), &floats("noise.first")) < 1e-3);

    // 두 걸음짜리 실행. 가이던스가 1이면 U-Net을 걸음마다 한 번만 부른다.
    let mut calls = 0;
    let settings = Settings {
        steps: 2,
        guidance: 1.0,
    };
    let short = denoise(
        &fast,
        start.clone(),
        &context,
        &context,
        settings,
        |_, _, _| {
            calls += 1;
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    assert!(short.data.iter().all(|value| value.is_finite()));

    for decoder in [
        Decoder::load(directory.join("vae.safetensors")).unwrap(),
        Decoder::load(directory.join("vae.safetensors"))
            .unwrap()
            .fast()
            .unwrap(),
    ] {
        let image = decoder.decode(&end).unwrap();
        assert_eq!((image.height, image.width, image.channels), (512, 512, 3));
        assert!(largest_gap(&image.to_planes(), &floats("image")) < 5e-3);
    }
}
