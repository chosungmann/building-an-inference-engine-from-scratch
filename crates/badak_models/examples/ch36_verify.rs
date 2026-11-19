//! 36장: SmolVLM-256M의 계산을 transformers의 정답과 단계마다 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch36_verify [fast]`

use std::time::Instant;

use badak_formats::safetensors::SafeTensors;
use badak_image::Image;
use badak_image::lanczos::squash;
use badak_models::models_dir;
use badak_models::smolvlm::SmolVlm;
use badak_models::tokenizer::{TokenId, Tokenizer};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const IMAGES: [&str; 2] = ["tabby_cat", "pelican"];

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
    let directory = models_dir().join("smolvlm");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let floats = |key: String| -> Outcome<Vec<f32>> { Ok(reference.tensor(&key)?.to_vec()?) };
    let ids = |key: String| -> Outcome<Vec<TokenId>> {
        Ok(reference
            .tensor(&key)?
            .to_vec::<i32>()?
            .iter()
            .map(|&id| id as TokenId)
            .collect())
    };
    let start = Instant::now();
    let model = SmolVlm::load(&directory)?;
    let model = if fast { model.fast()? } else { model };
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    println!(
        "SmolVLM-256M{}: 사진 {}픽셀, 조각 {}개, 사진 한 장에 토큰 {}개 (읽는 데 {:.0}ms)",
        if fast { " (빠른 길)" } else { "" },
        model.vision.image_size(),
        model.vision.patches(),
        model.image_tokens(),
        start.elapsed().as_secs_f64() * 1e3
    );

    // 글의 틀. 사진 자리 64개를 앞뒤의 표시 토큰이 감싼다.
    let prompt = format!(
        "<|im_start|>User:<fake_token_around_image><global-img>{}<fake_token_around_image>Describe this image.<end_of_utterance>\nAssistant:",
        "<image>".repeat(model.image_tokens())
    );
    let prompt_ids = tokenizer.encode(&prompt, true);
    let stop = tokenizer
        .token_id("<end_of_utterance>")
        .ok_or("끝 토큰이 없습니다")?;

    let mut failed = false;
    for name in IMAGES {
        println!("\n{name}");
        println!(
            "  {:<22} {:>10} {:>10} {:>9}",
            "단계", "차이", "값의 크기", "시간"
        );
        let expected_ids = ids(format!("{name}.ids"))?;
        let same_prompt = prompt_ids == expected_ids;
        println!(
            "  {:<22} {}",
            format!("글의 토큰 {}개", prompt_ids.len()),
            if same_prompt { "같음" } else { "다름" }
        );

        // 전처리. 사진 파일에서 우리가 만든 값과 정답의 값.
        let pixels = floats(format!("{name}.pixels"))?;
        let photo = Image::open(models_dir().join(format!("images/{name}.jpg")))?;
        let start = Instant::now();
        let ours = squash(&photo, 2048, model.vision.image_size());
        let squash_time = start.elapsed();
        let different = ours
            .iter()
            .zip(&pixels)
            .filter(|(a, b)| (**a - **b).abs() > 1e-6)
            .count();
        let (pixel_gap, _) = gap(&ours, &pixels);
        println!(
            "  {:<22} {pixel_gap:>10.2e} {:>10} {:>7.0}ms",
            "전처리",
            format!("{different}개 다름"),
            squash_time.as_secs_f64() * 1e3
        );
        let start = Instant::now();
        let patches = model.vision.encode(&pixels)?;
        let vision_time = start.elapsed();
        let (patch_gap, scale) = gap(&patches, &floats(format!("{name}.patches"))?);
        println!(
            "  {:<22} {patch_gap:>10.2e} {scale:>10.1} {:>7.0}ms",
            "비전 인코더",
            vision_time.as_secs_f64() * 1e3
        );

        let start = Instant::now();
        let seen = model.connect(&patches)?;
        let connect_time = start.elapsed();
        let (seen_gap, scale) = gap(&seen, &floats(format!("{name}.seen"))?);
        println!(
            "  {:<22} {seen_gap:>10.2e} {scale:>10.1} {:>7.1}ms",
            "커넥터",
            connect_time.as_secs_f64() * 1e3
        );

        let start = Instant::now();
        let mut times = Vec::new();
        let mut last = Instant::now();
        let (written, logits) = model.generate(&expected_ids, &seen, stop, 48, |_| {
            times.push(last.elapsed());
            last = Instant::now();
        })?;
        let total = start.elapsed();
        let (logits_gap, scale) = gap(&logits, &floats(format!("{name}.logits"))?);
        println!(
            "  {:<22} {logits_gap:>10.2e} {scale:>10.1} {:>7.0}ms",
            "글을 읽은 직후의 점수",
            times.first().map_or(0.0, |time| time.as_secs_f64() * 1e3)
        );
        let expected = ids(format!("{name}.written"))?;
        // 정답은 끝 토큰까지 들어 있을 수 있다.
        let expected: Vec<TokenId> = expected.into_iter().take_while(|&id| id != stop).collect();
        let same = written == expected;
        let rest: f64 = times.iter().skip(1).map(|time| time.as_secs_f64()).sum();
        println!(
            "  쓴 토큰 {}개: {} (모두 {:.0}ms, 둘째 토큰부터 토큰마다 {:.1}ms)",
            written.len(),
            if same {
                "정답과 같음"
            } else {
                "정답과 다름"
            },
            total.as_secs_f64() * 1e3,
            rest * 1e3 / (times.len().max(2) - 1) as f64
        );
        println!("  {:?}", tokenizer.decode(&written));
        failed |= !same_prompt || !same || patch_gap > 5e-2 || seen_gap > 5e-2 || logits_gap > 5e-2;
    }
    if failed {
        return Err("정답과 다릅니다".into());
    }
    Ok(())
}
