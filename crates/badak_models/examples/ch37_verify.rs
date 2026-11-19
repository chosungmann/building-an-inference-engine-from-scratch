//! 37장: OuteTTS-0.1-350M의 언어 모델을 transformers의 정답과 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch37_verify`

use std::time::Instant;

use badak_audio::Wav;
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::{TokenId, Tokenizer};
use badak_models::tts::{OuteTts, Voice, prompt};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const TEXT: &str = "the engine reads a model file and turns text into speech";

fn main() -> Outcome<()> {
    let directory = models_dir().join("outetts");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let ids = |key: &str| -> Outcome<Vec<TokenId>> {
        Ok(reference
            .tensor(key)?
            .to_vec::<i32>()?
            .iter()
            .map(|&id| id as TokenId)
            .collect())
    };
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let model = OuteTts::load(&directory)?;
    let stop = tokenizer
        .token_id("<|audio_end|>")
        .ok_or("끝 토큰이 없습니다")?;

    let prompt_ids = tokenizer.encode(&prompt(TEXT), true);
    let same_prompt = prompt_ids == ids("ids")?;
    println!(
        "글의 토큰 {}개: {}",
        prompt_ids.len(),
        if same_prompt { "같음" } else { "다름" }
    );

    let start = Instant::now();
    let mut first = None;
    // 정답과 같은 방법: 점수가 가장 높은 토큰, 반복 벌점 1.1.
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.1,
        ..SamplerConfig::default()
    });
    let (written, logits) = model.generate(&ids("ids")?, stop, 700, &mut sampler, |_| {
        first.get_or_insert_with(|| start.elapsed());
    })?;
    let total = start.elapsed();
    let theirs = reference.tensor("logits")?.to_vec::<f32>()?;
    let gap = logits
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("글을 읽은 직후의 점수: 차이 {gap:.2e}");

    let expected = ids("written")?;
    let agree = written
        .iter()
        .zip(&expected)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "쓴 토큰 {}개 (정답 {}개), 앞에서부터 {}개가 같음. {:.2}초, 토큰마다 {:.1}ms",
        written.len(),
        expected.len(),
        agree,
        total.as_secs_f64(),
        total.as_secs_f64() * 1e3 / written.len().max(1) as f64
    );
    let text = tokenizer.decode(&written);
    println!("{}", text.chars().take(160).collect::<String>());
    if !same_prompt || gap > 5e-2 || written != expected {
        return Err("정답과 다릅니다".into());
    }

    // 오디오 디코더의 몸통. 변환해 둔 가중치가 있을 때만.
    let weights = directory.join("wavtokenizer.safetensors");
    if !weights.exists() {
        println!("오디오 디코더: 가중치가 없어 건너뜁니다 (reference/ch37_convert.py)");
        return Ok(());
    }
    let voice = Voice::load(weights)?;
    let codes = ids("codes")?;
    let start = Instant::now();
    let hidden = voice.backbone().forward(&codes)?;
    let elapsed = start.elapsed();
    let theirs = reference.tensor("hidden")?.to_vec::<f32>()?;
    let (gap, scale) = hidden
        .iter()
        .zip(&theirs)
        .fold((0.0_f32, 0.0_f32), |(gap, scale), (a, b)| {
            (gap.max((a - b).abs()), scale.max(b.abs()))
        });
    println!(
        "오디오 디코더의 몸통: 코드 {}개, 차이 {gap:.2e} (값의 크기 {scale:.1}), {:.0}ms",
        codes.len(),
        elapsed.as_secs_f64() * 1e3
    );
    if gap > 1e-2 * scale.max(1.0) {
        return Err("오디오 디코더가 정답과 다릅니다".into());
    }

    // 소리.
    let start = Instant::now();
    let audio = voice.decode(&codes)?;
    let elapsed = start.elapsed();
    let theirs = reference.tensor("audio")?.to_vec::<f32>()?;
    if audio.len() != theirs.len() {
        return Err(format!(
            "소리의 길이가 {}입니다. 정답은 {}입니다",
            audio.len(),
            theirs.len()
        )
        .into());
    }
    let (gap, scale) = audio
        .iter()
        .zip(&theirs)
        .fold((0.0_f32, 0.0_f32), |(gap, scale), (a, b)| {
            (gap.max((a - b).abs()), scale.max(b.abs()))
        });
    println!(
        "소리: 표본 {}개 ({:.2}초), 차이 {gap:.2e} (값의 크기 {scale:.2}), 몸통까지 합쳐 {:.0}ms",
        audio.len(),
        audio.len() as f64 / 24_000.0,
        elapsed.as_secs_f64() * 1e3
    );
    if gap > 1e-2 {
        return Err("소리가 정답과 다릅니다".into());
    }
    let path = directory.join("generated.wav");
    Wav {
        sample_rate: 24_000,
        samples: audio,
    }
    .save(&path)?;
    println!("{} 기록", path.display());
    Ok(())
}
