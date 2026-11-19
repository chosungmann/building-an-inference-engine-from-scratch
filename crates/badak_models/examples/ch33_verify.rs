//! 33장: Whisper-tiny의 계산을 transformers의 정답과 단계마다 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch33_verify [fast]`

use std::time::Instant;

use badak_audio::{LogMel, Wav};
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::whisper::{Vocabulary, Whisper};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn gap(ours: &[f32], theirs: &[f32]) -> (f32, f32) {
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
    let directory = models_dir().join("whisper");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let start = Instant::now();
    let model = Whisper::load(&directory)?;
    let model = if fast { model.fast()? } else { model };
    let vocabulary = Vocabulary::load(&directory)?;
    println!(
        "Whisper-tiny{}: 토큰 {}가지, 인코더의 자리 {}개 (읽는 데 {:.1}ms)",
        if fast { " (빠른 길)" } else { "" },
        model.vocabulary(),
        model.frames(),
        start.elapsed().as_secs_f64() * 1e3
    );

    let wav = Wav::open(directory.join("jfk.wav"))?;
    let start = Instant::now();
    let mel = LogMel::WHISPER.spectrogram(&wav.samples);
    let mel_time = start.elapsed();
    let start = Instant::now();
    let encoded = model.encode(&mel.data)?;
    let encode_time = start.elapsed();

    println!(
        "\njfk.wav ({:.1}초): 단계마다의 가장 큰 차이",
        wav.seconds()
    );
    println!("  {:<22} {:>10} {:>12}", "단계", "차이", "값의 크기");
    let (largest, scale) = gap(&mel.data, &reference.tensor("jfk.mel")?.to_vec::<f32>()?);
    println!("  {:<22} {largest:>10.2e} {scale:>12.2}", "멜 스펙트로그램");
    let ours = encoded.output.to_vec::<f32>()?;
    let (largest, scale) = gap(&ours, &reference.tensor("jfk.encoded")?.to_vec::<f32>()?);
    println!("  {:<22} {largest:>10.2e} {scale:>12.2}", "인코더의 출력");
    let encoder_gap = largest;

    // 지시 토큰 넷을 넣은 뒤의 점수.
    let prompt = vocabulary.prompt("en")?;
    let mut state = model.start();
    let mut logits = Vec::new();
    for &token in &prompt {
        logits = model.step(&encoded, &mut state, token)?;
    }
    let (largest, scale) = gap(&logits, &reference.tensor("jfk.logits")?.to_vec::<f32>()?);
    println!(
        "  {:<22} {largest:>10.2e} {scale:>12.2}",
        "디코더의 첫 점수"
    );
    let logits_gap = largest;

    let (language, confidence) = model.detect_language(&encoded, &vocabulary)?;
    println!("\n언어: {language} ({:.1}%)", 100.0 * confidence);

    let start = Instant::now();
    let greedy = model.greedy(&encoded, &vocabulary, &prompt, 200)?;
    let greedy_time = start.elapsed();
    let start = Instant::now();
    let beam = model.beam(&encoded, &vocabulary, &prompt, 5, 200)?;
    let beam_time = start.elapsed();
    let expected = |key: &str| -> Outcome<Vec<u32>> {
        let ids = reference.tensor(&format!("jfk.{key}"))?.to_vec::<i32>()?;
        // transformers는 끝 토큰을 붙여 돌려주기도 한다. 글의 조각만 견준다.
        Ok(ids
            .into_iter()
            .map(|id| id as u32)
            .filter(|&id| id < 50257)
            .collect())
    };
    for (label, transcript, key) in [("탐욕", &greedy, "greedy"), ("빔 5", &beam, "beam")] {
        let same = transcript.tokens == expected(key)?;
        println!(
            "{label}: 토큰 {}개, 로그 확률의 합 {:.3}, transformers와 {}",
            transcript.tokens.len(),
            transcript.log_probability,
            if same {
                "같은 토큰"
            } else {
                "다른 토큰"
            }
        );
        println!("  {:?}", vocabulary.decode(&transcript.tokens));
    }
    println!(
        "\n시간: 멜 {:.0}ms, 인코더 {:.0}ms, 탐욕 디코딩 {:.0}ms (토큰당 {:.1}ms), 빔 서치 {:.0}ms",
        mel_time.as_secs_f64() * 1e3,
        encode_time.as_secs_f64() * 1e3,
        greedy_time.as_secs_f64() * 1e3,
        greedy_time.as_secs_f64() * 1e3 / (greedy.tokens.len() + 1) as f64,
        beam_time.as_secs_f64() * 1e3
    );
    if encoder_gap > 5e-2 || logits_gap > 5e-2 || greedy.tokens != expected("greedy")? {
        return Err("transformers와 다릅니다".into());
    }
    Ok(())
}
