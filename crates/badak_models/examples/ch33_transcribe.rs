//! 33장: 긴 소리를 받아쓴다. 말이 없는 곳에서 끊어 30초 이하의 창으로
//! 나누고, 창마다 Whisper를 돌린다.
//!
//! 실행: `cargo run --release -p badak_models --example ch33_transcribe [파일.wav]`
//!
//! 파일을 주지 않으면 `jfk.wav`를 침묵을 사이에 두고 다섯 번 이어 붙인
//! 62초짜리 소리를 만들어 쓴다.

use std::time::Instant;

use badak_audio::vad::{self, Settings};
use badak_audio::{LogMel, Wav};
use badak_models::models_dir;
use badak_models::whisper::{Vocabulary, Whisper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = models_dir().join("whisper");
    let model = Whisper::load(&directory)?.fast()?;
    let vocabulary = Vocabulary::load(&directory)?;
    let (name, wav) = match std::env::args().nth(1) {
        Some(path) => (path.clone(), Wav::open(path)?),
        None => {
            let one = Wav::open(directory.join("jfk.wav"))?;
            let mut samples = Vec::new();
            for index in 0..5 {
                samples.extend_from_slice(&one.samples);
                // 사이마다 1.5초에서 2.5초의 침묵.
                samples.resize(samples.len() + 16_000 * (3 + index % 3) / 2, 0.0);
            }
            let wav = Wav {
                sample_rate: one.sample_rate,
                samples,
            };
            ("jfk.wav 다섯 번".to_string(), wav)
        }
    };
    if wav.sample_rate != LogMel::WHISPER.sample_rate {
        return Err(format!(
            "표본 속도가 {}Hz입니다. 16,000Hz여야 합니다",
            wav.sample_rate
        )
        .into());
    }
    let rate = f64::from(wav.sample_rate);
    println!("{name}: {:.1}초", wav.seconds());

    let start = Instant::now();
    let segments = vad::segments(&wav.samples, wav.sample_rate, &Settings::default());
    let windows = vad::windows(&segments, LogMel::WHISPER.samples);
    let speech: usize = segments.iter().map(|segment| segment.len()).sum();
    println!(
        "말소리 구간 {}개 (모두 {:.1}초) → 창 {}개, 찾는 데 {:.1}ms",
        segments.len(),
        speech as f64 / rate,
        windows.len(),
        start.elapsed().as_secs_f64() * 1e3
    );
    for segment in &segments {
        print!(
            " [{:.1}~{:.1}]",
            segment.start as f64 / rate,
            segment.end as f64 / rate
        );
    }
    println!();

    let prompt = vocabulary.prompt("en")?;
    let started = Instant::now();
    for window in &windows {
        let start = Instant::now();
        let mel = LogMel::WHISPER.spectrogram(&wav.samples[window.clone()]);
        let encoded = model.encode(&mel.data)?;
        let transcript = model.beam(&encoded, &vocabulary, &prompt, 5, 200)?;
        println!(
            "[{:>5.1}초 ~ {:>5.1}초] ({:.0}ms){}",
            window.start as f64 / rate,
            window.end as f64 / rate,
            start.elapsed().as_secs_f64() * 1e3,
            vocabulary.decode(&transcript.tokens)
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "받아쓰는 데 {:.2}초. 소리 길이의 {:.0}분의 1",
        elapsed,
        wav.seconds() / elapsed
    );
    Ok(())
}
