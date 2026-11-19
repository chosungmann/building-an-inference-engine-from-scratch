//! 37장: 글을 소리로 바꾼다. 낱말이 끝날 때마다 소리를 내보낸다.
//!
//! 실행: `cargo run --release -p badak_models --example ch37_speak -- \
//!        [--text 글] [--seed 번호] [--out 파일.wav] [--whisper]`
//!
//! `--seed`를 주면 이 모델의 권장 설정(온도 0.4, 상위 40, 누적 0.9)으로
//! 무작위로 고른다. 주지 않으면 점수가 가장 높은 토큰을 고른다. 반복
//! 벌점은 늘 1.1이다. `--whisper`를 주면 만든 소리를 33장의 Whisper로
//! 받아써 본다.

use std::time::Instant;

use badak_audio::{LogMel, Wav, resample};
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::Tokenizer;
use badak_models::tts::{OuteTts, SAMPLE_RATE, SAMPLES_PER_CODE, Voice, prompt};
use badak_models::whisper::{Vocabulary, Whisper};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// 조각을 풀 때 함께 넣는 앞의 코드 수.
const BEFORE: usize = 75;

fn main() -> Outcome<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|at| arguments.get(at + 1))
    };
    let text = option("--text").map_or(
        "the engine reads a model file and turns text into speech",
        String::as_str,
    );
    let seed: Option<u64> = option("--seed").map(|text| text.parse()).transpose()?;
    let directory = models_dir().join("outetts");
    let out =
        option("--out").map_or_else(|| directory.join("spoken.wav"), std::path::PathBuf::from);

    let start = Instant::now();
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let model = OuteTts::load(&directory)?;
    let voice = Voice::load(directory.join("wavtokenizer.safetensors"))?.fast();
    let load_time = start.elapsed();
    let token = |name: &str| {
        tokenizer
            .token_id(name)
            .ok_or_else(|| format!("{name} 토큰이 없습니다"))
    };
    let (stop, word_end, first_code) = (
        token("<|audio_end|>")?,
        token("<|code_end|>")?,
        token("<|0|>")?,
    );

    let config = match seed {
        Some(seed) => SamplerConfig {
            temperature: 0.4,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed,
            ..SamplerConfig::default()
        },
        None => SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..SamplerConfig::default()
        },
    };
    let mut sampler = Sampler::new(config);
    let ids = tokenizer.encode(&prompt(text), true);
    println!("글: {text:?} (토큰 {}개)", ids.len());

    // ANCHOR: stream
    // 쓰는 대로 모은 오디오 코드와, 이미 소리로 내보낸 코드의 수.
    let mut codes: Vec<u32> = Vec::new();
    let mut spoken = 0;
    let mut audio: Vec<f32> = Vec::new();
    let mut chunks: Vec<(f64, usize, f64)> = Vec::new();
    let mut failure = None;
    let start = Instant::now();
    let (written, _) = model.generate(&ids, stop, 1500, &mut sampler, |token| {
        if (first_code..first_code + 4096).contains(&token) {
            codes.push(token - first_code);
        }
        // 낱말 하나의 코드가 끝났다. 그 낱말의 소리를 만들어 내보낸다. 앞의
        // 코드 75개(1초)를 함께 넣어 풀어야 이음매가 매끄럽다.
        if token == word_end && codes.len() > spoken {
            let began = Instant::now();
            match voice.decode_part(&codes, spoken, codes.len(), BEFORE, 0) {
                Ok(samples) => {
                    audio.extend_from_slice(&samples);
                    chunks.push((
                        start.elapsed().as_secs_f64(),
                        codes.len() - spoken,
                        began.elapsed().as_secs_f64(),
                    ));
                    spoken = codes.len();
                }
                Err(error) => failure = Some(error),
            }
        }
    })?;
    // ANCHOR_END: stream
    if let Some(error) = failure {
        return Err(error.into());
    }
    let total = start.elapsed().as_secs_f64();
    let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);

    println!("  낱말  나온 때   코드  소리의 길이  푸는 시간");
    for (index, (at, count, decode)) in chunks.iter().enumerate() {
        println!(
            "  {:>4} {:>7.0}ms {:>5} {:>9.2}초 {:>8.1}ms",
            index + 1,
            at * 1e3,
            count,
            (count * SAMPLES_PER_CODE) as f64 / f64::from(SAMPLE_RATE),
            decode * 1e3
        );
    }
    let first = chunks.first().map_or(total, |chunk| chunk.0);
    let decode_time: f64 = chunks.iter().map(|chunk| chunk.2).sum();
    println!(
        "쓴 토큰 {}개(오디오 코드 {}개), 소리 {seconds:.2}초",
        written.len(),
        codes.len()
    );
    println!(
        "첫 소리까지 {:.0}ms, 모두 {:.2}초(그 가운데 소리로 푸는 데 {:.0}ms), 실시간의 {:.2}배 빠름 (읽는 데 {:.0}ms)",
        first * 1e3,
        total,
        decode_time * 1e3,
        seconds / total,
        load_time.as_secs_f64() * 1e3
    );

    // 한 번에 푼 소리와 견준다. 조각내어 푼 것이 얼마나 다른가.
    let whole = voice.decode(&codes[..spoken])?;
    let (gap, peak) = audio
        .iter()
        .zip(&whole)
        .fold((0.0_f32, 0.0_f32), |(gap, peak), (a, b)| {
            (gap.max((a - b).abs()), peak.max(b.abs()))
        });
    println!(
        "조각내어 푼 소리와 한 번에 푼 소리의 가장 큰 차이 {gap:.3} (소리의 가장 큰 값 {peak:.2})"
    );

    // `--whole`을 주면 조각낸 소리 대신 한 번에 푼 소리를 쓰고 듣는다.
    let audio = if arguments.iter().any(|argument| argument == "--whole") {
        whole
    } else {
        audio
    };
    Wav {
        sample_rate: SAMPLE_RATE,
        samples: audio.clone(),
    }
    .save(&out)?;
    println!("{} 기록", out.display());

    if arguments.iter().any(|argument| argument == "--whisper") {
        // ANCHOR: listen
        // 만든 소리를 16kHz로 바꿔 33장의 Whisper에 들려준다.
        let directory = models_dir().join("whisper");
        let whisper = Whisper::load(&directory)?.fast()?;
        let vocabulary = Vocabulary::load(&directory)?;
        let heard = resample(&audio, SAMPLE_RATE, 16_000);
        let mel = LogMel::WHISPER.spectrogram(&heard);
        let encoded = whisper.encode(&mel.data)?;
        let transcript = whisper.beam(&encoded, &vocabulary, &vocabulary.prompt("en")?, 5, 200)?;
        println!(
            "Whisper가 받아쓴 글: {:?}",
            vocabulary.decode(&transcript.tokens)
        );
        // ANCHOR_END: listen
    }
    Ok(())
}
