//! 37장: 음성 합성의 시간과, 소리를 조각내어 풀 때의 차이.
//!
//! 실행: `cargo run --release -p badak_models --example ch37_bench [chunks] [fast]`
//!
//! 인자가 없으면 시간을 잰다. 글 읽기, 토큰마다의 시간, 오디오 코드의
//! 수에 따른 디코더의 시간. `chunks`를 주면 정답의 오디오 코드를 낱말
//! 단위로 조각내어 풀고 한 번에 푼 소리와 견준다. 스레드의 수는 환경 변수
//! `BADAK_THREADS`로 정한다. `fast`를 주면 디코더가 31장의 SME 커널을 쓴다.

use std::hint::black_box;
use std::time::Instant;

use badak_audio::{LogMel, resample};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::{TokenId, Tokenizer};
use badak_models::tts::{OuteTts, SAMPLES_PER_CODE, Voice};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// 신호 대 잡음의 비(dB). `whole`이 신호, 둘의 차이가 잡음이다.
fn snr(part: &[f32], whole: &[f32]) -> f64 {
    let signal: f64 = whole.iter().map(|&value| f64::from(value).powi(2)).sum();
    let noise: f64 = part
        .iter()
        .zip(whole)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    10.0 * (signal / noise.max(1e-30)).log10()
}

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
    let fast = std::env::args().any(|argument| argument == "fast");
    let voice = Voice::load(directory.join("wavtokenizer.safetensors"))?;
    let voice = if fast { voice.fast() } else { voice };
    let path = if fast { "빠른 길" } else { "기본   " };
    let codes = ids("codes")?;
    let workers = threads::global().pool().threads();

    if std::env::args().nth(1).as_deref() == Some("chunks") {
        // 정답이 쓴 토큰에서 낱말의 경계(코드가 몇 개째에서 끝나는가)를 찾는다.
        let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
        let (word_end, first_code) = (
            tokenizer
                .token_id("<|code_end|>")
                .ok_or("토큰이 없습니다")?,
            tokenizer.token_id("<|0|>").ok_or("토큰이 없습니다")?,
        );
        let (mut ends, mut count) = (Vec::new(), 0);
        for token in ids("written")? {
            if (first_code..first_code + 4096).contains(&token) {
                count += 1;
            } else if token == word_end {
                ends.push(count);
            }
        }
        let whole = voice.decode(&codes)?;
        println!("낱말 {}개, 코드 {}개", ends.len(), codes.len());
        // 앞뒤를 모두 넣으면 한 번에 푼 것과 같아야 한다. 자르는 자리가 맞는지
        // 확인한다.
        let mut exact = Vec::new();
        let mut from = 0;
        for &until in &ends {
            exact.extend(voice.decode_part(&codes, from, until, usize::MAX, usize::MAX)?);
            from = until;
        }
        let same = exact.iter().zip(&whole).all(|(a, b)| (a - b).abs() < 1e-5);
        println!(
            "앞뒤를 모두 넣고 조각마다 꺼낸 것: 한 번에 푼 것과 {}",
            if same { "같음" } else { "다름" }
        );
        let mel = |audio: &[f32]| LogMel::WHISPER.spectrogram(&resample(audio, 24_000, 16_000));
        let whole_mel = mel(&whole);
        let frames = whole.len() * 2 / 3 / 160;
        println!(
            "{:>6} {:>6} {:>12} {:>12} {:>12}",
            "앞", "뒤", "SNR (dB)", "가장 큰 차이", "푸는 시간"
        );
        for before in [0, 8, 25, 75, usize::MAX] {
            for after in [0, 4, 8] {
                let start = Instant::now();
                let mut audio = Vec::new();
                let mut from = 0;
                for &until in &ends {
                    // 뒤의 코드를 기다리는 만큼 늦게 내보내는 셈이다. 여기서는
                    // 차이만 보려고 이미 있는 코드를 쓴다.
                    audio.extend(voice.decode_part(&codes, from, until, before, after)?);
                    from = until;
                }
                let elapsed = start.elapsed();
                let gap = audio
                    .iter()
                    .zip(&whole)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                let label = if before == usize::MAX {
                    "모두".to_owned()
                } else {
                    before.to_string()
                };
                // 멜 스펙트로그램의 차이. 소리가 있는 토막들만 본다. Whisper의
                // 로그 멜은 로그(밑 10)를 4로 나눈 값이므로 40을 곱하면 dB다.
                let part_mel = mel(&audio);
                let (mut sum, mut cells) = (0.0_f64, 0_usize);
                for band in 0..whole_mel.mels {
                    for frame in 0..frames.min(whole_mel.frames) {
                        let at = band * whole_mel.frames + frame;
                        sum += f64::from((part_mel.data[at] - whole_mel.data[at]).abs());
                        cells += 1;
                    }
                }
                println!(
                    "chunks 앞 {label:>4} 뒤 {after:>2} 파형의 SNR {:>5.1}dB 가장 큰 차이 {gap:>5.3} 멜의 차이 {:>5.2}dB 푸는 시간 {:>4.0}ms",
                    snr(&audio, &whole),
                    40.0 * sum / cells as f64,
                    elapsed.as_secs_f64() * 1e3
                );
            }
        }
        return Ok(());
    }

    // 1. 언어 모델: 글 읽기와 토큰마다의 시간.
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json"))?;
    let model = OuteTts::load(&directory)?;
    let stop = tokenizer
        .token_id("<|audio_end|>")
        .ok_or("토큰이 없습니다")?;
    let prompt = ids("ids")?;
    let (mut reads, mut tokens) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..SamplerConfig::default()
        });
        let start = Instant::now();
        let mut first = None;
        let (written, _) = model.generate(&prompt, stop, 700, &mut sampler, |_| {
            first.get_or_insert_with(|| start.elapsed());
        })?;
        let total = start.elapsed();
        let first = first.unwrap_or(total);
        reads.push(first.as_secs_f64());
        tokens.push((total - first).as_secs_f64() / (written.len().max(2) - 1) as f64);
    }
    println!(
        "read    스레드 {workers:>2} 중앙값 {:>8.2}ms (토큰 {}개)",
        median(reads) * 1e3,
        prompt.len()
    );
    println!(
        "token   스레드 {workers:>2} 중앙값 {:>8.2}ms",
        median(tokens) * 1e3
    );

    // 2. 디코더: 코드의 수에 따른 시간. 정답의 코드를 되풀이해 길이를 맞춘다.
    for count in [75, 150, 300, 600, 1200] {
        let repeated: Vec<u32> = codes.iter().copied().cycle().take(count).collect();
        let samples = (0..7)
            .map(|_| {
                let start = Instant::now();
                black_box(voice.decode(&repeated).expect("실행"));
                start.elapsed().as_secs_f64()
            })
            .collect();
        let seconds = median(samples);
        println!(
            "decode  {path} 스레드 {workers:>2} 코드 {count:>4} ({:>4.1}초의 소리) 중앙값 {:>8.2}ms 실시간의 {:>6.1}배",
            (count * SAMPLES_PER_CODE) as f64 / 24_000.0,
            seconds * 1e3,
            (count * SAMPLES_PER_CODE) as f64 / 24_000.0 / seconds
        );
    }
    Ok(())
}
