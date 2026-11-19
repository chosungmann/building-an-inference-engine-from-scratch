//! 33장: Whisper-tiny의 단계마다의 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch33_bench [fast]`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다. `fast`를 주면 31장의
//! SME 커널과 32장의 근사한 지수 함수를 쓴다.

use std::hint::black_box;

use badak_audio::{LogMel, Wav};
use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_models::models_dir;
use badak_models::whisper::{Vocabulary, Whisper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fast = std::env::args().nth(1).as_deref() == Some("fast");
    let directory = models_dir().join("whisper");
    let model = Whisper::load(&directory)?;
    let model = if fast { model.fast()? } else { model };
    let vocabulary = Vocabulary::load(&directory)?;
    let wav = Wav::open(directory.join("jfk.wav"))?;
    let prompt = vocabulary.prompt("en")?;
    let plan = Plan {
        warmup: 2,
        repetitions: 11,
    };
    let median = |samples: Vec<f64>| Summary::new(&samples).median * 1e3;

    let mel_time = median(measure(plan, || {
        black_box(LogMel::WHISPER.spectrogram(&wav.samples));
    }));
    let mel = LogMel::WHISPER.spectrogram(&wav.samples);
    let encode_time = median(measure(plan, || {
        black_box(model.encode(&mel.data).expect("인코더").output);
    }));
    let encoded = model.encode(&mel.data)?;
    let mut tokens = 0;
    let greedy_time = median(measure(plan, || {
        let transcript = model
            .greedy(&encoded, &vocabulary, &prompt, 200)
            .expect("디코더");
        tokens = transcript.tokens.len() + 1;
    }));
    let beam_time = median(measure(plan, || {
        black_box(
            model
                .beam(&encoded, &vocabulary, &prompt, 5, 200)
                .expect("빔"),
        );
    }));
    println!(
        "whisper_tiny {} 스레드 {:>2}: 멜 {mel_time:>6.1}ms, 인코더 {encode_time:>7.1}ms, 탐욕 {greedy_time:>6.1}ms (토큰당 {:.2}ms), 빔 5 {beam_time:>6.1}ms, 모두 {:>7.1}ms",
        if fast { "빠른 길" } else { "기본   " },
        threads::global().pool().threads(),
        greedy_time / (tokens + prompt.len()) as f64,
        mel_time + encode_time + greedy_time
    );
    Ok(())
}
