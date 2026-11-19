//! 33장: 소리 파일을 멜 스펙트로그램으로 바꾸고 Python의 것과 견준다.
//!
//! 실행: `cargo run --release -p badak_audio --example ch33_mel`

use std::path::PathBuf;
use std::time::Instant;

use badak_audio::fft::{Bluestein, Complex, dft, fft};
use badak_audio::{LogMel, Wav};
use badak_formats::safetensors::SafeTensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::var_os("BADAK_MODELS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        PathBuf::from,
    );
    let directory = directory.join("whisper");
    let wav = Wav::open(directory.join("jfk.wav"))?;
    println!(
        "jfk.wav: {}Hz, 표본 {}개, {:.2}초",
        wav.sample_rate,
        wav.samples.len(),
        wav.seconds()
    );

    let start = Instant::now();
    let picture = LogMel::WHISPER.spectrogram(&wav.samples);
    let elapsed = start.elapsed();
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let expected = reference.tensor("jfk.mel")?.to_vec::<f32>()?;
    let gap = picture
        .data
        .iter()
        .zip(&expected)
        .map(|(ours, theirs)| (ours - theirs).abs())
        .fold(0.0, f32::max);
    let (low, high) = picture
        .data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    println!(
        "멜 스펙트로그램: {} × {} (값 {low:.3} ~ {high:.3}), {:.1}ms",
        picture.mels,
        picture.frames,
        elapsed.as_secs_f64() * 1e3
    );
    println!("  Python(transformers)의 것과의 가장 큰 차이: {gap:.2e}");

    // 토막 하나(표본 400개)의 변환을 세 가지로 해 본다.
    let frame: Vec<Complex> = wav.samples[16_000..16_400]
        .iter()
        .map(|&sample| Complex {
            re: f64::from(sample),
            im: 0.0,
        })
        .collect();
    let time = |work: &mut dyn FnMut()| {
        let mut best = f64::INFINITY;
        for _ in 0..200 {
            let start = Instant::now();
            work();
            best = best.min(start.elapsed().as_secs_f64());
        }
        best * 1e6
    };
    let plan = Bluestein::new(400);
    let (mut scratch, mut buffer) = (Vec::new(), frame.clone());
    let slow = time(&mut || {
        std::hint::black_box(dft(&frame));
    });
    let any = time(&mut || {
        buffer.copy_from_slice(&frame);
        plan.transform(&mut buffer, &mut scratch);
    });
    let mut padded = vec![Complex::default(); 512];
    let fast = time(&mut || {
        padded[..400].copy_from_slice(&frame);
        padded[400..].fill(Complex::default());
        fft(&mut padded, false);
    });
    println!("\n토막 하나의 푸리에 변환");
    println!("  정의 그대로, 길이 400           {slow:>8.1}µs");
    println!("  Bluestein, 길이 400             {any:>8.1}µs");
    println!("  FFT, 길이 512 (0을 붙여서)      {fast:>8.1}µs");
    if gap > 1e-3 {
        return Err("멜 스펙트로그램이 Python의 것과 다릅니다".into());
    }
    Ok(())
}
