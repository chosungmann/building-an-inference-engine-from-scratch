//! Whisper-tiny를 transformers의 정답과 견준다 (33장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch33.sh`로 받는다.

use badak_audio::vad::{self, Settings};
use badak_audio::{LogMel, Wav};
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::whisper::{Vocabulary, Whisper};

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn transcribes_like_transformers() {
    let directory = models_dir().join("whisper");
    if !directory.join("model.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let floats = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let tokens = |key: &str| -> Vec<u32> {
        let ids = reference.tensor(key).unwrap().to_vec::<i32>().unwrap();
        ids.into_iter()
            .map(|id| id as u32)
            .filter(|&id| id < 50257)
            .collect()
    };
    let vocabulary = Vocabulary::load(&directory).unwrap();
    assert_eq!(
        vocabulary.prompt("en").unwrap(),
        [50258, 50259, 50359, 50363]
    );
    assert_eq!(vocabulary.end().unwrap(), 50257);
    assert_eq!(vocabulary.special_name(50264), Some("<|ko|>"));
    assert!(vocabulary.special("<|xx|>").is_err());
    assert_eq!(vocabulary.suppressed_at_start, [220, 50257]);
    assert_eq!(vocabulary.decode(&[400, 370, 50257, 13]), " And so.");

    let wav = Wav::open(directory.join("jfk.wav")).unwrap();
    assert_eq!((wav.sample_rate, wav.samples.len()), (16_000, 176_000));
    let mel = LogMel::WHISPER.spectrogram(&wav.samples);
    assert_eq!((mel.mels, mel.frames), (80, 3000));
    assert!(largest_gap(&mel.data, &floats("jfk.mel")) < 1e-3);

    let prompt = vocabulary.prompt("en").unwrap();
    let basic = Whisper::load(&directory).unwrap();
    assert_eq!(
        (basic.vocabulary(), basic.frames(), basic.max_tokens()),
        (51865, 1500, 448)
    );
    // 스펙트로그램의 크기가 다르면 거절한다.
    assert!(basic.encode(&mel.data[..100]).is_err());
    let fast = Whisper::load(&directory).unwrap().fast().unwrap();
    for model in [&basic, &fast] {
        let encoded = model.encode(&mel.data).unwrap();
        assert_eq!(encoded.output.shape(), [1500, 384]);
        let output = encoded.output.to_vec::<f32>().unwrap();
        assert!(largest_gap(&output, &floats("jfk.encoded")) < 2e-2);

        let mut state = model.start();
        let mut logits = Vec::new();
        for &token in &prompt {
            logits = model.step(&encoded, &mut state, token).unwrap();
        }
        assert!(largest_gap(&logits, &floats("jfk.logits")) < 5e-3);
        assert_eq!(
            model.detect_language(&encoded, &vocabulary).unwrap().0,
            "en"
        );

        let greedy = model.greedy(&encoded, &vocabulary, &prompt, 200).unwrap();
        assert_eq!(greedy.tokens, tokens("jfk.greedy"));
        let beam = model.beam(&encoded, &vocabulary, &prompt, 5, 200).unwrap();
        assert_eq!(beam.tokens, tokens("jfk.beam"));
        // 빔 서치가 찾은 글이 탐욕 디코딩의 것보다 확률이 높다.
        assert!(beam.log_probability > greedy.log_probability);
        // 폭이 1인 빔 서치는 탐욕 디코딩이다.
        let single = model.beam(&encoded, &vocabulary, &prompt, 1, 200).unwrap();
        assert_eq!(single.tokens, greedy.tokens);
        // 토큰 수를 막으면 거기서 멈춘다.
        assert_eq!(
            model
                .greedy(&encoded, &vocabulary, &prompt, 3)
                .unwrap()
                .tokens
                .len(),
            3
        );
    }

    // 말소리는 처음부터 끝까지 한 구간이거나, 숨을 쉰 곳에서 몇으로 나뉜다.
    let segments = vad::segments(&wav.samples, wav.sample_rate, &Settings::default());
    assert!(!segments.is_empty() && segments.len() <= 4, "{segments:?}");
    let windows = vad::windows(&segments, LogMel::WHISPER.samples);
    assert_eq!(windows.len(), 1);
}
