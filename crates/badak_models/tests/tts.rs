//! OuteTTS-0.1-350M과 오디오 디코더를 PyTorch의 정답과 견준다 (37장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch37.sh`로 받는다. 디코더의
//! 가중치는 `reference/ch37_convert.py`로 만들어야 하고, 없으면 그 부분만
//! 건너뛴다.

use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::{TokenId, Tokenizer};
use badak_models::tts::{OuteTts, SAMPLES_PER_CODE, Voice, prompt};

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn speaks_like_the_reference() {
    let directory = models_dir().join("outetts");
    if !directory.join("model.safetensors").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let floats = |key: &str| reference.tensor(key).unwrap().to_vec::<f32>().unwrap();
    let ids = |key: &str| -> Vec<TokenId> {
        reference
            .tensor(key)
            .unwrap()
            .to_vec::<i32>()
            .unwrap()
            .iter()
            .map(|&id| id as TokenId)
            .collect()
    };
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap();
    let text = "The engine reads a model file, and turns text into speech!";
    let prompt_ids = tokenizer.encode(&prompt(text), true);
    assert_eq!(prompt_ids, ids("ids"));

    // 오디오 토큰 4,096개의 번호가 이어져 있다.
    let first_code = tokenizer.token_id("<|0|>").unwrap();
    assert_eq!(tokenizer.token_id("<|4095|>").unwrap(), first_code + 4095);

    // 언어 모델. 앞의 40토큰만 쓴다.
    let model = OuteTts::load(&directory).unwrap();
    let stop = tokenizer.token_id("<|audio_end|>").unwrap();
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.1,
        ..SamplerConfig::default()
    });
    let mut seen = 0;
    let (written, logits) = model
        .generate(&prompt_ids, stop, 40, &mut sampler, |_| seen += 1)
        .unwrap();
    assert!(largest_gap(&logits, &floats("logits")) < 1e-2);
    assert_eq!(written, ids("written")[..40]);
    assert_eq!(seen, 40);

    // 오디오 디코더.
    let weights = directory.join("wavtokenizer.safetensors");
    if !weights.exists() {
        eprintln!("건너뜀: 디코더의 가중치가 없습니다 (reference/ch37_convert.py)");
        return;
    }
    let codes = ids("codes");
    let expected = floats("audio");
    for voice in [
        Voice::load(&weights).unwrap(),
        Voice::load(&weights).unwrap().fast(),
    ] {
        let hidden = voice.backbone().forward(&codes).unwrap();
        // 값의 크기가 50쯤이다.
        assert!(largest_gap(&hidden, &floats("hidden")) < 2e-2);
        let audio = voice.decode(&codes).unwrap();
        assert_eq!(audio.len(), codes.len() * SAMPLES_PER_CODE);
        assert!(largest_gap(&audio, &expected) < 5e-3);

        // 앞뒤를 모두 넣어 조각을 꺼내면 한 번에 푼 것의 그 자리와 같다.
        let part = voice
            .decode_part(&codes, 50, 80, usize::MAX, usize::MAX)
            .unwrap();
        assert!(largest_gap(&part, &audio[50 * SAMPLES_PER_CODE..80 * SAMPLES_PER_CODE]) < 1e-5);
        // 조각만 따로 풀면 길이는 같고 값은 다르다.
        let alone = voice.decode_part(&codes, 50, 80, 0, 0).unwrap();
        assert_eq!(alone.len(), part.len());
        assert!(largest_gap(&alone, &part) > 1e-2);
        // 코드북 밖의 코드는 거절한다.
        assert!(voice.decode(&[5000]).is_err());
    }
}
