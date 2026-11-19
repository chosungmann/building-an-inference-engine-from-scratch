//! SmolVLM-256M을 transformers의 정답과 견준다 (36장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch36.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::smolvlm::SmolVlm;
use badak_models::tokenizer::{TokenId, Tokenizer};

fn largest_gap(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn sees_and_describes_like_the_reference() {
    let directory = models_dir().join("smolvlm");
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
    let stop = tokenizer.token_id("<end_of_utterance>").unwrap();
    let basic = SmolVlm::load(&directory).unwrap();
    assert_eq!(
        (
            basic.vision.image_size(),
            basic.vision.patches(),
            basic.vision.width(),
            basic.image_tokens(),
            basic.text.config.hidden_size,
        ),
        (512, 1024, 768, 64, 576)
    );

    // 글의 틀을 토큰으로 바꾼 것이 정답과 같다. 사진 자리는 64개다.
    let prompt = format!(
        "<|im_start|>User:<fake_token_around_image><global-img>{}<fake_token_around_image>Describe this image.<end_of_utterance>\nAssistant:",
        "<image>".repeat(basic.image_tokens())
    );
    let prompt = tokenizer.encode(&prompt, true);
    assert_eq!(prompt, ids("tabby_cat.ids"));
    assert_eq!(
        prompt.iter().filter(|&&id| id == basic.image_token).count(),
        64
    );

    let pixels = floats("tabby_cat.pixels");
    let expected = ids("tabby_cat.written");
    let fast = SmolVlm::load(&directory).unwrap().fast().unwrap();
    for model in [&basic, &fast] {
        // 값의 크기가 20을 넘는다.
        let patches = model.vision.encode(&pixels).unwrap();
        assert!(largest_gap(&patches, &floats("tabby_cat.patches")) < 2e-2);
        let seen = model.connect(&patches).unwrap();
        assert_eq!(seen.len(), 64 * 576);
        assert!(largest_gap(&seen, &floats("tabby_cat.seen")) < 2e-2);

        let (written, logits) = model.generate(&prompt, &seen, stop, 12, |_| {}).unwrap();
        assert!(largest_gap(&logits, &floats("tabby_cat.logits")) < 2e-2);
        assert_eq!(written, expected[..12]);

        // 사진의 벡터와 사진 자리의 수가 맞지 않으면 거절한다.
        assert!(model.embed(&prompt, &seen[..576]).is_err());
        assert!(model.embed(&prompt[..4], &seen).is_err());
    }
    // 크기가 다른 사진은 거절한다.
    assert!(basic.vision.encode(&pixels[..100]).is_err());
}
