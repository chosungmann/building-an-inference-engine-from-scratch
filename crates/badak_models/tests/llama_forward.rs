//! 모델 전체의 순전파를 PyTorch의 계산과 대조한다 (10장).
//!
//! 모델 파일이 없으면 건너뛴다. `sh scripts/download_smollm2.sh` 또는
//! `sh scripts/download_llama32.sh`로 받는다.

use badak_core::approx::compare;
use badak_core::{Tensor, ops};
use badak_formats::safetensors::SafeTensors;
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

/// 블록을 여러 개 지난 결과에 허용하는 상대 오차.
const RTOL: f32 = 1e-4;

fn magnitude(values: &[f32]) -> f32 {
    values.iter().fold(0.0, |max, value| max.max(value.abs()))
}

/// 전체와, 첫 줄(첫 토큰)을 뺀 나머지를 따로 비교한다. 첫 줄의 큰 값이
/// 나머지 줄의 오차를 가리지 못하게 한다.
fn assert_close(name: &str, actual: &Tensor, expected: &Tensor) {
    let width = expected.shape()[1];
    let (actual, expected) = (
        actual.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap(),
    );
    for start in [0, width] {
        let (actual, expected) = (&actual[start..], &expected[start..]);
        let report = compare(actual, expected, RTOL, RTOL * magnitude(expected));
        assert!(report.is_close(), "{name} ({start}번째 값부터): {report}");
    }
}

fn check_forward(name: &str) {
    let directory = models_dir().join(name);
    let reference = directory.join("reference.safetensors");
    if !directory.join("model.safetensors").exists() || !reference.exists() {
        eprintln!(
            "건너뜀: {}에 모델이나 정답 파일이 없습니다",
            directory.display()
        );
        return;
    }
    let model = LlamaModel::load(&directory).unwrap();
    let reference = SafeTensors::open(reference).unwrap();
    let ids: Vec<TokenId> = reference
        .tensor("ids")
        .unwrap()
        .to_vec::<i32>()
        .unwrap()
        .iter()
        .map(|&id| id as TokenId)
        .collect();

    // 토크나이저도 같은 번호를 내는지 본다. 정답 파일의 글에는 앞에 붙이는
    // 토큰이 없다(9장에서 그렇게 만들었다).
    let tokenizer = Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap();
    let encoded = tokenizer.encode(&reference.metadata()["prompt"], false);
    assert_eq!(encoded, ids, "토큰 번호");

    let expected_states = reference.tensor("hidden_states").unwrap();
    let states = model.hidden_states(&ids).unwrap();
    assert_eq!(states.len(), model.config.num_layers + 1);
    let last = states.len() - 1;
    for (index, state) in states.iter().enumerate() {
        let expected = expected_states
            .slice(0, index..index + 1)
            .unwrap()
            .squeeze(0)
            .unwrap();
        if index == last {
            let normed = model.final_norm(state).unwrap();
            assert_close("마지막 정규화", &normed, &expected);
        } else {
            assert_close(&format!("단계 {index}"), state, &expected);
        }
    }

    let expected = reference.tensor("logits").unwrap();
    let logits = model.logits(&states[last]).unwrap();
    assert_close("점수", &logits, &expected);
    assert_eq!(
        ops::argmax(&logits, 1).unwrap().to_vec::<i32>().unwrap(),
        ops::argmax(&expected, 1).unwrap().to_vec::<i32>().unwrap(),
        "가장 높은 점수의 토큰"
    );

    // 마지막 줄만 계산한 점수는 전체의 마지막 줄과 같아야 한다.
    let next = model.next_token_logits(&ids).unwrap();
    let rows = logits.shape()[0];
    let last_row = logits.slice(0, rows - 1..rows).unwrap();
    assert_eq!(
        next.to_vec::<f32>().unwrap(),
        last_row.to_vec::<f32>().unwrap()
    );
}

#[test]
fn smollm2_forward_matches_pytorch() {
    check_forward("smollm2_135m");
}

#[test]
fn llama32_forward_matches_pytorch() {
    check_forward("llama32_1b");
}
