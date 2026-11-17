//! 샘플러가 점수를 손질한 결과를 transformers의 구현과 대조한다 (12장).
//!
//! 정답 파일이 없으면 건너뛴다. `sh scripts/download_smollm2.sh` 또는
//! `sh scripts/download_llama32.sh`로 받는다.

use badak_core::approx::compare;
use badak_formats::json::{self, Value};
use badak_formats::safetensors::SafeTensors;
use badak_models::models_dir;
use badak_models::sampling::{Sampler, SamplerConfig};
use badak_models::tokenizer::TokenId;

// ANCHOR: check
fn check_distributions(name: &str) {
    let directory = models_dir().join(name);
    let (reference, sampling) = (
        directory.join("reference.safetensors"),
        directory.join("sampling.safetensors"),
    );
    if !reference.exists() || !sampling.exists() {
        eprintln!("건너뜀: {}에 정답 파일이 없습니다", directory.display());
        return;
    }
    // 입력: 9장의 정답 파일에 든 마지막 줄의 점수와, 그 글의 토큰들.
    let reference = SafeTensors::open(reference).unwrap();
    let logits = reference.tensor("logits").unwrap();
    let rows = logits.shape()[0];
    let logits = logits
        .slice(0, rows - 1..rows)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let history: Vec<TokenId> = reference
        .tensor("ids")
        .unwrap()
        .to_vec::<i32>()
        .unwrap()
        .iter()
        .map(|&id| id as TokenId)
        .collect();

    let sampling = SafeTensors::open(sampling).unwrap();
    let cases = json::parse(&sampling.metadata()["cases"]).unwrap();
    for (index, case) in cases.as_array().unwrap().iter().enumerate() {
        let number = |key: &str| case.get(key).and_then(Value::as_f64);
        let defaults = SamplerConfig::default();
        let config = SamplerConfig {
            temperature: number("temperature").map_or(defaults.temperature, |t| t as f32),
            top_k: number("top_k").map_or(defaults.top_k, |k| k as usize),
            top_p: number("top_p").map_or(defaults.top_p, |p| p as f32),
            repetition_penalty: number("repetition_penalty")
                .map_or(defaults.repetition_penalty, |r| r as f32),
            ..defaults
        };

        // 후보의 목록을 토큰 번호 순서의 확률로 펼친다. 걸러진 토큰은 0이다.
        let mut actual = vec![0.0_f32; logits.len()];
        for candidate in Sampler::new(config).distribution(&logits, &history) {
            actual[candidate.id as usize] = candidate.probability;
        }
        let expected = sampling
            .tensor(&format!("case{index}"))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        // 남은 토큰의 집합이 같아야 하고, 확률이 같아야 한다.
        let kept = |p: &[f32]| p.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(
            kept(&actual),
            kept(&expected),
            "설정 {index}: 남은 토큰의 수"
        );
        let report = compare(&actual, &expected, 1e-5, 1e-6);
        assert!(report.is_close(), "설정 {index}: {report}");
    }
}
// ANCHOR_END: check

#[test]
fn smollm2_distributions_match_transformers() {
    check_distributions("smollm2_135m");
}

#[test]
fn llama32_distributions_match_transformers() {
    check_distributions("llama32_1b");
}
