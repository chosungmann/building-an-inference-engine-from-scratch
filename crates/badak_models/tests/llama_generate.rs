//! KV 캐시와 생성 루프를 검증한다 (11장).
//!
//! 모델 파일이 없으면 건너뛴다. `sh scripts/download_smollm2.sh` 또는
//! `sh scripts/download_llama32.sh`로 받는다.

use badak_formats::json::{self, Value};
use badak_models::llama::cache::KvCache;
use badak_models::llama::generate::{generate, generate_without_cache};
use badak_models::llama::model::LlamaModel;
use badak_models::models_dir;
use badak_models::tokenizer::{TokenId, Tokenizer};

fn ids(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_usize().unwrap() as TokenId)
        .collect()
}

/// 모델, 토크나이저, PyTorch가 이어 쓴 결과를 연다. 없으면 `None`.
fn open(name: &str) -> Option<(LlamaModel, Tokenizer, Vec<Value>)> {
    let directory = models_dir().join(name);
    let reference = directory.join("generation.json");
    if !directory.join("model.safetensors").exists() || !reference.exists() {
        eprintln!(
            "건너뜀: {}에 모델이나 정답 파일이 없습니다",
            directory.display()
        );
        return None;
    }
    let cases = json::parse(&std::fs::read_to_string(reference).unwrap()).unwrap();
    Some((
        LlamaModel::load(&directory).unwrap(),
        Tokenizer::from_tokenizer_json(directory.join("tokenizer.json")).unwrap(),
        cases.as_array().unwrap().to_vec(),
    ))
}

// ANCHOR: cache_is_exact
/// 캐시를 쓴 점수가 쓰지 않은 점수와 비트 단위로 같은지 본다.
fn check_cache_is_exact(model: &LlamaModel, tokens: &[TokenId]) {
    // 토큰을 두 개, 세 개, 한 개, … 로 나누어 넣어 본다.
    let mut cache = KvCache::new(&model.config);
    let mut done = 0;
    for step in [2, 3, 1].into_iter().cycle() {
        let end = (done + step).min(tokens.len());
        let cached = model
            .forward_cached(&tokens[done..end], &mut cache)
            .unwrap();
        let full = model.next_token_logits(&tokens[..end]).unwrap();
        assert_eq!(
            cached.to_vec::<f32>().unwrap(),
            full.to_vec::<f32>().unwrap(),
            "토큰 {end}개까지 넣었을 때"
        );
        assert_eq!(cache.len(), end);
        done = end;
        if done == tokens.len() {
            break;
        }
    }

    // 뒤를 잘라 내고 다른 토큰을 이어도 같아야 한다.
    let keep = tokens.len() / 2;
    cache.truncate(keep);
    assert_eq!(cache.len(), keep);
    let mut other = tokens[..keep].to_vec();
    other.extend([tokens[0], tokens[1]]);
    let cached = model.forward_cached(&other[keep..], &mut cache).unwrap();
    let full = model.next_token_logits(&other).unwrap();
    assert_eq!(
        cached.to_vec::<f32>().unwrap(),
        full.to_vec::<f32>().unwrap()
    );
}
// ANCHOR_END: cache_is_exact

fn check_generation(name: &str) {
    let Some((model, tokenizer, cases)) = open(name) else {
        return;
    };
    for case in &cases {
        let text = case.get("text").unwrap().as_str().unwrap();
        let prompt = ids(case.get("ids").unwrap());
        let expected = ids(case.get("generated").unwrap());

        // 토크나이저: 설정이 정한 토큰을 앞에 붙이면 PyTorch 쪽과 같아야 한다.
        let mut encoded = tokenizer.prefix().to_vec();
        encoded.extend(tokenizer.encode(text, true));
        assert_eq!(encoded, prompt, "토큰 번호: {text:?}");

        // 같은 글에서 PyTorch와 같은 토큰을 뽑아야 한다.
        let mut cache = KvCache::new(&model.config);
        let mut seen = Vec::new();
        let generated = generate(&model, &prompt, expected.len(), &mut cache, |token| {
            seen.push(token)
        })
        .unwrap();
        assert_eq!(generated, expected, "이어 쓴 토큰: {text:?}");
        assert_eq!(seen, expected);
        // 마지막에 뽑은 토큰은 아직 모델에 넣지 않았다.
        assert_eq!(cache.len(), prompt.len() + expected.len() - 1);

        // 캐시 없는 생성도 같은 토큰을 뽑는다. 느리므로 앞의 몇 개만 본다.
        let few = expected.len().min(4);
        let slow = generate_without_cache(&model, &prompt, few, |_| {}).unwrap();
        assert_eq!(slow, expected[..few]);
    }

    let first = ids(cases[0].get("ids").unwrap());
    let mut tokens = first.clone();
    tokens.extend(ids(cases[0].get("generated").unwrap()).iter().take(6));
    check_cache_is_exact(&model, &tokens);
}

#[test]
fn smollm2_generates_what_pytorch_generates() {
    check_generation("smollm2_135m");
}

#[test]
fn llama32_generates_what_pytorch_generates() {
    check_generation("llama32_1b");
}
