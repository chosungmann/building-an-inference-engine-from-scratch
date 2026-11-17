//! 생성 루프: 토큰을 하나씩 뽑아 글을 이어 쓴다 (11장).

use badak_core::prob::argmax;
use badak_formats::Result;

use super::cache::KvCache;
use super::model::LlamaModel;
use crate::sampling::Sampler;
use crate::tokenizer::TokenId;

/// 점수가 가장 높은 토큰을 고른다.
fn pick(logits: &[f32]) -> TokenId {
    argmax(logits).expect("점수에 유효한 값이 있다") as TokenId
}

// ANCHOR: generate_without_cache
/// 캐시 없이 글을 이어 쓴다. 토큰을 하나 뽑을 때마다 글 전체를 처음부터
/// 다시 계산한다.
///
/// 새 토큰이 나올 때마다 `on_token`을 부른다. 글의 끝을 알리는 토큰이
/// 나오거나 `max_new_tokens`개를 뽑으면 멈춘다. 돌려주는 값은 새로 뽑은
/// 토큰들이다.
pub fn generate_without_cache(
    model: &LlamaModel,
    prompt: &[TokenId],
    max_new_tokens: usize,
    mut on_token: impl FnMut(TokenId),
) -> Result<Vec<TokenId>> {
    let mut ids = prompt.to_vec();
    while ids.len() < prompt.len() + max_new_tokens {
        // 지금까지의 모든 토큰을 넣는다.
        let logits = model.next_token_logits(&ids)?;
        let next = pick(logits.as_slice()?);
        if model.config.eos_token_ids.contains(&next) {
            break;
        }
        ids.push(next);
        on_token(next);
    }
    Ok(ids.split_off(prompt.len()))
}
// ANCHOR_END: generate_without_cache

// ANCHOR: generate
/// 글을 이어 쓴다. KV 캐시를 써서, 한 번 처리한 토큰은 다시 계산하지 않는다.
///
/// `cache`에는 `prompt`의 앞부분이 이미 들어 있을 수 있다. 그러면 나머지만
/// 처리한다. 끝나고 나면 캐시에는 프롬프트와 새로 뽑은 토큰이 (마지막
/// 하나를 빼고) 들어 있다.
pub fn generate(
    model: &LlamaModel,
    prompt: &[TokenId],
    max_new_tokens: usize,
    cache: &mut KvCache,
    mut on_token: impl FnMut(TokenId),
) -> Result<Vec<TokenId>> {
    let mut generated = Vec::new();
    // 처음에는 프롬프트에서 캐시에 없는 부분을 한꺼번에 넣는다.
    let mut input = prompt[cache.len()..].to_vec();
    while generated.len() < max_new_tokens {
        let logits = model.forward_cached(&input, cache)?;
        let next = pick(logits.as_slice()?);
        if model.config.eos_token_ids.contains(&next) {
            break;
        }
        generated.push(next);
        on_token(next);
        // 그다음부터는 방금 뽑은 토큰 하나만 넣는다.
        input = vec![next];
    }
    Ok(generated)
}
// ANCHOR_END: generate

// ANCHOR: generate_sampled
/// 글을 이어 쓴다. 다음 토큰을 `sampler`가 뽑는다는 점만 `generate`와 다르다.
///
/// 샘플러에는 프롬프트와 지금까지 뽑은 토큰을 함께 넘긴다. 반복 벌점이
/// 그것을 본다.
pub fn generate_sampled(
    model: &LlamaModel,
    prompt: &[TokenId],
    max_new_tokens: usize,
    cache: &mut KvCache,
    sampler: &mut Sampler,
    mut on_token: impl FnMut(TokenId),
) -> Result<Vec<TokenId>> {
    let mut history = prompt.to_vec();
    let mut input = prompt[cache.len()..].to_vec();
    while history.len() < prompt.len() + max_new_tokens {
        let logits = model.forward_cached(&input, cache)?;
        let next = sampler.sample(logits.as_slice()?, &history);
        if model.config.eos_token_ids.contains(&next) {
            break;
        }
        history.push(next);
        on_token(next);
        input = vec![next];
    }
    Ok(history.split_off(prompt.len()))
}
// ANCHOR_END: generate_sampled
