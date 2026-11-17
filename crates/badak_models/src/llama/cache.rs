//! KV 캐시: 한 번 계산한 키와 값을 모아 두고 다시 쓴다 (11장).

use badak_core::linalg::dot;
use badak_core::prob::softmax;
use badak_core::{Tensor, ops};
use badak_formats::Result;

use super::{BlockWeights, LlamaConfig, mlp, rope};

// ANCHOR: cache
/// 블록 하나가 지금까지 본 토큰들의 키와 값.
///
/// 둘 다 `(토큰 수, 키·값 헤드 수, head_dim)`의 값을 한 줄로 늘어놓은
/// 것이다. 키는 위치에 따라 돌린 뒤의 값을 넣는다.
#[derive(Debug, Default)]
pub struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// 모델 전체의 KV 캐시. 블록마다 하나씩 있다.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    /// 지금까지 넣은 토큰의 수. 다음 토큰의 위치이기도 하다.
    tokens: usize,
    /// 토큰 하나가 블록 하나에서 차지하는 키(또는 값)의 길이.
    width: usize,
}

impl KvCache {
    /// 빈 캐시를 만든다.
    pub fn new(config: &LlamaConfig) -> Self {
        Self {
            layers: (0..config.num_layers)
                .map(|_| LayerCache::default())
                .collect(),
            tokens: 0,
            width: config.num_kv_heads * config.head_dim(),
        }
    }

    /// 캐시에 든 토큰의 수.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// 캐시가 비었는가.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// 캐시가 차지하는 메모리. 바이트.
    pub fn size_in_bytes(&self) -> usize {
        // 블록마다 키와 값, 토큰마다 `width`개의 `f32`.
        self.layers.len() * 2 * self.tokens * self.width * size_of::<f32>()
    }

    /// 앞의 `tokens`개만 남기고 나머지를 버린다.
    ///
    /// 앞부분이 같은 다른 글을 이어서 처리할 때, 같은 부분의 계산을 다시
    /// 하지 않으려고 쓴다.
    pub fn truncate(&mut self, tokens: usize) {
        if tokens < self.tokens {
            for layer in &mut self.layers {
                layer.keys.truncate(tokens * self.width);
                layer.values.truncate(tokens * self.width);
            }
            self.tokens = tokens;
        }
    }

    /// 블록들의 캐시. 순전파가 블록마다 하나씩 꺼내 쓴다.
    pub(super) fn layers_mut(&mut self) -> &mut [LayerCache] {
        &mut self.layers
    }

    /// 새 토큰 `count`개를 넣었다고 기록한다.
    pub(super) fn advance(&mut self, count: usize) {
        self.tokens += count;
    }
}
// ANCHOR_END: cache

// ANCHOR: attention_cached
/// 캐시를 쓰는 자기 어텐션. `x`는 **새 토큰들만**의 `(새 토큰 수,
/// hidden_size)`다. `past`는 캐시에 이미 든 토큰의 수다.
///
/// 9장의 `attention`과 계산은 같다. 다른 점은 셋이다. 질의, 키, 값을 새
/// 토큰에 대해서만 만든다. 새 키와 값을 캐시의 끝에 덧붙인다. 그리고
/// 점수를 낼 때 키와 값을 캐시에서 읽는다.
pub fn attention_cached(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
    cache: &mut LayerCache,
    past: usize,
) -> Result<Tensor> {
    // 14장: 프로파일러가 켜져 있으면 이 함수에 쓴 시간을 적는다.
    let _scope = badak_core::profile::scope("attention");
    let new_tokens = x.shape()[0];
    let (heads, kv_heads, dim) = (config.num_heads, config.num_kv_heads, config.head_dim());

    // 1. 새 토큰의 질의, 키, 값을 만든다.
    let split = |projected: Tensor, count: usize| projected.reshape(&[new_tokens, count, dim]);
    let q = split(ops::linear(x, &weights.q_proj, None)?, heads)?;
    let k = split(ops::linear(x, &weights.k_proj, None)?, kv_heads)?;
    let v = split(ops::linear(x, &weights.v_proj, None)?, kv_heads)?;

    // 2. 질의와 키를 돌린다. 새 토큰의 위치는 0이 아니라 `past`에서 시작한다.
    let q = rope::apply(&q, frequencies, past)?;
    let k = rope::apply(&k, frequencies, past)?;

    // 3. 새 키와 값을 캐시에 덧붙인다. 이제 캐시에는 모든 토큰의 것이 있다.
    cache.keys.extend_from_slice(k.as_slice::<f32>()?);
    cache
        .values
        .extend_from_slice(v.contiguous().as_slice::<f32>()?);
    let (q, k, v) = (q.as_slice::<f32>()?, &cache.keys, &cache.values);

    let vector = |count: usize, token: usize, head: usize| {
        let start = (token * count + head) * dim;
        start..start + dim
    };
    let scale = 1.0 / (dim as f32).sqrt();
    let mut output = vec![0.0; new_tokens * heads * dim];
    // ANCHOR: by_tokens
    // 18장: 새 토큰마다 따로 계산할 수 있으므로 토큰을 스레드에 나눈다. 뒤의
    // 토큰일수록 볼 키가 많아 일이 크다. 스레드들이 토큰을 하나씩 뽑아 가므로
    // 일의 크기가 달라도 고르게 나뉜다.
    let threads = badak_core::threads::global();
    let count = threads.for_vectors(new_tokens);
    threads
        .pool()
        .run_chunks(&mut output, heads * dim, count, &|token, output| {
            let position = past + token;
            for (head, mixed) in output.chunks_mut(dim).enumerate() {
                let kv_head = head / config.heads_per_kv();
                // 4. 새 토큰의 질의를, 자기까지의 모든 토큰의 키와 내적한다.
                let query = &q[vector(heads, token, head)];
                let scores: Vec<f32> = (0..=position)
                    .map(|earlier| dot(query, &k[vector(kv_heads, earlier, kv_head)]) * scale)
                    .collect();
                let attention = softmax(&scores);

                // 5. 값들을 비중대로 섞는다.
                for (earlier, &weight) in attention.iter().enumerate() {
                    let value = &v[vector(kv_heads, earlier, kv_head)];
                    for (sum, value) in mixed.iter_mut().zip(value) {
                        *sum += weight * value;
                    }
                }
            }
        });

    // ANCHOR_END: by_tokens
    let mixed = Tensor::from_vec(output, &[new_tokens, heads * dim])?;
    Ok(ops::linear(&mixed, &weights.o_proj, None)?)
}
// ANCHOR_END: attention_cached

// ANCHOR: block_cached
/// 캐시를 쓰는 블록. 10장의 `block`에서 어텐션만 바꾼 것이다.
pub fn block_cached(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
    cache: &mut LayerCache,
    past: usize,
) -> Result<Tensor> {
    let normed = ops::rms_norm(x, &weights.attention_norm, config.rms_norm_eps)?;
    let attended = attention_cached(&normed, weights, config, frequencies, cache, past)?;
    let h = ops::add(x, &attended)?;

    let normed = ops::rms_norm(&h, &weights.mlp_norm, config.rms_norm_eps)?;
    Ok(ops::add(&h, &mlp(&normed, weights)?)?)
}
// ANCHOR_END: block_cached
