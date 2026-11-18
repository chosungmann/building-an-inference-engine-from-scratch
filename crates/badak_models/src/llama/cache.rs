//! KV 캐시: 한 번 계산한 키와 값을 모아 두고 다시 쓴다 (11장).

use std::borrow::Cow;

use badak_core::linalg::dot;
use badak_core::prob::softmax_in_place;
use badak_core::quant::F16;
use badak_core::{Tensor, arena, ops};
use badak_formats::Result;

use super::{BlockWeights, LlamaConfig, mlp, rope};

// ANCHOR: cache
/// 블록 하나가 지금까지 본 토큰들의 키와 값.
///
/// 둘 다 `(토큰 수, 키·값 헤드 수, head_dim)`의 값을 한 줄로 늘어놓은
/// 것이다. 키는 위치에 따라 돌린 뒤의 값을 넣는다.
#[derive(Debug)]
pub struct LayerCache {
    // 21장: `f32`로 담거나 16비트로 줄여 담는다. 처음에는 `Vec<f32>`였다.
    keys: Store,
    values: Store,
}

// ANCHOR_END: cache
// ANCHOR: precision
/// 캐시에 값을 담는 정밀도 (21장).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPrecision {
    /// 계산한 `f32`를 그대로 담는다.
    #[default]
    F32,
    /// 16비트 실수로 줄여 담는다. 메모리가 절반이고 값이 조금 달라진다.
    F16,
}

impl KvPrecision {
    /// 환경 변수 `BADAK_KV`(`f32`, `f16`)로 정한 정밀도. 없으면 `f32`다.
    pub fn from_environment() -> Self {
        match std::env::var("BADAK_KV").as_deref() {
            Ok("f16") => Self::F16,
            _ => Self::F32,
        }
    }
}

/// 키나 값이 담기는 곳.
#[derive(Debug)]
enum Store {
    F32(Vec<f32>),
    F16(Vec<F16>),
}

impl Store {
    fn new(precision: KvPrecision) -> Self {
        match precision {
            KvPrecision::F32 => Self::F32(Vec::new()),
            KvPrecision::F16 => Self::F16(Vec::new()),
        }
    }

    /// 새 값들을 덧붙인다. 16비트로 담을 때는 가장 가까운 값으로 줄인다.
    fn extend(&mut self, values: &[f32]) {
        match self {
            Self::F32(stored) => stored.extend_from_slice(values),
            Self::F16(stored) => stored.extend(values.iter().map(|&value| F16::from_f32(value))),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            Self::F32(stored) => stored.truncate(len),
            Self::F16(stored) => stored.truncate(len),
        }
    }

    /// 값이 모두 `len`개가 될 때까지 다시 자리를 잡지 않아도 되게 한다.
    fn reserve_total(&mut self, len: usize) {
        match self {
            Self::F32(stored) => stored.reserve(len.saturating_sub(stored.len())),
            Self::F16(stored) => stored.reserve(len.saturating_sub(stored.len())),
        }
    }

    /// 담긴 값 전체를 `f32`로 본다. 16비트로 담았으면 펼친 사본을 만든다.
    fn as_f32(&self) -> Cow<'_, [f32]> {
        match self {
            Self::F32(stored) => Cow::Borrowed(stored),
            Self::F16(stored) => Cow::Owned(stored.iter().map(|half| half.to_f32()).collect()),
        }
    }
}
// ANCHOR_END: precision
// ANCHOR: cache

/// 모델 전체의 KV 캐시. 블록마다 하나씩 있다.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    /// 지금까지 넣은 토큰의 수. 다음 토큰의 위치이기도 하다.
    tokens: usize,
    /// 토큰 하나가 블록 하나에서 차지하는 키(또는 값)의 길이.
    width: usize,
    /// 값을 담는 정밀도 (21장).
    precision: KvPrecision,
}

impl KvCache {
    /// 빈 캐시를 만든다.
    ///
    /// 21장: 값을 담는 정밀도는 환경 변수 `BADAK_KV`를 따르고, 없으면 `f32`다.
    pub fn new(config: &LlamaConfig) -> Self {
        Self::with_precision(config, KvPrecision::from_environment())
    }

    /// 값을 담는 정밀도를 정해서 빈 캐시를 만든다 (21장).
    pub fn with_precision(config: &LlamaConfig, precision: KvPrecision) -> Self {
        Self {
            layers: (0..config.num_layers)
                .map(|_| LayerCache {
                    keys: Store::new(precision),
                    values: Store::new(precision),
                })
                .collect(),
            tokens: 0,
            width: config.num_kv_heads * config.head_dim(),
            precision,
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
        // 블록마다 키와 값, 토큰마다 값 `width`개.
        let element = match self.precision {
            KvPrecision::F32 => size_of::<f32>(),
            KvPrecision::F16 => size_of::<F16>(),
        };
        self.layers.len() * 2 * self.tokens * self.width * element
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

    // ANCHOR: reserve
    /// 토큰이 모두 `tokens`개가 될 때까지 쓸 자리를 미리 잡는다 (22장).
    ///
    /// 잡아 두지 않으면 캐시가 자랄 때마다 더 큰 자리를 새로 얻어 지금까지의
    /// 값을 옮겨 적는다. 글의 길이를 아는 쪽이 처음에 한 번 불러 두면 그
    /// 일이 없다.
    pub fn reserve(&mut self, tokens: usize) {
        for layer in &mut self.layers {
            layer.keys.reserve_total(tokens * self.width);
            layer.values.reserve_total(tokens * self.width);
        }
    }
    // ANCHOR_END: reserve

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
    cache.keys.extend(k.as_slice::<f32>()?);
    cache.values.extend(v.contiguous().as_slice::<f32>()?);
    // 21장: 캐시가 16비트이면 여기서 한 번 `f32`로 펼쳐 쓴다. 방금 넣은 새
    // 토큰의 키와 값도 16비트를 거친 값으로 읽게 된다.
    let (k, v) = (cache.keys.as_f32(), cache.values.as_f32());
    let (q, k, v) = (q.as_slice::<f32>()?, &k[..], &v[..]);

    let vector = |count: usize, token: usize, head: usize| {
        let start = (token * count + head) * dim;
        start..start + dim
    };
    let scale = 1.0 / (dim as f32).sqrt();
    // ANCHOR: by_tokens
    // 18장: 새 토큰마다 따로 계산할 수 있으므로 토큰을 스레드에 나눈다. 뒤의
    // 토큰일수록 볼 키가 많아 일이 크다. 스레드들이 토큰을 하나씩 뽑아 가므로
    // 일의 크기가 달라도 고르게 나뉜다.
    let threads = badak_core::threads::global();
    let count = threads.for_vectors(new_tokens);
    let mix = |token: usize, output: &mut [f32], attention: &mut [f32]| {
        for (head, mixed) in output.chunks_mut(dim).enumerate() {
            let kv_head = head / config.heads_per_kv();
            // 4. 새 토큰의 질의를, 자기까지의 모든 토큰의 키와 내적한다.
            let query = &q[vector(heads, token, head)];
            for (earlier, score) in attention.iter_mut().enumerate() {
                *score = dot(query, &k[vector(kv_heads, earlier, kv_head)]) * scale;
            }
            softmax_in_place(attention);

            // 5. 값들을 비중대로 섞는다.
            for (earlier, &weight) in attention.iter().enumerate() {
                let value = &v[vector(kv_heads, earlier, kv_head)];
                for (sum, value) in mixed.iter_mut().zip(value) {
                    *sum += weight * value;
                }
            }
        }
    };
    let mixed = Tensor::build(&[new_tokens, heads * dim], |output| {
        threads
            .pool()
            .run_chunks(output, heads * dim, count, &|token, output| {
                // 22장: 점수와 비중을 담을 자리는 스레드가 돌려 쓰는 것을 빌려,
                // 헤드들이 함께 쓴다. 헤드마다 새로 만들면 프롬프트 하나에
                // 할당이 수십만 번이 된다.
                arena::with_scratch(past + token + 1, |attention| mix(token, output, attention));
            });
        Ok(())
    })?;

    // ANCHOR_END: by_tokens
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

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn sixteen_bit_store_rounds_on_the_way_in_and_halves_the_size() {
        let values = [1.0_f32, -2.5, 0.1, 3.0e-5];
        let mut exact = Store::new(KvPrecision::F32);
        let mut half = Store::new(KvPrecision::F16);
        exact.extend(&values);
        half.extend(&values);
        assert_eq!(&exact.as_f32()[..], &values);
        // 16비트로 정확히 담기는 값은 그대로, 나머지는 가장 가까운 값으로.
        let restored = half.as_f32();
        assert_eq!(restored[..2], values[..2]);
        assert!(restored[2] != 0.1 && (restored[2] - 0.1).abs() < 1e-4);
        half.truncate(2);
        assert_eq!(half.as_f32().len(), 2);

        let config = LlamaConfig {
            vocab_size: 8,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        };
        let mut wide = KvCache::with_precision(&config, KvPrecision::F32);
        let mut narrow = KvCache::with_precision(&config, KvPrecision::F16);
        wide.advance(10);
        narrow.advance(10);
        assert_eq!(wide.size_in_bytes(), 2 * narrow.size_in_bytes());
    }
}
