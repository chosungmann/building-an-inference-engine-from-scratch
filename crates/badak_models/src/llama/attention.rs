//! 어텐션: 토큰이 앞의 토큰들에서 정보를 가져온다.

// 24장: 내적을 17장의 정의로 바꿨다. 처음에는 2장의 `linalg::dot`이었다.
use badak_core::prob::softmax;
use badak_core::simd::dot;
use badak_core::{Tensor, ops};
use badak_formats::Result;

use super::{BlockWeights, LlamaConfig, rope};

// ANCHOR: attention
/// 어텐션의 결과.
pub struct AttentionOutput {
    /// 블록의 다음 단계로 넘어가는 값. `(토큰 수, hidden_size)`.
    pub output: Tensor,
    /// 토큰이 앞의 토큰 각각에 준 비중. `(헤드 수, 토큰 수, 토큰 수)`.
    pub weights: Tensor,
}

/// 자기 어텐션. `x`는 `(토큰 수, hidden_size)`다.
///
/// 토큰마다 질의, 키, 값 벡터를 만든다. 어떤 토큰의 질의와 앞선 토큰들의
/// 키를 내적해 점수를 얻고, 소프트맥스로 비중을 만들고, 그 비중으로 값
/// 벡터들을 섞는다. 이 일을 헤드마다 따로 한다.
pub fn attention(
    x: &Tensor,
    weights: &BlockWeights,
    config: &LlamaConfig,
    frequencies: &[f32],
) -> Result<AttentionOutput> {
    let tokens = x.shape()[0];
    let (heads, kv_heads, dim) = (config.num_heads, config.num_kv_heads, config.head_dim());

    // 1. 선형 층으로 질의, 키, 값을 만들고 헤드별로 나눈다.
    let split = |projected: Tensor, count: usize| projected.reshape(&[tokens, count, dim]);
    let q = split(ops::linear(x, &weights.q_proj, None)?, heads)?;
    let k = split(ops::linear(x, &weights.k_proj, None)?, kv_heads)?;
    let v = split(ops::linear(x, &weights.v_proj, None)?, kv_heads)?;

    // 2. 질의와 키를 위치에 따라 돌린다. 값은 돌리지 않는다.
    let q = rope::apply(&q, frequencies, 0)?;
    let k = rope::apply(&k, frequencies, 0)?;
    let (q, k, v) = (
        q.as_slice::<f32>()?,
        k.as_slice::<f32>()?,
        v.as_slice::<f32>()?,
    );
    // (토큰, 헤드)의 벡터가 놓인 구간. `count`는 토큰 하나에 딸린 헤드의 수다.
    let vector = |count: usize, token: usize, head: usize| {
        let start = (token * count + head) * dim;
        start..start + dim
    };

    let scale = 1.0 / (dim as f32).sqrt();
    let mut output = vec![0.0; tokens * heads * dim];
    let mut all_weights = vec![0.0; heads * tokens * tokens];
    for head in 0..heads {
        // 질의 헤드 여럿이 키·값 헤드 하나를 함께 쓴다.
        let kv_head = head / config.heads_per_kv();
        for token in 0..tokens {
            // 3. 점수: 자기와 앞선 토큰들의 키와 내적한다. 뒤의 토큰은 보지 않는다.
            let query = &q[vector(heads, token, head)];
            let scores: Vec<f32> = (0..=token)
                .map(|earlier| dot(query, &k[vector(kv_heads, earlier, kv_head)]) * scale)
                .collect();

            // 4. 비중: 점수를 확률분포로 바꾼다.
            let attention = softmax(&scores);

            // 5. 값들을 비중대로 섞는다.
            let mixed = &mut output[vector(heads, token, head)];
            for (earlier, &weight) in attention.iter().enumerate() {
                let value = &v[vector(kv_heads, earlier, kv_head)];
                for (sum, value) in mixed.iter_mut().zip(value) {
                    *sum += weight * value;
                }
            }
            let row = (head * tokens + token) * tokens;
            all_weights[row..row + attention.len()].copy_from_slice(&attention);
        }
    }

    // 6. 헤드들의 결과를 이어 붙여 선형 층 하나를 더 지난다.
    let mixed = Tensor::from_vec(output, &[tokens, heads * dim])?;
    Ok(AttentionOutput {
        output: ops::linear(&mixed, &weights.o_proj, None)?,
        weights: Tensor::from_vec(all_weights, &[heads, tokens, tokens])?,
    })
}
// ANCHOR_END: attention
