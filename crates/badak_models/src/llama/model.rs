//! 모델 전체의 순전파 (10장).

use std::path::Path;

use badak_core::{Tensor, ops};
use badak_formats::Result;

use super::cache::{KvCache, block_cached};
use super::{BlockWeights, LlamaConfig, LlamaFiles, block, rope};
use crate::tokenizer::TokenId;

// ANCHOR: model
/// 메모리에 올린 언어 모델.
pub struct LlamaModel {
    /// 모델의 설정.
    pub config: LlamaConfig,
    /// 토큰 임베딩 표. 출력 층도 이 표를 쓴다.
    embedding: Tensor,
    blocks: Vec<BlockWeights>,
    final_norm: Tensor,
    /// 위치를 회전으로 나타낼 때 쓰는 회전 속도.
    frequencies: Vec<f32>,
}

impl LlamaModel {
    /// 모델 디렉터리에서 가중치를 모두 읽어 들인다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let files = LlamaFiles::open(directory)?;
        let blocks = (0..files.config.num_layers)
            .map(|index| files.block(index))
            .collect::<Result<_>>()?;
        Ok(Self {
            frequencies: rope::inverse_frequencies(&files.config),
            embedding: files.embedding()?,
            final_norm: files.final_norm()?,
            blocks,
            config: files.config,
        })
    }
    // ANCHOR_END: model

    // ANCHOR: hidden_states
    /// 토큰들을 블록에 차례로 통과시키며 단계마다의 결과를 모은다.
    ///
    /// 첫 원소는 임베딩이고, 그 뒤로 블록 하나를 지날 때마다의 결과가
    /// 이어진다. 모두 `(토큰 수, hidden_size)`다.
    pub fn hidden_states(&self, ids: &[TokenId]) -> Result<Vec<Tensor>> {
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let count = ids.len();
        let ids = Tensor::from_vec(ids, &[count])?;

        let mut states = vec![ops::embedding(&self.embedding, &ids)?];
        for weights in &self.blocks {
            let x = states.last().expect("임베딩이 들어 있다");
            states.push(block(x, weights, &self.config, &self.frequencies)?);
        }
        Ok(states)
    }
    // ANCHOR_END: hidden_states

    // ANCHOR: logits
    /// 블록을 모두 지난 벡터에 마지막 정규화를 한다.
    pub fn final_norm(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(ops::rms_norm(
            hidden,
            &self.final_norm,
            self.config.rms_norm_eps,
        )?)
    }

    /// 블록을 모두 지난 벡터를 토큰마다의 점수로 바꾼다.
    ///
    /// 마지막 정규화를 하고, 임베딩 표의 모든 행과 내적한다. `hidden`이
    /// `(줄 수, hidden_size)`이면 결과는 `(줄 수, 단어 사전 크기)`다.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.final_norm(hidden)?;
        Ok(ops::linear(&normed, &self.embedding, None)?)
    }
    // ANCHOR_END: logits

    // ANCHOR: forward
    /// 순전파 전체. 토큰마다 "그다음 토큰"의 점수를 낸다.
    /// `(토큰 수, 단어 사전 크기)`.
    pub fn forward(&self, ids: &[TokenId]) -> Result<Tensor> {
        let states = self.hidden_states(ids)?;
        self.logits(states.last().expect("임베딩이 들어 있다"))
    }

    /// 마지막 토큰 다음에 올 토큰의 점수만 낸다. `(단어 사전 크기)`.
    ///
    /// 글을 이어 쓸 때는 마지막 줄만 필요하다. 출력 층을 한 줄에만
    /// 적용하므로 [`forward`](Self::forward)보다 빠르다.
    pub fn next_token_logits(&self, ids: &[TokenId]) -> Result<Tensor> {
        let states = self.hidden_states(ids)?;
        let last = states.last().expect("임베딩이 들어 있다");
        let tokens = last.shape()[0];
        let row = last.slice(0, tokens.saturating_sub(1)..tokens)?;
        Ok(self.logits(&row)?.squeeze(0)?)
    }
    // ANCHOR_END: forward

    // ANCHOR: forward_cached
    /// 새 토큰들만 넣어, 그 마지막 토큰 다음에 올 토큰의 점수를 낸다.
    ///
    /// `ids`는 캐시에 아직 없는 토큰들이다. 앞선 토큰들의 키와 값은 캐시에서
    /// 읽고, 새 토큰의 키와 값은 캐시에 덧붙인다.
    pub fn forward_cached(&self, ids: &[TokenId], cache: &mut KvCache) -> Result<Tensor> {
        let past = cache.len();
        let count = ids.len();
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let ids = Tensor::from_vec(ids, &[count])?;

        let mut x = ops::embedding(&self.embedding, &ids)?;
        for (weights, layer) in self.blocks.iter().zip(cache.layers_mut()) {
            x = block_cached(&x, weights, &self.config, &self.frequencies, layer, past)?;
        }
        cache.advance(count);

        let row = x.slice(0, count.saturating_sub(1)..count)?;
        Ok(self.logits(&row)?.squeeze(0)?)
    }
    // ANCHOR_END: forward_cached

    /// 파라미터의 수. 임베딩 표는 출력 층과 함께 쓰므로 한 번만 센다.
    pub fn parameter_count(&self) -> usize {
        let block_count = |b: &BlockWeights| {
            [
                &b.attention_norm,
                &b.q_proj,
                &b.k_proj,
                &b.v_proj,
                &b.o_proj,
                &b.mlp_norm,
                &b.gate_proj,
                &b.up_proj,
                &b.down_proj,
            ]
            .iter()
            .map(|tensor| tensor.len())
            .sum::<usize>()
        };
        self.embedding.len()
            + self.final_norm.len()
            + self.blocks.iter().map(block_count).sum::<usize>()
    }
}
