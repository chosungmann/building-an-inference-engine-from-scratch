//! 모델 전체의 순전파 (10장).

use std::path::Path;

use badak_core::{Tensor, arena, ops};
use badak_formats::Result;

use super::cache::{KvCache, block_cached};
use super::paged::{self, PagedCache, Step};
use badak_core::packed::Matrix;

use super::{BlockWeights, LlamaConfig, LlamaFiles, WeightFormat, block, rope};
use crate::tokenizer::TokenId;

// ANCHOR: model
/// 메모리에 올린 언어 모델.
pub struct LlamaModel {
    /// 모델의 설정.
    pub config: LlamaConfig,
    /// 토큰 임베딩 표. 출력 층도 이 표를 쓴다.
    embedding: Matrix,
    blocks: Vec<BlockWeights>,
    final_norm: Tensor,
    /// 위치를 회전으로 나타낼 때 쓰는 회전 속도.
    frequencies: Vec<f32>,
}

impl LlamaModel {
    /// 모델 디렉터리에서 가중치를 모두 읽어 들인다.
    ///
    /// 20장: 행렬 가중치를 메모리에 두는 방식은 환경 변수 `BADAK_WEIGHTS`를
    /// 따르고, 없으면 파일에 저장된 그대로 둔다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        Self::load_as(directory, WeightFormat::from_environment())
    }

    /// 행렬 가중치를 메모리에 두는 방식을 정해서 읽어 들인다.
    pub fn load_as(directory: impl AsRef<Path>, format: WeightFormat) -> Result<Self> {
        let files = LlamaFiles::open(directory)?.with_format(format);
        let blocks = (0..files.config.num_layers)
            .map(|index| files.block(index))
            .collect::<Result<_>>()?;
        Ok(Self {
            frequencies: rope::inverse_frequencies(&files.config),
            embedding: files.embedding_matrix()?,
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

        let mut states = vec![self.embedding.lookup(&ids)?];
        for weights in &self.blocks {
            let x = states.last().expect("임베딩이 들어 있다");
            // 22장: 블록이 만드는 텐서들은 아레나에서 자리를 얻는다. 모아 둘
            // 결과만 힙으로 옮겨 적는다.
            states.push(arena::scope(|| -> Result<Tensor> {
                Ok(block(x, weights, &self.config, &self.frequencies)?.detached()?)
            })?);
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
        // 14장: 프로파일러가 켜져 있으면 이 함수에 쓴 시간을 적는다.
        let _scope = badak_core::profile::scope("output_layer");
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
        let x = self.hidden_cached(ids, cache)?;
        let count = x.shape()[0];
        let row = x.slice(0, count.saturating_sub(1)..count)?;
        Ok(self.logits(&row)?.squeeze(0)?)
    }
    // ANCHOR_END: forward_cached

    // ANCHOR: hidden_cached
    /// 새 토큰들을 블록에 모두 통과시킨 결과. `(새 토큰 수, hidden_size)`.
    ///
    /// [`forward_cached`](Self::forward_cached)는 이 가운데 마지막 줄만
    /// 점수로 바꾼다. 여러 자리의 점수가 필요하면(21장의 퍼플렉서티) 이
    /// 함수의 결과에서 필요한 줄을 골라 [`logits`](Self::logits)에 넣는다.
    pub fn hidden_cached(&self, ids: &[TokenId], cache: &mut KvCache) -> Result<Tensor> {
        let past = cache.len();
        let count = ids.len();
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let ids = Tensor::from_vec(ids, &[count])?;

        let mut x = self.embedding.lookup(&ids)?;
        for (weights, layer) in self.blocks.iter().zip(cache.layers_mut()) {
            // 22장: 블록 하나가 만드는 텐서들은 아레나에서 자리를 얻는다. 블록의
            // 결과만 `x`가 쥔 힙의 자리에 덮어쓰고 나오면, 다음 블록이 아레나를
            // 처음부터 다시 쓴다.
            arena::scope(|| -> Result<()> {
                let output =
                    block_cached(&x, weights, &self.config, &self.frequencies, layer, past)?;
                Ok(x.assign(&output)?)
            })?;
        }
        cache.advance(count);
        Ok(x)
    }
    // ANCHOR_END: hidden_cached

    // ANCHOR: forward_batch
    /// 여러 글을 한 번의 순전파로 한 스텝씩 나아가게 한다 (40장).
    ///
    /// 글마다 새 토큰들과 제 캐시를 낸다. 글마다 **마지막 새 토큰 다음에 올
    /// 토큰의 점수**를 돌려준다. 새 토큰의 수는 글마다 달라도 된다. 글을
    /// 읽고 있는 글은 여러 개를, 답을 쓰고 있는 글은 하나를 낸다.
    pub fn forward_batch(&self, steps: &mut [Step<'_>]) -> Result<Vec<Tensor>> {
        let spans = paged::prepare(steps)?;
        let ids: Vec<i32> = steps
            .iter()
            .flat_map(|step| step.ids.iter().map(|&id| id as i32))
            .collect();
        let rows = ids.len();
        let ids = Tensor::from_vec(ids, &[rows])?;

        // 임베딩부터 블록까지, 모든 글의 새 토큰이 한 텐서의 줄들이다.
        let mut x = self.embedding.lookup(&ids)?;
        {
            let mut caches: Vec<&mut PagedCache> =
                steps.iter_mut().map(|step| &mut *step.cache).collect();
            for (layer, weights) in self.blocks.iter().enumerate() {
                arena::scope(|| -> Result<()> {
                    let output = paged::block_batch(
                        &x,
                        weights,
                        &self.config,
                        &self.frequencies,
                        layer,
                        &mut caches,
                        &spans,
                    )?;
                    Ok(x.assign(&output)?)
                })?;
            }
        }
        paged::advance(steps);

        // 글마다 마지막 줄만 골라, 출력 층도 한 번에 지나간다. 단어 사전
        // 크기의 행렬을 글의 수만큼이 아니라 한 번만 읽는다.
        let hidden = self.config.hidden_size;
        let values = x.as_slice::<f32>()?;
        let last: Vec<f32> = spans
            .iter()
            .flat_map(|span| {
                let row = span.start + span.count - 1;
                values[row * hidden..(row + 1) * hidden].iter().copied()
            })
            .collect();
        let logits = self.logits(&Tensor::from_vec(last, &[spans.len(), hidden])?)?;
        (0..spans.len())
            .map(|index| Ok(logits.slice(0, index..index + 1)?.squeeze(0)?))
            .collect()
    }
    // ANCHOR_END: forward_batch

    /// 이미 읽어 둔 가중치들로 모델을 만든다 (36장). 가중치가 다른 이름으로
    /// 다른 모델의 파일 안에 들어 있을 때 쓴다.
    pub fn from_parts(
        config: LlamaConfig,
        embedding: Matrix,
        blocks: Vec<BlockWeights>,
        final_norm: Tensor,
    ) -> Self {
        Self {
            frequencies: rope::inverse_frequencies(&config),
            embedding,
            blocks,
            final_norm,
            config,
        }
    }

    /// 토큰들을 벡터로 바꾸기만 한다 (36장). `(토큰 수, hidden_size)`.
    pub fn embed(&self, ids: &[TokenId]) -> Result<Tensor> {
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let count = ids.len();
        Ok(self.embedding.lookup(&Tensor::from_vec(ids, &[count])?)?)
    }

    /// [`hidden_cached`](Self::hidden_cached)와 같되, 토큰 번호가 아니라
    /// 이미 벡터로 바꾼 것을 받는다 (36장). 토큰이 아닌 것(사진의 벡터)을
    /// 글 사이에 끼워 넣을 때 쓴다.
    pub fn hidden_cached_from(&self, x: Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let past = cache.len();
        let count = x.shape()[0];
        // 블록의 결과를 제자리에 덮어쓰므로 제 것인 텐서로 옮겨 둔다.
        let mut x = x.detached()?;
        for (weights, layer) in self.blocks.iter().zip(cache.layers_mut()) {
            arena::scope(|| -> Result<()> {
                let output =
                    block_cached(&x, weights, &self.config, &self.frequencies, layer, past)?;
                Ok(x.assign(&output)?)
            })?;
        }
        cache.advance(count);
        Ok(x)
    }

    // ANCHOR: matrices
    /// 행렬 모양의 가중치를 모두 돌려준다. `(종류, 블록의 번호, 텐서)`다.
    ///
    /// 임베딩 표와 블록마다의 선형 층 일곱 개다. 정규화의 가중치는 벡터라서
    /// 들어 있지 않다. 파라미터의 거의 전부가 여기에 있다.
    pub fn matrices(&self) -> Vec<(&'static str, Option<usize>, &Matrix)> {
        let mut all = vec![("embedding", None, &self.embedding)];
        for (index, b) in self.blocks.iter().enumerate() {
            let layer = Some(index);
            all.extend([
                ("q_proj", layer, &b.q_proj),
                ("k_proj", layer, &b.k_proj),
                ("v_proj", layer, &b.v_proj),
                ("o_proj", layer, &b.o_proj),
                ("gate_proj", layer, &b.gate_proj),
                ("up_proj", layer, &b.up_proj),
                ("down_proj", layer, &b.down_proj),
            ]);
        }
        all
    }

    // ANCHOR: retain_blocks
    /// `keep`이 참이라고 답한 번호의 블록만 남긴다 (46장).
    ///
    /// 블록을 통째로 덜어 냈을 때 모델이 어떻게 달라지는지 볼 때 쓴다. 블록은
    /// 입력에 무언가를 더해 내보내므로(9.2절의 잔차 연결), 하나가 빠져도
    /// 나머지의 형상은 그대로 맞는다.
    pub fn retain_blocks(&mut self, keep: impl Fn(usize) -> bool) {
        let mut index = 0;
        self.blocks.retain(|_| {
            index += 1;
            keep(index - 1)
        });
        self.config.num_layers = self.blocks.len();
    }
    // ANCHOR_END: retain_blocks

    /// 행렬 모양의 가중치를 하나씩 `change`가 돌려준 텐서로 바꿔 끼운다.
    ///
    /// 양자화한 가중치로 모델이 어떻게 달라지는지 볼 때 쓴다(19장).
    pub fn replace_matrices(
        &mut self,
        mut change: impl FnMut(&'static str, Option<usize>, &Matrix) -> Result<Matrix>,
    ) -> Result<()> {
        self.embedding = change("embedding", None, &self.embedding)?;
        for (index, b) in self.blocks.iter_mut().enumerate() {
            let layer = Some(index);
            b.q_proj = change("q_proj", layer, &b.q_proj)?;
            b.k_proj = change("k_proj", layer, &b.k_proj)?;
            b.v_proj = change("v_proj", layer, &b.v_proj)?;
            b.o_proj = change("o_proj", layer, &b.o_proj)?;
            b.gate_proj = change("gate_proj", layer, &b.gate_proj)?;
            b.up_proj = change("up_proj", layer, &b.up_proj)?;
            b.down_proj = change("down_proj", layer, &b.down_proj)?;
        }
        Ok(())
    }
    // ANCHOR_END: matrices

    // ANCHOR: override_mlps
    /// 블록마다 MLP 전체를 대신 계산할 것을 끼운다 (23장).
    ///
    /// `make`는 블록의 번호와 MLP의 세 행렬(문, 위로, 아래로)을 받는다.
    pub fn override_mlps(
        &mut self,
        mut make: impl FnMut(
            usize,
            &Matrix,
            &Matrix,
            &Matrix,
        ) -> Result<std::sync::Arc<dyn ops::Operator>>,
    ) -> Result<()> {
        for (index, b) in self.blocks.iter_mut().enumerate() {
            b.mlp_override = Some(make(index, &b.gate_proj, &b.up_proj, &b.down_proj)?);
        }
        Ok(())
    }
    // ANCHOR_END: override_mlps

    /// 파라미터의 수. 임베딩 표는 출력 층과 함께 쓰므로 한 번만 센다.
    pub fn parameter_count(&self) -> usize {
        let block_count = |b: &BlockWeights| {
            let norms = b.attention_norm.len() + b.mlp_norm.len();
            let matrices = [
                &b.q_proj,
                &b.k_proj,
                &b.v_proj,
                &b.o_proj,
                &b.gate_proj,
                &b.up_proj,
                &b.down_proj,
            ];
            norms + matrices.iter().map(|matrix| matrix.len()).sum::<usize>()
        };
        self.embedding.len()
            + self.final_norm.len()
            + self.blocks.iter().map(block_count).sum::<usize>()
    }
}
