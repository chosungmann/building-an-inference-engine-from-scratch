//! 사진을 보고 글로 답하는 모델 (36장).
//!
//! SmolVLM-256M이다. 부품은 셋이고 둘은 이미 만든 것이다.
//!
//! 1. 비전 인코더: 32장의 ViT. 사진을 조각마다의 벡터 1,024개로 바꾼다.
//! 2. 커넥터: 그 벡터들을 언어 모델이 읽는 벡터 64개로 줄인다. 이 장에서
//!    새로 만드는 것은 이것뿐이다.
//! 3. 언어 모델: 2부의 Llama. 사진의 벡터 64개를 **토큰의 벡터처럼**
//!    글의 벡터들 사이에 끼워 받는다.

use std::path::Path;

use badak_core::packed::Matrix;
use badak_core::{Tensor, ops};
use badak_formats::json::{self, Value};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

use crate::diffusion::layers::{Kernel, attend, layer_norm, pair};
use crate::llama::cache::KvCache;
use crate::llama::model::LlamaModel;
use crate::llama::{BlockWeights, LlamaConfig};
use crate::tokenizer::TokenId;

fn invalid(message: String) -> Error {
    Error::Parse {
        format: "SmolVLM",
        offset: 0,
        message,
    }
}

/// 비전 인코더의 블록 하나.
struct Layer {
    norm1: (Vec<f32>, Vec<f32>),
    query: Kernel,
    key: Kernel,
    value: Kernel,
    out: Kernel,
    norm2: (Vec<f32>, Vec<f32>),
    fc1: Kernel,
    fc2: Kernel,
}

// ANCHOR: vision
/// 비전 인코더. SigLIP의 ViT다.
pub struct Vision {
    /// 조각 하나를 벡터로 바꾸는 선형 층. `(폭, 3 × 조각 × 조각)`.
    patch: Kernel,
    /// 자리마다 더하는 벡터. `(조각 수 × 폭)`.
    positions: Vec<f32>,
    layers: Vec<Layer>,
    norm: (Vec<f32>, Vec<f32>),
    /// 조각의 한 변(픽셀).
    patch_size: usize,
    /// 사진의 한 변(픽셀).
    image_size: usize,
    heads: usize,
    fast: bool,
}
// ANCHOR_END: vision

/// 층 정규화의 `eps`.
const EPS: f32 = 1e-6;

impl Vision {
    fn load(file: &SafeTensors, heads: usize) -> Result<Self> {
        let root = "model.vision_model";
        let patch = Kernel::load_flat(file, &format!("{root}.embeddings.patch_embedding"))?;
        let positions = file.tensor(&format!("{root}.embeddings.position_embedding.weight"))?;
        let &[count, _] = positions.shape() else {
            return Err(invalid("자리 벡터의 표가 행렬이 아닙니다".into()));
        };
        let side = count.isqrt();
        let patch_size = (patch.inputs / 3).isqrt();
        if side * side != count || patch_size * patch_size * 3 != patch.inputs {
            return Err(invalid("조각이나 자리가 정사각형이 아닙니다".into()));
        }
        let mut layers = Vec::new();
        while file.names().any(|name| {
            name == format!("{root}.encoder.layers.{}.layer_norm1.weight", layers.len())
        }) {
            let name = format!("{root}.encoder.layers.{}", layers.len());
            layers.push(Layer {
                norm1: pair(file, &format!("{name}.layer_norm1"))?,
                query: Kernel::load(file, &format!("{name}.self_attn.q_proj"))?,
                key: Kernel::load(file, &format!("{name}.self_attn.k_proj"))?,
                value: Kernel::load(file, &format!("{name}.self_attn.v_proj"))?,
                out: Kernel::load(file, &format!("{name}.self_attn.out_proj"))?,
                norm2: pair(file, &format!("{name}.layer_norm2"))?,
                fc1: Kernel::load(file, &format!("{name}.mlp.fc1"))?,
                fc2: Kernel::load(file, &format!("{name}.mlp.fc2"))?,
            });
        }
        Ok(Self {
            patch,
            positions: positions.to_vec::<f32>()?,
            layers,
            norm: pair(file, &format!("{root}.post_layernorm"))?,
            patch_size,
            image_size: side * patch_size,
            heads,
            fast: false,
        })
    }

    fn pack(&mut self) -> Result<()> {
        self.fast = true;
        self.patch.pack()?;
        for layer in &mut self.layers {
            for kernel in [
                &mut layer.query,
                &mut layer.key,
                &mut layer.value,
                &mut layer.out,
                &mut layer.fc1,
                &mut layer.fc2,
            ] {
                kernel.pack()?;
            }
        }
        Ok(())
    }

    /// 사진의 한 변(픽셀).
    pub fn image_size(&self) -> usize {
        self.image_size
    }

    /// 조각의 수.
    pub fn patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// 조각 하나의 벡터의 폭.
    pub fn width(&self) -> usize {
        self.norm.0.len()
    }

    // ANCHOR: encode
    /// 사진을 조각마다의 벡터로 바꾼다. `image`는 정규화한 `(3, 한 변,
    /// 한 변)`이고 결과는 `(조각 수, 폭)`이다.
    ///
    /// 32장의 ViT와 같다. 분류 토큰이 없고(조각들의 벡터를 모두 쓴다),
    /// GELU가 `tanh`로 근사한 버전이다.
    pub fn encode(&self, image: &[f32]) -> Result<Vec<f32>> {
        let (side, patch, width) = (self.image_size, self.patch_size, self.width());
        if image.len() != 3 * side * side {
            return Err(invalid(format!(
                "사진의 값이 {}개입니다. (3, {side}, {side})이어야 합니다",
                image.len()
            )));
        }
        // 조각마다 (채널, 세로, 가로)의 순서로 편다(32.2절).
        let grid = side / patch;
        let mut flat = Vec::with_capacity(image.len());
        for row in 0..grid {
            for column in 0..grid {
                for channel in 0..3 {
                    for y in 0..patch {
                        let start = (channel * side + row * patch + y) * side + column * patch;
                        flat.extend_from_slice(&image[start..start + patch]);
                    }
                }
            }
        }
        let mut x = self.patch.linear(&flat)?;
        x.iter_mut().zip(&self.positions).for_each(|(a, b)| *a += b);

        for layer in &self.layers {
            let mut normed = x.clone();
            layer_norm(&mut normed, &layer.norm1.0, &layer.norm1.1, EPS);
            let mixed = attend(
                &layer.query.linear(&normed)?,
                &layer.key.linear(&normed)?,
                &layer.value.linear(&normed)?,
                width,
                self.heads,
                self.fast,
            );
            let mixed = layer.out.linear(&mixed)?;
            x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);

            let mut normed = x.clone();
            layer_norm(&mut normed, &layer.norm2.0, &layer.norm2.1, EPS);
            let mut hidden = layer.fc1.linear(&normed)?;
            gelu_tanh_in_place(&mut hidden, self.fast);
            let hidden = layer.fc2.linear(&hidden)?;
            x.iter_mut().zip(&hidden).for_each(|(a, b)| *a += b);
        }
        layer_norm(&mut x, &self.norm.0, &self.norm.1, EPS);
        Ok(x)
    }
    // ANCHOR_END: encode
}

/// `tanh`로 근사한 GELU. 이 모델의 비전 인코더가 이 함수로 학습되었다.
fn gelu_tanh(x: f32) -> f32 {
    let inner = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ANCHOR: gelu_tanh
/// 값들에 `tanh` GELU를 제자리에서 적용한다. 값이 블록마다 300만 개여서
/// 스레드에 나눈다. 빠른 길에서는 `tanh`를 32장의 근사한 지수 함수로
/// 계산한다. `tanh(u) = 1 − 2 / (e^(2u) + 1)`이다.
fn gelu_tanh_in_place(values: &mut [f32], fast: bool) {
    let pool = badak_core::threads::global().pool();
    let piece = values.len().div_ceil(4 * pool.threads()).max(1);
    pool.run_chunks(values, piece, pool.threads(), &|_, chunk| {
        if !fast {
            chunk
                .iter_mut()
                .for_each(|value| *value = gelu_tanh(*value));
            return;
        }
        let root = (2.0 / std::f32::consts::PI).sqrt();
        for value in chunk.iter_mut() {
            let x = *value;
            // 지수 함수의 근사가 끝값에서 잘린다. 그 밖에서 tanh는 이미 ±1이다.
            let inner = (root * (x + 0.044_715 * x * x * x)).clamp(-20.0, 20.0);
            let tanh = 1.0 - 2.0 / (badak_core::nn::exp_fast(2.0 * inner) + 1.0);
            *value = 0.5 * x * (1.0 + tanh);
        }
    });
}
// ANCHOR_END: gelu_tanh

// ANCHOR: shuffle
/// 이웃한 조각 `scale × scale`개의 벡터를 이어 붙여 하나로 만든다.
///
/// `(한 변 × 한 변, 폭)`이 `((한 변 / scale)², 폭 × scale²)`이 된다.
/// 조각 16개의 벡터가 긴 벡터 하나가 되므로, 값은 하나도 버리지 않고
/// 토큰의 수만 16분의 1이 된다. 이어 붙이는 순서는 줄 순서다(위의 넷,
/// 그 아랫줄의 넷, …).
pub fn shuffle(tokens: &[f32], side: usize, width: usize, scale: usize) -> Vec<f32> {
    let small = side / scale;
    let mut out = Vec::with_capacity(tokens.len());
    for row in 0..small {
        for column in 0..small {
            for y in 0..scale {
                let from = ((row * scale + y) * side + column * scale) * width;
                out.extend_from_slice(&tokens[from..from + scale * width]);
            }
        }
    }
    out
}
// ANCHOR_END: shuffle

// ANCHOR: model
/// SmolVLM.
pub struct SmolVlm {
    /// 비전 인코더.
    pub vision: Vision,
    /// 커넥터의 선형 층. `(언어 모델의 폭, 비전의 폭 × 16)`.
    projection: Kernel,
    /// 커넥터가 한 변을 몇 분의 1로 줄이는가.
    scale: usize,
    /// 언어 모델.
    pub text: LlamaModel,
    /// 출력 층. 이 모델은 토큰 벡터의 표와 출력 층이 따로다.
    head: Matrix,
    /// 글 속에서 사진의 벡터가 들어갈 자리를 나타내는 토큰.
    pub image_token: TokenId,
}
// ANCHOR_END: model

impl SmolVlm {
    // ANCHOR: load
    /// 디렉터리에서 `config.json`과 `model.safetensors`를 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let root = json::parse(&std::fs::read_to_string(directory.join("config.json"))?)?;
        let number = |object: &Value, key: &str| -> Result<usize> {
            object
                .get(key)
                .and_then(Value::as_usize)
                .ok_or_else(|| invalid(format!("config.json에 {key}가 없습니다")))
        };
        let section = |key: &str| {
            root.get(key)
                .ok_or_else(|| invalid(format!("config.json에 {key}가 없습니다")))
        };
        let (text, vision) = (section("text_config")?, section("vision_config")?);
        let real = |key: &str| -> Result<f32> {
            Ok(text
                .get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid(format!("text_config에 {key}가 없습니다")))?
                as f32)
        };
        let config = LlamaConfig {
            vocab_size: number(text, "vocab_size")?,
            hidden_size: number(text, "hidden_size")?,
            num_layers: number(text, "num_hidden_layers")?,
            num_heads: number(text, "num_attention_heads")?,
            num_kv_heads: number(text, "num_key_value_heads")?,
            intermediate_size: number(text, "intermediate_size")?,
            rms_norm_eps: real("rms_norm_eps")?,
            rope_theta: real("rope_theta")?,
            rope_scaling: None,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        };

        let file = SafeTensors::open(directory.join("model.safetensors"))?;
        // 언어 모델의 가중치는 2부와 같은 이름에 `text_model.`이 끼어 있다.
        let matrix = |name: String| -> Result<Matrix> { Ok(file.tensor(&name)?.into()) };
        let blocks = (0..config.num_layers)
            .map(|index| {
                let name = |part: &str| format!("model.text_model.layers.{index}.{part}.weight");
                Ok(BlockWeights {
                    attention_norm: file.tensor(&name("input_layernorm"))?,
                    q_proj: matrix(name("self_attn.q_proj"))?,
                    k_proj: matrix(name("self_attn.k_proj"))?,
                    v_proj: matrix(name("self_attn.v_proj"))?,
                    o_proj: matrix(name("self_attn.o_proj"))?,
                    mlp_norm: file.tensor(&name("post_attention_layernorm"))?,
                    gate_proj: matrix(name("mlp.gate_proj"))?,
                    up_proj: matrix(name("mlp.up_proj"))?,
                    down_proj: matrix(name("mlp.down_proj"))?,
                    mlp_override: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let text = LlamaModel::from_parts(
            config,
            matrix("model.text_model.embed_tokens.weight".into())?,
            blocks,
            file.tensor("model.text_model.norm.weight")?,
        );
        Ok(Self {
            vision: Vision::load(&file, number(vision, "num_attention_heads")?)?,
            projection: Kernel::load(&file, "model.connector.modality_projection.proj")?,
            scale: number(&root, "scale_factor")?,
            text,
            head: matrix("lm_head.weight".into())?,
            image_token: number(&root, "image_token_id")? as TokenId,
        })
    }
    // ANCHOR_END: load

    /// 빠른 길을 켠다. 비전 인코더와 커넥터의 선형 층을 31장의 SME
    /// 커널로 계산한다. 언어 모델은 2부의 길 그대로다.
    pub fn fast(mut self) -> Result<Self> {
        self.vision.pack()?;
        self.projection.pack()?;
        Ok(self)
    }

    /// 사진 한 장이 차지하는 토큰의 수.
    pub fn image_tokens(&self) -> usize {
        self.vision.patches() / (self.scale * self.scale)
    }

    // ANCHOR: connect
    /// 비전 인코더가 낸 벡터들을 언어 모델이 읽는 벡터들로 바꾼다.
    /// `(조각 수, 비전의 폭)`이 `(조각 수 / 16, 언어 모델의 폭)`이 된다.
    pub fn connect(&self, patches: &[f32]) -> Result<Vec<f32>> {
        let width = self.vision.width();
        let side = (patches.len() / width).isqrt();
        self.projection
            .linear(&shuffle(patches, side, width, self.scale))
    }

    /// 사진 한 장을 언어 모델이 읽는 벡터들로 바꾼다.
    pub fn see(&self, image: &[f32]) -> Result<Vec<f32>> {
        self.connect(&self.vision.encode(image)?)
    }
    // ANCHOR_END: connect

    // ANCHOR: embed
    /// 토큰들을 벡터로 바꾸되, 사진 자리의 토큰에는 사진의 벡터를 차례로
    /// 넣는다. `seen`은 [`see`](Self::see)의 결과다.
    pub fn embed(&self, ids: &[TokenId], seen: &[f32]) -> Result<Tensor> {
        let width = self.text.config.hidden_size;
        let embedded = self.text.embed(ids)?;
        let mut values = embedded.to_vec::<f32>()?;
        let mut vectors = seen.chunks_exact(width);
        for (row, &id) in values.chunks_exact_mut(width).zip(ids) {
            if id == self.image_token {
                let vector = vectors
                    .next()
                    .ok_or_else(|| invalid("사진 자리가 사진의 벡터보다 많습니다".into()))?;
                row.copy_from_slice(vector);
            }
        }
        if vectors.next().is_some() {
            return Err(invalid("사진의 벡터가 사진 자리보다 많습니다".into()));
        }
        Ok(Tensor::from_vec(values, &[ids.len(), width])?)
    }
    // ANCHOR_END: embed

    /// 블록을 모두 지난 마지막 줄을 토큰마다의 점수로 바꾼다.
    fn logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let count = hidden.shape()[0];
        let row = hidden.slice(0, count.saturating_sub(1)..count)?;
        let normed = self.text.final_norm(&row)?;
        Ok(ops::linear(&normed, &self.head, None)?.to_vec::<f32>()?)
    }

    // ANCHOR: generate
    /// 사진과 글을 받아 답을 쓴다. 점수가 가장 높은 토큰을 고른다.
    ///
    /// `ids`는 사진 자리가 든 글의 토큰들이다. `stop`이 나오거나
    /// `limit`개를 쓰면 멈춘다. 돌려주는 것은 새로 쓴 토큰들과, 글을 다
    /// 읽은 직후의 점수다.
    pub fn generate(
        &self,
        ids: &[TokenId],
        seen: &[f32],
        stop: TokenId,
        limit: usize,
        mut each: impl FnMut(TokenId),
    ) -> Result<(Vec<TokenId>, Vec<f32>)> {
        let mut cache = KvCache::new(&self.text.config);
        // 1. 사진과 글을 한꺼번에 읽는다. 사진의 벡터도 KV 캐시에 들어간다.
        let hidden = self
            .text
            .hidden_cached_from(self.embed(ids, seen)?, &mut cache)?;
        let first = self.logits(&hidden)?;
        // 2. 한 토큰씩 쓴다. 여기부터는 2부와 같다.
        let mut written = Vec::new();
        let mut logits = first.clone();
        while written.len() < limit {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(stop, |(index, _)| index as TokenId);
            if next == stop {
                break;
            }
            written.push(next);
            each(next);
            let hidden = self
                .text
                .hidden_cached_from(self.text.embed(&[next])?, &mut cache)?;
            logits = self.logits(&hidden)?;
        }
        Ok((written, first))
    }
    // ANCHOR_END: generate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffling_joins_neighbours_in_row_order() {
        // 4 × 4의 조각, 폭 1. 값이 곧 조각의 번호다.
        let tokens: Vec<f32> = (0..16).map(|index| index as f32).collect();
        let out = shuffle(&tokens, 4, 1, 2);
        assert_eq!(
            out,
            [
                0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.
            ]
        );
        // 폭이 2이면 조각의 두 값이 함께 다닌다.
        let tokens: Vec<f32> = (0..8).map(|index| index as f32).collect();
        assert_eq!(shuffle(&tokens, 2, 2, 2), tokens);
    }

    #[test]
    fn the_tanh_gelu_is_close_to_the_exact_one() {
        for x in [-3.0_f32, -1.0, 0.0, 0.5, 2.0] {
            assert!((gelu_tanh(x) - badak_core::nn::gelu(x)).abs() < 2e-3, "{x}");
        }
        assert_eq!(gelu_tanh(0.0), 0.0);
        // 빠른 길의 근사는 정의대로 계산한 것과 거의 같다.
        let values: Vec<f32> = (-400..400).map(|index| index as f32 / 40.0).collect();
        let (mut exact, mut quick) = (values.clone(), values);
        gelu_tanh_in_place(&mut exact, false);
        gelu_tanh_in_place(&mut quick, true);
        for (a, b) in exact.iter().zip(&quick) {
            assert!((a - b).abs() < 2e-5, "{a} {b}");
        }
    }
}
