//! CLIP의 텍스트 인코더 (35장).
//!
//! 글을 토큰마다의 벡터 77개로 바꾼다. U-Net이 그 벡터들을 교차 어텐션의
//! 키와 값으로 본다. 구조는 32장의 ViT와 같은 블록이고(정규화가 앞),
//! 9장처럼 뒤의 토큰을 보지 못하게 가린다.

use std::path::Path;

use badak_core::prob;
use badak_formats::Result;
use badak_formats::safetensors::SafeTensors;

use super::layers::{Kernel, invalid, layer_norm, pair};
use crate::tokenizer::TokenId;

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

/// CLIP의 텍스트 인코더.
pub struct TextEncoder {
    /// 토큰마다의 벡터. `(단어 사전의 크기 × 폭)`.
    tokens: Vec<f32>,
    /// 자리마다 더하는 벡터. `(77 × 폭)`.
    positions: Vec<f32>,
    layers: Vec<Layer>,
    norm: (Vec<f32>, Vec<f32>),
    width: usize,
    heads: usize,
}

/// 층 정규화의 `eps`.
const EPS: f32 = 1e-5;

impl TextEncoder {
    /// safetensors 파일에서 읽는다. `heads`는 어텐션 헤드의 수다.
    pub fn load(path: impl AsRef<Path>, heads: usize) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let tokens = file.tensor("text_model.embeddings.token_embedding.weight")?;
        let &[_, width] = tokens.shape() else {
            return Err(invalid("토큰 벡터의 표가 행렬이 아닙니다".into()));
        };
        let mut layers = Vec::new();
        while file.names().any(|name| {
            name == format!(
                "text_model.encoder.layers.{}.layer_norm1.weight",
                layers.len()
            )
        }) {
            let name = format!("text_model.encoder.layers.{}", layers.len());
            layers.push(Layer {
                norm1: pair(&file, &format!("{name}.layer_norm1"))?,
                query: Kernel::load(&file, &format!("{name}.self_attn.q_proj"))?,
                key: Kernel::load(&file, &format!("{name}.self_attn.k_proj"))?,
                value: Kernel::load(&file, &format!("{name}.self_attn.v_proj"))?,
                out: Kernel::load(&file, &format!("{name}.self_attn.out_proj"))?,
                norm2: pair(&file, &format!("{name}.layer_norm2"))?,
                fc1: Kernel::load(&file, &format!("{name}.mlp.fc1"))?,
                fc2: Kernel::load(&file, &format!("{name}.mlp.fc2"))?,
            });
        }
        Ok(Self {
            tokens: tokens.to_vec::<f32>()?,
            positions: file
                .tensor("text_model.embeddings.position_embedding.weight")?
                .to_vec::<f32>()?,
            layers,
            norm: pair(&file, "text_model.final_layer_norm")?,
            width,
            heads,
        })
    }

    /// 토큰 하나의 폭.
    pub fn width(&self) -> usize {
        self.width
    }

    /// 받는 토큰의 수.
    pub fn length(&self) -> usize {
        self.positions.len() / self.width
    }

    // ANCHOR: encode
    /// 토큰 번호 77개를 벡터 77개로 바꾼다. 결과는 `(77, 폭)`이다.
    pub fn encode(&self, ids: &[TokenId]) -> Result<Vec<f32>> {
        let (width, count) = (self.width, ids.len());
        if count != self.length() {
            return Err(invalid(format!(
                "토큰이 {count}개입니다. {}개여야 합니다",
                self.length()
            )));
        }
        let mut x = Vec::with_capacity(count * width);
        for (position, &id) in ids.iter().enumerate() {
            let token = self
                .tokens
                .get(id as usize * width..(id as usize + 1) * width)
                .ok_or_else(|| invalid(format!("토큰 번호 {id}가 단어 사전 밖입니다")))?;
            let place = &self.positions[position * width..][..width];
            x.extend(token.iter().zip(place).map(|(t, p)| t + p));
        }
        for layer in &self.layers {
            let mut normed = x.clone();
            layer_norm(&mut normed, &layer.norm1.0, &layer.norm1.1, EPS);
            let attended = self.attention(layer, &normed)?;
            x.iter_mut().zip(&attended).for_each(|(a, b)| *a += b);

            let mut normed = x.clone();
            layer_norm(&mut normed, &layer.norm2.0, &layer.norm2.1, EPS);
            let mut hidden = layer.fc1.linear(&normed)?;
            // CLIP의 활성화 함수는 "빠른 GELU"다. x × sigmoid(1.702 x).
            hidden
                .iter_mut()
                .for_each(|value| *value /= 1.0 + (-1.702 * *value).exp());
            let hidden = layer.fc2.linear(&hidden)?;
            x.iter_mut().zip(&hidden).for_each(|(a, b)| *a += b);
        }
        layer_norm(&mut x, &self.norm.0, &self.norm.1, EPS);
        Ok(x)
    }
    // ANCHOR_END: encode

    /// 셀프 어텐션. 토큰은 자기와 그 앞만 본다(9장의 마스크).
    fn attention(&self, layer: &Layer, x: &[f32]) -> Result<Vec<f32>> {
        let (width, head) = (self.width, self.width / self.heads);
        let count = x.len() / width;
        let (queries, keys, values) = (
            layer.query.linear(x)?,
            layer.key.linear(x)?,
            layer.value.linear(x)?,
        );
        let scale = 1.0 / (head as f32).sqrt();
        let mut mixed = vec![0.0; x.len()];
        let mut scores = vec![0.0; count];
        for index in 0..self.heads {
            let part = |from: &[f32], row: usize| -> std::ops::Range<usize> {
                let start = row * width + index * head;
                debug_assert!(start + head <= from.len());
                start..start + head
            };
            for row in 0..count {
                let query = &queries[part(&queries, row)];
                // 뒤의 토큰은 보지 않는다. 앞의 row + 1개만 견준다.
                let seen = &mut scores[..=row];
                for (other, score) in seen.iter_mut().enumerate() {
                    let key = &keys[part(&keys, other)];
                    *score = scale * query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>();
                }
                prob::softmax_in_place(seen);
                let out = &mut mixed[row * width + index * head..][..head];
                for (other, &score) in seen.iter().enumerate() {
                    let value = &values[part(&values, other)];
                    out.iter_mut().zip(value).for_each(|(o, v)| *o += score * v);
                }
            }
        }
        layer.out.linear(&mixed)
    }
}
