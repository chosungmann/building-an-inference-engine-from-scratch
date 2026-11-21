//! BERT 계열의 문장 임베딩 모델 (34장).
//!
//! 인코더만 있는 트랜스포머다. 토큰들을 받아 토큰마다 벡터 하나를 내고,
//! 그 벡터들의 평균이 글 하나의 임베딩이다. 32장의 ViT와 블록의 재료가
//! 같고(어텐션, 층 정규화, GELU), 정규화의 자리가 다르다. BERT는 더한
//! 다음에 정규화한다.
//!
//! 이 장의 주제는 **여러 글을 한 번에** 계산하는 일이다. 글마다 길이가
//! 달라서, 한 배치로 만드는 방법이 둘 있다([`Batching`]).

use std::path::Path;
use std::sync::Mutex;

use badak_core::{Tensor, ops, prob, simd};
use badak_formats::json;
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

use crate::tokenizer::TokenId;

fn invalid(message: String) -> Error {
    Error::Parse {
        format: "BERT",
        offset: 0,
        message,
    }
}

/// 빠른 길에서, 글이 이 토큰 수 이상이면 어텐션 안의 곱도 SME 커널로
/// 계산한다.
#[cfg(target_arch = "aarch64")]
const SME_ATTENTION: usize = 32;

// ANCHOR: model
/// 블록 하나의 가중치.
struct Layer {
    query: (Tensor, Tensor),
    key: (Tensor, Tensor),
    value: (Tensor, Tensor),
    /// 어텐션의 결과를 한 번 더 섞는 선형 층.
    mixed: (Tensor, Tensor),
    norm1: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
}

/// BERT 계열의 인코더.
pub struct Bert {
    /// 토큰마다의 벡터. `(단어 사전의 크기, 폭)`.
    words: Tensor,
    /// 자리마다 더하는 벡터. `(가장 긴 길이, 폭)`.
    positions: Tensor,
    /// 문장의 종류마다 더하는 벡터 중 첫째. 글을 하나씩만 넣으므로 첫째만
    /// 쓴다. `(폭)`.
    kind: Vec<f32>,
    norm: (Tensor, Tensor),
    layers: Vec<Layer>,
    heads: usize,
    /// 층 정규화의 `eps`.
    eps: f32,
    /// 31장의 SME 커널과 근사한 지수 함수를 쓸지. [`Bert::fast`]가 켠다.
    fast: bool,
    #[cfg(target_arch = "aarch64")]
    packed: Vec<(usize, badak_core::simd::sme::Packed)>,
}
// ANCHOR_END: model

// ANCHOR: batching
/// 길이가 다른 글들을 한 배치로 만드는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Batching {
    /// 가장 긴 글에 맞춰 나머지의 뒤를 `[PAD]`로 채운다. 배치가
    /// `(글의 수, 가장 긴 길이, 폭)`의 반듯한 텐서가 된다. 채운 자리는
    /// 어텐션에서 가린다. 텐서를 주고받는 엔진들이 쓰는 방법이다.
    Padded,
    /// 채우지 않고 글들을 이어 붙인다. `(토큰의 합, 폭)`이 된다. 글마다
    /// 어디서 어디까지인지를 따로 적어 두고, 어텐션은 글마다 따로 한다.
    Ragged,
}

/// 배치 안에서 글 하나가 차지하는 줄.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// 첫 줄.
    pub start: usize,
    /// 차지한 줄의 수. 채웠으면 채운 자리까지 센다.
    pub rows: usize,
    /// 진짜 토큰의 수.
    pub real: usize,
}

/// 글들이 배치 안에서 차지할 줄을 정한다.
pub fn layout(lengths: &[usize], batching: Batching) -> Vec<Span> {
    let longest = lengths.iter().copied().max().unwrap_or(0);
    let mut start = 0;
    lengths
        .iter()
        .map(|&real| {
            let rows = match batching {
                Batching::Padded => longest,
                Batching::Ragged => real,
            };
            let span = Span { start, rows, real };
            start += rows;
            span
        })
        .collect()
}
// ANCHOR_END: batching

/// 순전파의 중간 결과. 정답과 단계마다 대조할 때 쓴다.
pub struct Trace {
    /// 글마다 차지한 줄.
    pub spans: Vec<Span>,
    /// 토큰, 자리, 종류의 벡터를 더하고 정규화한 것. `(줄 수, 폭)`.
    pub embedded: Tensor,
    /// 블록마다의 출력. `(줄 수, 폭)`.
    pub layers: Vec<Tensor>,
    /// 글마다의 임베딩. 길이가 1이다. `(글의 수, 폭)`.
    pub pooled: Tensor,
}

// ANCHOR: plan
/// 많은 글을 임베딩으로 바꿀 때의 방법([`Bert::embed_many`]).
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    /// 한 배치에 넣을 글의 수.
    pub batch: usize,
    /// 배치를 만드는 방법.
    pub batching: Batching,
    /// 묶기 전에 길이의 순서로 줄 세울지. 비슷한 길이끼리 묶이면 채울
    /// 자리가 줄어든다.
    pub sorted: bool,
    /// 배치들을 스레드에 나눌지. 아니면 배치는 하나씩 차례로 하고, 배치
    /// 안의 행렬 곱을 스레드에 나눈다.
    pub across: bool,
}
// ANCHOR_END: plan

impl Bert {
    // ANCHOR: load
    /// 디렉터리에서 `config.json`과 `model.safetensors`를 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let config = json::parse(&std::fs::read_to_string(directory.join("config.json"))?)?;
        let heads = config
            .get("num_attention_heads")
            .and_then(json::Value::as_usize)
            .ok_or_else(|| invalid("config.json에 헤드의 수가 없습니다".into()))?;
        let eps = config
            .get("layer_norm_eps")
            .and_then(json::Value::as_f64)
            .ok_or_else(|| invalid("config.json에 층 정규화의 eps가 없습니다".into()))?
            as f32;

        let file = SafeTensors::open(directory.join("model.safetensors"))?;
        let pair = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                file.tensor(&format!("{name}.weight"))?,
                file.tensor(&format!("{name}.bias"))?,
            ))
        };
        let words = file.tensor("embeddings.word_embeddings.weight")?;
        let &[_, width] = words.shape() else {
            return Err(invalid("토큰 벡터의 표가 행렬이 아닙니다".into()));
        };
        if heads == 0 || width % heads != 0 {
            return Err(invalid(format!(
                "폭 {width}이 헤드의 수 {heads}로 나누어지지 않습니다"
            )));
        }
        let kind = file
            .tensor("embeddings.token_type_embeddings.weight")?
            .slice(0, 0..1)?
            .to_vec::<f32>()?;
        let mut layers = Vec::new();
        while file.names().any(|name| {
            name == format!("encoder.layer.{}.attention.self.query.weight", layers.len())
        }) {
            let name = format!("encoder.layer.{}", layers.len());
            layers.push(Layer {
                query: pair(&format!("{name}.attention.self.query"))?,
                key: pair(&format!("{name}.attention.self.key"))?,
                value: pair(&format!("{name}.attention.self.value"))?,
                mixed: pair(&format!("{name}.attention.output.dense"))?,
                norm1: pair(&format!("{name}.attention.output.LayerNorm"))?,
                fc1: pair(&format!("{name}.intermediate.dense"))?,
                fc2: pair(&format!("{name}.output.dense"))?,
                norm2: pair(&format!("{name}.output.LayerNorm"))?,
            });
        }
        Ok(Self {
            words,
            positions: file.tensor("embeddings.position_embeddings.weight")?,
            kind,
            norm: pair("embeddings.LayerNorm")?,
            layers,
            heads,
            eps,
            fast: false,
            #[cfg(target_arch = "aarch64")]
            packed: Vec::new(),
        })
    }
    // ANCHOR_END: load

    /// 빠른 길을 켠다. 32장의 `Vit::fast`와 같다. 선형 층을 31장의 SME
    /// 커널로(쓸 수 있는 기계에서), GELU와 소프트맥스의 지수 함수를
    /// 근사로 계산한다.
    pub fn fast(mut self) -> Result<Self> {
        self.fast = true;
        #[cfg(target_arch = "aarch64")]
        if badak_core::simd::sme::available() {
            for layer in &self.layers {
                for weight in [
                    &layer.query.0,
                    &layer.key.0,
                    &layer.value.0,
                    &layer.mixed.0,
                    &layer.fc1.0,
                    &layer.fc2.0,
                ] {
                    let &[rows, cols] = weight.shape() else {
                        return Err(invalid("가중치가 행렬이 아닙니다".into()));
                    };
                    let values = weight.as_slice::<f32>()?;
                    let packed = badak_core::simd::sme::Packed::new(values, rows, cols)
                        .ok_or_else(|| invalid("가중치를 늘어놓지 못했습니다".into()))?;
                    self.packed.push((values.as_ptr() as usize, packed));
                }
            }
        }
        Ok(self)
    }

    /// 선형 층. 빠른 길이 켜져 있고 이 가중치를 늘어놓은 것이 있으면 SME
    /// 커널로, 아니면 5장의 `ops::linear`로 계산한다.
    fn linear(&self, x: &Tensor, layer: &(Tensor, Tensor)) -> Result<Tensor> {
        #[cfg(target_arch = "aarch64")]
        {
            let key = layer
                .0
                .as_slice::<f32>()
                .map_or(0, |values| values.as_ptr() as usize);
            if let Some((_, packed)) = self.packed.iter().find(|(address, _)| *address == key) {
                let (count, rows) = (x.len() / packed.cols(), packed.rows());
                let input = x.contiguous();
                let bias = layer.1.contiguous();
                let (input, bias) = (input.as_slice::<f32>()?, bias.as_slice::<f32>()?);
                return Ok(Tensor::build(&[count, rows], |y| {
                    badak_core::simd::sme::linear(input, count, packed, y);
                    for row in y.chunks_exact_mut(rows) {
                        row.iter_mut().zip(bias).for_each(|(value, b)| *value += b);
                    }
                    Ok(())
                })?);
            }
        }
        Ok(ops::linear(x, &layer.0, Some(&layer.1))?)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        if !self.fast {
            return Ok(ops::gelu(x)?);
        }
        let mut values = x.to_vec::<f32>()?;
        badak_core::nn::gelu_in_place(&mut values);
        Ok(Tensor::from_vec(values, x.shape())?)
    }

    /// 임베딩의 폭.
    pub fn width(&self) -> usize {
        self.kind.len()
    }

    /// 블록의 수.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 받을 수 있는 가장 긴 글(토큰).
    pub fn longest(&self) -> usize {
        self.positions.shape()[0]
    }

    // ANCHOR: embed_tokens
    /// 토큰 번호들을 벡터로 바꾼다. 토큰의 벡터, 자리의 벡터, 종류의
    /// 벡터를 더하고 정규화한다. 채운 자리에는 `[PAD]`(`padding`)를 넣는다.
    fn embed_tokens(
        &self,
        sentences: &[&[TokenId]],
        spans: &[Span],
        padding: TokenId,
    ) -> Result<Tensor> {
        let width = self.width();
        let rows = spans.last().map_or(0, |span| span.start + span.rows);
        let (words, positions) = (self.words.contiguous(), self.positions.contiguous());
        let (words, positions) = (words.as_slice::<f32>()?, positions.as_slice::<f32>()?);
        let vocabulary = words.len() / width;
        let mut values = Vec::with_capacity(rows * width);
        for (sentence, span) in sentences.iter().zip(spans) {
            if span.rows > self.longest() {
                return Err(invalid(format!(
                    "글이 토큰 {}개입니다. {}개까지 받습니다",
                    span.rows,
                    self.longest()
                )));
            }
            for position in 0..span.rows {
                let id = sentence.get(position).copied().unwrap_or(padding) as usize;
                if id >= vocabulary {
                    return Err(invalid(format!("토큰 번호 {id}가 단어 사전 밖입니다")));
                }
                let word = &words[id * width..][..width];
                let place = &positions[position * width..][..width];
                values.extend(
                    word.iter()
                        .zip(place)
                        .zip(&self.kind)
                        .map(|((w, p), k)| w + p + k),
                );
            }
        }
        let summed = Tensor::from_vec(values, &[rows, width])?;
        Ok(ops::layer_norm(
            &summed,
            &self.norm.0,
            &self.norm.1,
            self.eps,
        )?)
    }
    // ANCHOR_END: embed_tokens

    // ANCHOR: attention
    /// 셀프 어텐션. `x`는 `(줄 수, 폭)`이고, 글마다 제 줄들 안에서만 본다.
    fn attention(&self, layer: &Layer, x: &Tensor, spans: &[Span]) -> Result<Tensor> {
        let width = self.width();
        // 질의, 키, 값을 만드는 선형 층은 글의 경계를 모른다. 배치의 모든
        // 줄을 한 번에 계산한다.
        let queries = self.linear(x, &layer.query)?;
        let keys = self.linear(x, &layer.key)?;
        let values = self.linear(x, &layer.value)?;
        let (queries, keys, values) = (
            queries.as_slice::<f32>()?,
            keys.as_slice::<f32>()?,
            values.as_slice::<f32>()?,
        );

        // 글마다의 어텐션은 서로 무관하다. 스레드에 나눈다. 결과를 적을
        // 자리를 글마다 잘라 두고 잠금을 하나씩 씌운다(18장).
        let mut mixed = vec![0.0; x.len()];
        let mut rest = mixed.as_mut_slice();
        let mut slots = Vec::with_capacity(spans.len());
        for span in spans {
            let (slot, tail) = rest.split_at_mut(span.rows * width);
            slots.push(Mutex::new(slot));
            rest = tail;
        }
        let threads = badak_core::threads::global();
        threads
            .pool()
            .run(spans.len(), threads.for_vectors(spans.len()), &|index| {
                let span = spans[index];
                let rows = span.start * width..(span.start + span.rows) * width;
                let mut slot = slots[index].lock().expect("글마다 한 번만 잠근다");
                self.attend(
                    &queries[rows.clone()],
                    &keys[rows.clone()],
                    &values[rows],
                    span.real,
                    &mut slot,
                );
            });
        drop(slots);
        let mixed = Tensor::from_vec(mixed, x.shape())?;
        self.linear(&mixed, &layer.mixed)
    }

    /// 글 하나의 어텐션. 질의, 키, 값은 `(줄 수, 폭)`이고 앞의 `real`줄만
    /// 진짜 토큰이다. 나머지는 채운 자리여서 키로 쓰이지 않게 가린다.
    fn attend(&self, queries: &[f32], keys: &[f32], values: &[f32], real: usize, out: &mut [f32]) {
        let width = self.width();
        let (rows, head) = (queries.len() / width, width / self.heads);
        let scale = 1.0 / (head as f32).sqrt();
        let (mut q, mut k) = (vec![0.0; rows * head], vec![0.0; rows * head]);
        // 값은 (헤드의 폭, 줄 수)로 뒤집어 둔다(32.3절).
        let mut v = vec![0.0; head * rows];
        let (mut scores, mut mixed) = (vec![0.0; rows * rows], vec![0.0; rows * head]);
        for index in 0..self.heads {
            for row in 0..rows {
                let from = row * width + index * head;
                q[row * head..][..head].copy_from_slice(&queries[from..][..head]);
                k[row * head..][..head].copy_from_slice(&keys[from..][..head]);
                for (lane, &value) in values[from..][..head].iter().enumerate() {
                    v[lane * rows + row] = value;
                }
            }
            self.product(&q, rows, &k, rows, head, &mut scores);
            for row in scores.chunks_exact_mut(rows) {
                row.iter_mut().for_each(|score| *score *= scale);
                // 마스크: 채운 자리의 점수를 −∞로 바꾼다. 소프트맥스를 지나면
                // 그 자리의 몫이 0이 된다.
                row[real..].fill(f32::NEG_INFINITY);
                if self.fast {
                    badak_core::nn::softmax_fast_in_place(row);
                } else {
                    prob::softmax_in_place(row);
                }
            }
            self.product(&scores, rows, &v, head, rows, &mut mixed);
            for row in 0..rows {
                out[row * width + index * head..][..head]
                    .copy_from_slice(&mixed[row * head..][..head]);
            }
        }
    }

    /// `y = x · wᵀ`. 어텐션 안의 두 곱(질의와 키, 점수와 값)에 쓴다. 빠른
    /// 길에서 글이 길면 SME 커널로 계산한다(33장의 Whisper와 같다). 키와
    /// 값은 실행할 때마다 달라서 그때마다 다시 늘어놓아야 하므로, 짧은
    /// 글에서는 늘어놓는 수고가 더 든다.
    fn product(&self, x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        if self.fast
            && count >= SME_ATTENTION
            && let Some(packed) = simd::sme::Packed::new(w, rows, cols)
        {
            simd::sme::linear(x, count, &packed, y);
            return;
        }
        simd::linear(x, count, w, rows, cols, y);
    }
    // ANCHOR_END: attention

    // ANCHOR: forward
    /// 글들을 한 배치로 계산한다. `sentences`의 글마다 `[CLS]`로
    /// 시작하고 `[SEP]`로 끝나는 토큰 번호들이다.
    pub fn forward(
        &self,
        sentences: &[&[TokenId]],
        batching: Batching,
        padding: TokenId,
    ) -> Result<Trace> {
        let lengths: Vec<usize> = sentences.iter().map(|sentence| sentence.len()).collect();
        let spans = layout(&lengths, batching);
        let embedded = self.embed_tokens(sentences, &spans, padding)?;

        // 블록들. 어텐션을 더하고 정규화, MLP를 더하고 정규화.
        let mut x = embedded.clone();
        let mut outputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let attended = self.attention(layer, &x, &spans)?;
            x = ops::layer_norm(
                &ops::add(&x, &attended)?,
                &layer.norm1.0,
                &layer.norm1.1,
                self.eps,
            )?;
            let hidden = self.gelu(&self.linear(&x, &layer.fc1)?)?;
            let hidden = self.linear(&hidden, &layer.fc2)?;
            x = ops::layer_norm(
                &ops::add(&x, &hidden)?,
                &layer.norm2.0,
                &layer.norm2.1,
                self.eps,
            )?;
            outputs.push(x.clone());
        }
        let pooled = self.pool(&x, &spans)?;
        Ok(Trace {
            spans,
            embedded,
            layers: outputs,
            pooled,
        })
    }
    // ANCHOR_END: forward

    // ANCHOR: pool
    /// 글마다 진짜 토큰들의 벡터를 평균하고 길이를 1로 맞춘다. 채운
    /// 자리는 평균에 넣지 않는다.
    fn pool(&self, x: &Tensor, spans: &[Span]) -> Result<Tensor> {
        let width = self.width();
        let values = x.as_slice::<f32>()?;
        let mut pooled = vec![0.0; spans.len() * width];
        for (span, mean) in spans.iter().zip(pooled.chunks_exact_mut(width)) {
            for row in values[span.start * width..][..span.real * width].chunks_exact(width) {
                mean.iter_mut()
                    .zip(row)
                    .for_each(|(sum, value)| *sum += value);
            }
            let count = span.real.max(1) as f32;
            mean.iter_mut().for_each(|sum| *sum /= count);
            normalize(mean);
        }
        Ok(Tensor::from_vec(pooled, &[spans.len(), width])?)
    }
    // ANCHOR_END: pool

    // ANCHOR: embed_many
    /// 많은 글을 임베딩으로 바꾼다. 결과는 `(글의 수, 폭)`이고 순서는
    /// 받은 순서다.
    pub fn embed_many(
        &self,
        sentences: &[Vec<TokenId>],
        padding: TokenId,
        plan: Plan,
    ) -> Result<Vec<f32>> {
        let width = self.width();
        // 1. 순서를 정한다. 긴 글부터 놓으면 배치마다 길이가 비슷해진다.
        let mut order: Vec<usize> = (0..sentences.len()).collect();
        if plan.sorted {
            order.sort_by_key(|&index| std::cmp::Reverse(sentences[index].len()));
        }
        // 2. 배치마다 계산한다.
        let batches: Vec<&[usize]> = order.chunks(plan.batch.max(1)).collect();
        let results: Vec<Mutex<Option<Result<Tensor>>>> =
            batches.iter().map(|_| Mutex::new(None)).collect();
        let run = |index: usize| {
            let batch: Vec<&[TokenId]> = batches[index]
                .iter()
                .map(|&sentence| sentences[sentence].as_slice())
                .collect();
            let pooled = self
                .forward(&batch, plan.batching, padding)
                .map(|trace| trace.pooled);
            *results[index].lock().expect("배치마다 한 번만 잠근다") = Some(pooled);
        };
        if plan.across {
            // 배치들을 스레드에 나눈다. 배치 안의 계산은 풀이 이미 바쁘므로
            // 맡은 스레드가 혼자 한다(18장).
            let threads = badak_core::threads::global();
            threads
                .pool()
                .run(batches.len(), threads.pool().threads(), &run);
        } else {
            (0..batches.len()).for_each(run);
        }
        // 3. 원래의 순서로 돌려놓는다.
        let mut embeddings = vec![0.0; sentences.len() * width];
        for (batch, result) in batches.iter().zip(results) {
            let pooled = result
                .into_inner()
                .expect("잠금이 오염되지 않았다")
                .ok_or_else(|| invalid("계산하지 않은 배치가 있습니다".into()))??;
            let pooled = pooled.as_slice::<f32>()?;
            for (&sentence, vector) in batch.iter().zip(pooled.chunks_exact(width)) {
                embeddings[sentence * width..][..width].copy_from_slice(vector);
            }
        }
        Ok(embeddings)
    }
    // ANCHOR_END: embed_many
}

/// 벡터의 길이를 1로 맞춘다. 길이가 0이면 그대로 둔다.
pub fn normalize(vector: &mut [f32]) {
    let length = simd::dot(vector, vector).sqrt();
    if length > 0.0 {
        vector.iter_mut().for_each(|value| *value /= length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_batches_are_rectangles_and_ragged_ones_are_not() {
        let padded = layout(&[3, 5, 2], Batching::Padded);
        assert_eq!(
            padded,
            [
                Span {
                    start: 0,
                    rows: 5,
                    real: 3
                },
                Span {
                    start: 5,
                    rows: 5,
                    real: 5
                },
                Span {
                    start: 10,
                    rows: 5,
                    real: 2
                },
            ]
        );
        let ragged = layout(&[3, 5, 2], Batching::Ragged);
        assert_eq!(
            ragged,
            [
                Span {
                    start: 0,
                    rows: 3,
                    real: 3
                },
                Span {
                    start: 3,
                    rows: 5,
                    real: 5
                },
                Span {
                    start: 8,
                    rows: 2,
                    real: 2
                },
            ]
        );
        assert!(layout(&[], Batching::Padded).is_empty());
    }

    #[test]
    fn normalizing_gives_unit_length() {
        let mut vector = [3.0, 4.0];
        normalize(&mut vector);
        assert_eq!(vector, [0.6, 0.8]);
        let mut zero = [0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, [0.0, 0.0]);
    }
}
