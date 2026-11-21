//! 음성 인식 모델 Whisper (33장).
//!
//! 두 부분이다. **인코더**는 소리의 그림(멜 스펙트로그램)을 한 번 읽어 벡터의
//! 줄로 바꾼다. 32장의 ViT와 같은 뼈대다. **디코더**는 그 벡터들을 보면서
//! 글의 토큰을 하나씩 낸다. 2부의 Llama와 같은 뼈대이고, 한 가지가 더
//! 있다. 블록마다 인코더의 출력을 들여다보는 어텐션(교차 어텐션)이다.

use std::collections::BTreeMap;
use std::path::Path;

use badak_core::{Tensor, ops, prob, simd};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result, json};

use crate::tokenizer::unicode::char_to_byte;

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: "Whisper",
        offset: 0,
        message: message.into(),
    }
}

/// 층 정규화의 `eps`.
const EPS: f32 = 1e-5;

// ANCHOR: dense
/// 선형 층 하나. 가중치와 치우침, 그리고 31장의 SME 커널이 읽을 수 있게
/// 다시 늘어놓은 가중치(빠른 길을 켰을 때만).
struct Dense {
    weight: Tensor,
    bias: Option<Tensor>,
    #[cfg(target_arch = "aarch64")]
    packed: Option<simd::sme::Packed>,
}

impl Dense {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
            #[cfg(target_arch = "aarch64")]
            packed: None,
        }
    }

    /// `x`는 `(벡터 수, 입력 폭)`이다.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(target_arch = "aarch64")]
        if let Some(packed) = &self.packed {
            let (count, rows) = (x.len() / packed.cols(), packed.rows());
            let input = x.contiguous();
            let input = input.as_slice::<f32>()?;
            let bias = self.bias.as_ref().map(Tensor::contiguous);
            let bias = bias.as_ref().map(Tensor::as_slice::<f32>).transpose()?;
            return Ok(Tensor::build(&[count, rows], |y| {
                simd::sme::linear(input, count, packed, y);
                if let Some(bias) = bias {
                    for row in y.chunks_exact_mut(rows) {
                        row.iter_mut().zip(bias).for_each(|(value, b)| *value += b);
                    }
                }
                Ok(())
            })?);
        }
        Ok(ops::linear(x, &self.weight, self.bias.as_ref())?)
    }

    /// SME 커널이 읽을 수 있게 가중치를 다시 늘어놓아 둔다. SME가 없는
    /// 기계에서는 아무 일도 하지 않는다.
    fn pack(&mut self) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        if let &[rows, cols] = self.weight.shape() {
            let values = self.weight.contiguous();
            self.packed = simd::sme::Packed::new(values.as_slice::<f32>()?, rows, cols);
        }
        Ok(())
    }
}
// ANCHOR_END: dense

/// 어텐션 하나의 선형 층 넷. 키를 만드는 층에는 치우침이 없다.
struct Attention {
    query: Dense,
    key: Dense,
    value: Dense,
    out: Dense,
}

struct EncoderLayer {
    norm1: (Tensor, Tensor),
    attention: Attention,
    norm2: (Tensor, Tensor),
    fc1: Dense,
    fc2: Dense,
}

struct DecoderLayer {
    norm1: (Tensor, Tensor),
    attention: Attention,
    norm_cross: (Tensor, Tensor),
    cross: Attention,
    norm2: (Tensor, Tensor),
    fc1: Dense,
    fc2: Dense,
}

// ANCHOR: model
/// Whisper.
pub struct Whisper {
    /// 인코더의 첫 합성곱. 필터를 `(폭, 멜 칸 × 3)` 행렬로 편 것이다.
    conv1: Dense,
    /// 둘째 합성곱. 스트라이드가 2다.
    conv2: Dense,
    /// 인코더의 자리마다 더하는 벡터. `(1500, 폭)`.
    encoder_positions: Tensor,
    encoder_layers: Vec<EncoderLayer>,
    encoder_norm: (Tensor, Tensor),
    /// 토큰의 임베딩 표. `(토큰 수, 폭)`. 디코더의 끝에서 점수를 낼 때 한 번
    /// 더 쓴다(10.2절).
    tokens: Tensor,
    /// 디코더의 자리마다 더하는 벡터. `(448, 폭)`.
    decoder_positions: Tensor,
    decoder_layers: Vec<DecoderLayer>,
    decoder_norm: (Tensor, Tensor),
    mels: usize,
    width: usize,
    heads: usize,
    fast: bool,
}

/// 인코더가 소리 하나를 읽은 결과.
pub struct Encoded {
    /// 인코더의 출력. `(자리 수, 폭)`.
    pub output: Tensor,
    /// 디코더의 층마다, 헤드마다 미리 만들어 둔 교차 어텐션의 키와 값.
    cross: Vec<Vec<HeadCache>>,
}

/// 헤드 하나의 키와 값. 각각 `(자리 수, 헤드의 폭)`으로 이어져 있다.
#[derive(Clone, Default)]
struct HeadCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// 디코더가 지금까지 읽은 토큰들의 흔적. 층마다, 헤드마다 셀프 어텐션의
/// 키와 값이 쌓인다(11장의 KV 캐시).
#[derive(Clone)]
pub struct State {
    layers: Vec<Vec<HeadCache>>,
    position: usize,
}
// ANCHOR_END: model

impl Whisper {
    // ANCHOR: load
    /// Hugging Face의 저장소에 있는 `model.safetensors`와 `config.json`을
    /// 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let file = SafeTensors::open(directory.join("model.safetensors"))?;
        let config = json::parse(&std::fs::read_to_string(directory.join("config.json"))?)?;
        let setting = |key: &str| {
            config
                .get(key)
                .and_then(json::Value::as_usize)
                .ok_or_else(|| invalid(format!("config.json에 {key}가 없습니다")))
        };
        let (mels, width, heads) = (
            setting("num_mel_bins")?,
            setting("d_model")?,
            setting("encoder_attention_heads")?,
        );
        if heads == 0 || width % heads != 0 || heads != setting("decoder_attention_heads")? {
            return Err(invalid("헤드의 수가 폭과 맞지 않습니다"));
        }
        let tensor = |name: &str| file.tensor(&format!("model.{name}"));
        let pair = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                tensor(&format!("{name}.weight"))?,
                tensor(&format!("{name}.bias"))?,
            ))
        };
        let dense = |name: &str| -> Result<Dense> {
            let bias = tensor(&format!("{name}.bias")).ok();
            Ok(Dense::new(tensor(&format!("{name}.weight"))?, bias))
        };
        let attention = |name: &str| -> Result<Attention> {
            Ok(Attention {
                query: dense(&format!("{name}.q_proj"))?,
                key: dense(&format!("{name}.k_proj"))?,
                value: dense(&format!("{name}.v_proj"))?,
                out: dense(&format!("{name}.out_proj"))?,
            })
        };
        // 합성곱의 필터 (폭, 채널, 3)을 (폭, 채널 × 3) 행렬로 편다.
        let conv = |name: &str| -> Result<Dense> {
            let (filter, bias) = pair(name)?;
            let &[out, channels, taps] = filter.shape() else {
                return Err(invalid(format!("{name}의 필터가 3차원이 아닙니다")));
            };
            Ok(Dense::new(
                filter.reshape(&[out, channels * taps])?,
                Some(bias),
            ))
        };

        let (mut encoder_layers, mut decoder_layers) = (Vec::new(), Vec::new());
        for index in 0..setting("encoder_layers")? {
            let name = format!("encoder.layers.{index}");
            encoder_layers.push(EncoderLayer {
                norm1: pair(&format!("{name}.self_attn_layer_norm"))?,
                attention: attention(&format!("{name}.self_attn"))?,
                norm2: pair(&format!("{name}.final_layer_norm"))?,
                fc1: dense(&format!("{name}.fc1"))?,
                fc2: dense(&format!("{name}.fc2"))?,
            });
        }
        for index in 0..setting("decoder_layers")? {
            let name = format!("decoder.layers.{index}");
            decoder_layers.push(DecoderLayer {
                norm1: pair(&format!("{name}.self_attn_layer_norm"))?,
                attention: attention(&format!("{name}.self_attn"))?,
                norm_cross: pair(&format!("{name}.encoder_attn_layer_norm"))?,
                cross: attention(&format!("{name}.encoder_attn"))?,
                norm2: pair(&format!("{name}.final_layer_norm"))?,
                fc1: dense(&format!("{name}.fc1"))?,
                fc2: dense(&format!("{name}.fc2"))?,
            });
        }
        Ok(Self {
            conv1: conv("encoder.conv1")?,
            conv2: conv("encoder.conv2")?,
            encoder_positions: tensor("encoder.embed_positions.weight")?,
            encoder_layers,
            encoder_norm: pair("encoder.layer_norm")?,
            tokens: tensor("decoder.embed_tokens.weight")?,
            decoder_positions: tensor("decoder.embed_positions.weight")?,
            decoder_layers,
            decoder_norm: pair("decoder.layer_norm")?,
            mels,
            width,
            heads,
            fast: false,
        })
    }
    // ANCHOR_END: load

    /// 빠른 길을 켠다. 선형 층을 31장의 SME 커널로 계산하고(쓸 수 있는
    /// 기계에서), GELU와 소프트맥스의 지수 함수를 32장의 근사로 바꾼다.
    pub fn fast(mut self) -> Result<Self> {
        self.fast = true;
        let mut layers: Vec<&mut Dense> = vec![&mut self.conv1, &mut self.conv2];
        for layer in &mut self.encoder_layers {
            let attention = &mut layer.attention;
            layers.extend([
                &mut attention.query,
                &mut attention.key,
                &mut attention.value,
                &mut attention.out,
                &mut layer.fc1,
                &mut layer.fc2,
            ]);
        }
        // 디코더의 층들은 벡터 하나씩을 받는다. 그 모양에서는 SME 커널이
        // 이득이 없어서(31.6절) 교차 어텐션의 키와 값만 늘어놓는다. 그 둘은
        // 인코더의 출력 1,500개를 한꺼번에 받는다.
        for layer in &mut self.decoder_layers {
            layers.extend([&mut layer.cross.key, &mut layer.cross.value]);
        }
        for layer in layers {
            layer.pack()?;
        }
        Ok(self)
    }

    /// 토큰의 수.
    pub fn vocabulary(&self) -> usize {
        self.tokens.shape()[0]
    }

    /// 인코더가 내는 벡터의 수. 소리 30초에 1,500개다.
    pub fn frames(&self) -> usize {
        self.encoder_positions.shape()[0]
    }

    /// 한 번에 낼 수 있는 토큰의 수.
    pub fn max_tokens(&self) -> usize {
        self.decoder_positions.shape()[0]
    }

    fn norm(x: &Tensor, norm: &(Tensor, Tensor)) -> Result<Tensor> {
        Ok(ops::layer_norm(x, &norm.0, &norm.1, EPS)?)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        if !self.fast {
            return Ok(ops::gelu(x)?);
        }
        let mut values = x.to_vec::<f32>()?;
        badak_core::nn::gelu_in_place(&mut values);
        Ok(Tensor::from_vec(values, x.shape())?)
    }

    fn softmax(&self, scores: &mut [f32]) {
        if self.fast {
            badak_core::nn::softmax_fast_in_place(scores);
        } else {
            prob::softmax_in_place(scores);
        }
    }

    /// `(벡터 수, 폭)`을 헤드마다의 `(벡터 수, 헤드의 폭)`으로 가른다.
    fn split_heads(&self, x: &Tensor) -> Result<Vec<Vec<f32>>> {
        let head = self.width / self.heads;
        let x = x.contiguous();
        let values = x.as_slice::<f32>()?;
        Ok((0..self.heads)
            .map(|index| {
                values
                    .chunks_exact(self.width)
                    .flat_map(|row| &row[index * head..][..head])
                    .copied()
                    .collect()
            })
            .collect())
    }

    // ANCHOR: conv
    /// 시간 방향의 합성곱(창 3, 패딩 1). `x`는 `(채널, 길이)`이고 결과는
    /// `(길이 / stride, 폭)`이다.
    ///
    /// 27.4절처럼 창을 펼쳐 선형 층으로 계산한다. 사진과 다른 점은 창이
    /// 한 방향으로만 움직인다는 것뿐이다.
    fn convolve(&self, x: &[f32], channels: usize, layer: &Dense, stride: usize) -> Result<Tensor> {
        let length = x.len() / channels;
        let frames = length.div_ceil(stride);
        let mut windows = vec![0.0_f32; frames * channels * 3];
        for (frame, window) in windows.chunks_exact_mut(channels * 3).enumerate() {
            for channel in 0..channels {
                for tap in 0..3 {
                    // 패딩한 자리(소리의 앞과 뒤)는 0으로 둔다.
                    let Some(at) = (frame * stride + tap).checked_sub(1) else {
                        continue;
                    };
                    if at < length {
                        window[channel * 3 + tap] = x[channel * length + at];
                    }
                }
            }
        }
        let windows = Tensor::from_vec(windows, &[frames, channels * 3])?;
        self.gelu(&layer.apply(&windows)?)
    }
    // ANCHOR_END: conv

    // ANCHOR: attend
    /// 여러 질의가 여러 키를 한꺼번에 보는 어텐션. 인코더가 쓴다.
    /// `x`는 `(자리 수, 폭)`이고 마스크가 없다.
    fn attend(&self, attention: &Attention, x: &Tensor) -> Result<Tensor> {
        let (count, head) = (x.shape()[0], self.width / self.heads);
        let queries = self.split_heads(&attention.query.apply(x)?)?;
        let keys = self.split_heads(&attention.key.apply(x)?)?;
        let values = self.split_heads(&attention.value.apply(x)?)?;
        let scale = 1.0 / (head as f32).sqrt();
        let mut mixed = vec![0.0; count * self.width];
        let mut scores = vec![0.0; count * count];
        let (mut turned, mut out) = (vec![0.0; head * count], vec![0.0; count * head]);
        let pool = badak_core::threads::global();
        let wanted = pool.for_vectors(count);
        for index in 0..self.heads {
            // 점수[i][j] = 질의 i · 키 j (32.3절).
            self.product(
                &queries[index],
                count,
                &keys[index],
                count,
                head,
                &mut scores,
            );
            // 줄마다의 소프트맥스는 서로 무관하다. 스레드에 나눈다.
            pool.pool()
                .run_chunks(&mut scores, count, wanted, &|_, row: &mut [f32]| {
                    row.iter_mut().for_each(|score| *score *= scale);
                    self.softmax(row);
                });
            for (token, value) in values[index].chunks_exact(head).enumerate() {
                for (lane, &value) in value.iter().enumerate() {
                    turned[lane * count + token] = value;
                }
            }
            self.product(&scores, count, &turned, head, count, &mut out);
            for (token, row) in out.chunks_exact(head).enumerate() {
                mixed[token * self.width + index * head..][..head].copy_from_slice(row);
            }
        }
        attention
            .out
            .apply(&Tensor::from_vec(mixed, &[count, self.width])?)
    }

    /// `y = x · wᵀ`. 어텐션 안의 두 곱(질의와 키, 점수와 값)에 쓴다. 빠른
    /// 길에서는 SME 커널로 계산한다. 가중치 쪽(키나 값)이 실행할 때마다
    /// 달라지므로 그때마다 다시 늘어놓는다.
    fn product(&self, x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        if self.fast
            && let Some(packed) = simd::sme::Packed::new(w, rows, cols)
        {
            simd::sme::linear(x, count, &packed, y);
            return;
        }
        simd::linear(x, count, w, rows, cols, y);
    }

    /// 질의 하나가 쌓아 둔 키와 값을 보는 어텐션. 디코더가 쓴다.
    fn attend_one(&self, query: &Tensor, caches: &[HeadCache]) -> Result<Vec<f32>> {
        let head = self.width / self.heads;
        let scale = 1.0 / (head as f32).sqrt();
        let query = query.to_vec::<f32>()?;
        let mut mixed = vec![0.0; self.width];
        let mut scores = Vec::new();
        for (index, cache) in caches.iter().enumerate() {
            let query = &query[index * head..][..head];
            scores.clear();
            scores.extend(
                cache
                    .keys
                    .chunks_exact(head)
                    .map(|key| simd::dot(query, key) * scale),
            );
            self.softmax(&mut scores);
            let out = &mut mixed[index * head..][..head];
            for (&score, value) in scores.iter().zip(cache.values.chunks_exact(head)) {
                for (slot, &v) in out.iter_mut().zip(value) {
                    *slot += score * v;
                }
            }
        }
        Ok(mixed)
    }
    // ANCHOR_END: attend

    // ANCHOR: encode
    /// 멜 스펙트로그램을 읽는다. `mel`은 `(멜 칸, 토막 수)`의 순서다.
    pub fn encode(&self, mel: &[f32]) -> Result<Encoded> {
        if mel.len() != self.mels * 2 * self.frames() {
            return Err(invalid(format!(
                "스펙트로그램의 값이 {}개입니다. ({}, {})이어야 합니다",
                mel.len(),
                self.mels,
                2 * self.frames()
            )));
        }
        // 1. 합성곱 둘. 둘째가 길이를 절반으로 줄인다. 토막 3,000개가 자리
        //    1,500개가 된다. 자리 하나가 소리 20밀리초다.
        let x = self.convolve(mel, self.mels, &self.conv1, 1)?;
        // 둘째 합성곱은 (채널, 길이)를 받는다. 방금의 (길이, 채널)을 뒤집는다.
        let turned = x.transpose(0, 1)?.contiguous();
        let x = self.convolve(turned.as_slice::<f32>()?, self.width, &self.conv2, 2)?;

        // 2. 자리마다의 벡터를 더하고 블록들을 지난다. 32장의 ViT와 같다.
        let mut x = ops::add(&x, &self.encoder_positions)?;
        for layer in &self.encoder_layers {
            let attended = self.attend(&layer.attention, &Self::norm(&x, &layer.norm1)?)?;
            x = ops::add(&x, &attended)?;
            let hidden = self.gelu(&layer.fc1.apply(&Self::norm(&x, &layer.norm2)?)?)?;
            x = ops::add(&x, &layer.fc2.apply(&hidden)?)?;
        }
        let output = Self::norm(&x, &self.encoder_norm)?;

        // 3. 디코더의 교차 어텐션이 볼 키와 값을 미리 만든다. 인코더의
        //    출력은 토큰을 몇 개 내든 바뀌지 않으므로 한 번이면 된다.
        let mut cross = Vec::with_capacity(self.decoder_layers.len());
        for layer in &self.decoder_layers {
            let keys = self.split_heads(&layer.cross.key.apply(&output)?)?;
            let values = self.split_heads(&layer.cross.value.apply(&output)?)?;
            cross.push(
                keys.into_iter()
                    .zip(values)
                    .map(|(keys, values)| HeadCache { keys, values })
                    .collect(),
            );
        }
        Ok(Encoded { output, cross })
    }
    // ANCHOR_END: encode

    /// 디코더의 빈 상태.
    pub fn start(&self) -> State {
        State {
            layers: vec![vec![HeadCache::default(); self.heads]; self.decoder_layers.len()],
            position: 0,
        }
    }

    // ANCHOR: step
    /// 토큰 하나를 디코더에 넣고, 다음 토큰의 점수를 받는다.
    pub fn step(&self, encoded: &Encoded, state: &mut State, token: u32) -> Result<Vec<f32>> {
        let token = token as usize;
        if token >= self.vocabulary() || state.position >= self.max_tokens() {
            return Err(invalid("토큰의 번호나 자리가 범위를 벗어납니다"));
        }
        // 1. 토큰의 벡터에 자리의 벡터를 더한다.
        let embedding = self.tokens.slice(0, token..token + 1)?;
        let position = self
            .decoder_positions
            .slice(0, state.position..state.position + 1)?;
        let mut x = ops::add(&embedding, &position)?;
        state.position += 1;

        let head = self.width / self.heads;
        for ((layer, caches), cross) in self
            .decoder_layers
            .iter()
            .zip(&mut state.layers)
            .zip(&encoded.cross)
        {
            // 2. 셀프 어텐션. 이 토큰의 키와 값을 캐시에 쌓고, 지금까지
            //    쌓인 것 전부를 본다(11.4절). 뒤의 토큰은 아직 없으므로
            //    마스크가 필요 없다.
            let normed = Self::norm(&x, &layer.norm1)?;
            let key = layer.attention.key.apply(&normed)?.to_vec::<f32>()?;
            let value = layer.attention.value.apply(&normed)?.to_vec::<f32>()?;
            for (index, cache) in caches.iter_mut().enumerate() {
                cache.keys.extend_from_slice(&key[index * head..][..head]);
                cache
                    .values
                    .extend_from_slice(&value[index * head..][..head]);
            }
            let mixed = self.attend_one(&layer.attention.query.apply(&normed)?, caches)?;
            let mixed = Tensor::from_vec(mixed, &[1, self.width])?;
            x = ops::add(&x, &layer.attention.out.apply(&mixed)?)?;

            // 3. 교차 어텐션. 질의는 글 쪽에서, 키와 값은 소리 쪽에서 온다.
            //    "지금 쓰려는 글자는 소리의 어디에 해당하는가"를 찾는 곳이다.
            let normed = Self::norm(&x, &layer.norm_cross)?;
            let mixed = self.attend_one(&layer.cross.query.apply(&normed)?, cross)?;
            let mixed = Tensor::from_vec(mixed, &[1, self.width])?;
            x = ops::add(&x, &layer.cross.out.apply(&mixed)?)?;

            // 4. MLP.
            let hidden = self.gelu(&layer.fc1.apply(&Self::norm(&x, &layer.norm2)?)?)?;
            x = ops::add(&x, &layer.fc2.apply(&hidden)?)?;
        }
        // 5. 임베딩 표를 한 번 더 써서 토큰마다의 점수를 낸다.
        let x = Self::norm(&x, &self.decoder_norm)?;
        Ok(ops::linear(&x, &self.tokens, None)?.to_vec::<f32>()?)
    }
    // ANCHOR_END: step
}

// ANCHOR: vocabulary
/// 토큰의 번호와 글 사이를 오가는 표.
///
/// Whisper의 토큰은 두 무리다. 번호가 작은 쪽은 글의 조각이고(8장의 BPE),
/// 큰 쪽은 **지시**다. "받아쓰기 시작", "영어", "번역이 아니라 받아쓰기",
/// "시각을 적지 않음", "끝" 같은 것들이고, 30초를 0.02초씩 나눈 시각 토큰
/// 1,501개도 여기에 든다.
pub struct Vocabulary {
    /// 번호마다의 바이트들. 지시 토큰은 비어 있다.
    pieces: Vec<Vec<u8>>,
    /// 지시 토큰의 이름에서 번호로.
    special: BTreeMap<String, u32>,
    /// 디코딩 내내 고르지 못하게 하는 토큰들.
    pub suppressed: Vec<u32>,
    /// 첫 토큰으로 고르지 못하게 하는 토큰들.
    pub suppressed_at_start: Vec<u32>,
}

impl Vocabulary {
    /// `vocab.json`, `added_tokens.json`, `generation_config.json`을 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let read = |name: &str| -> Result<json::Value> {
            let text = std::fs::read_to_string(directory.join(name))?;
            json::parse(&text)
        };
        let (words, added, generation) = (
            read("vocab.json")?,
            read("added_tokens.json")?,
            read("generation_config.json")?,
        );
        let table = |value: &json::Value| -> Result<Vec<(String, u32)>> {
            value
                .as_object()
                .ok_or_else(|| invalid("단어 사전이 표가 아닙니다"))?
                .iter()
                .map(|(text, id)| {
                    let id = id
                        .as_usize()
                        .ok_or_else(|| invalid("토큰의 번호가 숫자가 아닙니다"))?;
                    Ok((text.clone(), id as u32))
                })
                .collect()
        };
        let (words, added) = (table(&words)?, table(&added)?);
        let count = words
            .iter()
            .chain(&added)
            .map(|&(_, id)| id + 1)
            .max()
            .unwrap_or(0);
        let mut pieces = vec![Vec::new(); count as usize];
        let mut special = BTreeMap::new();
        for (text, id) in words {
            if text.starts_with("<|") && text.ends_with("|>") {
                special.insert(text, id);
                continue;
            }
            // 단어 사전은 바이트를 글자로 옮겨 적는다(8.4절). 되돌린다.
            pieces[id as usize] = text
                .chars()
                .map(|character| {
                    char_to_byte(character)
                        .ok_or_else(|| invalid(format!("토큰 {text:?}에 모르는 글자가 있습니다")))
                })
                .collect::<Result<_>>()?;
        }
        special.extend(added);
        let list = |key: &str| -> Vec<u32> {
            generation
                .get(key)
                .and_then(json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(json::Value::as_usize)
                        .map(|id| id as u32)
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(Self {
            pieces,
            special,
            suppressed: list("suppress_tokens"),
            suppressed_at_start: list("begin_suppress_tokens"),
        })
    }

    /// 지시 토큰의 번호. 이름은 `<|en|>`처럼 꺾쇠와 막대까지 적는다.
    pub fn special(&self, name: &str) -> Result<u32> {
        self.special
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("지시 토큰 {name}이 없습니다")))
    }

    /// 지시 토큰의 번호에서 이름으로.
    pub fn special_name(&self, id: u32) -> Option<&str> {
        self.special
            .iter()
            .find(|&(_, &other)| other == id)
            .map(|(name, _)| name.as_str())
    }

    /// 토큰들을 글로 바꾼다. 지시 토큰은 건너뛴다.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&id| self.pieces.get(id as usize))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// 받아쓰기를 시작하는 지시 토큰 넷: 시작, 언어, 받아쓰기, 시각 없이.
    pub fn prompt(&self, language: &str) -> Result<Vec<u32>> {
        Ok(vec![
            self.special("<|startoftranscript|>")?,
            self.special(&format!("<|{language}|>"))?,
            self.special("<|transcribe|>")?,
            self.special("<|notimestamps|>")?,
        ])
    }

    /// 끝을 알리는 토큰.
    pub fn end(&self) -> Result<u32> {
        self.special("<|endoftext|>")
    }
}
// ANCHOR_END: vocabulary

/// 디코딩의 결과.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// 고른 토큰들. 지시 토큰과 끝 토큰은 들어 있지 않다.
    pub tokens: Vec<u32>,
    /// 토큰마다의 로그 확률을 더한 것.
    pub log_probability: f32,
}

impl Whisper {
    // ANCHOR: suppress
    /// 점수에서 고르면 안 되는 토큰을 지우고 로그 확률로 바꾼다.
    fn log_probabilities(mut logits: Vec<f32>, vocabulary: &Vocabulary, first: bool) -> Vec<f32> {
        let banned = vocabulary.suppressed.iter();
        let at_start = vocabulary.suppressed_at_start.iter().filter(|_| first);
        for &token in banned.chain(at_start) {
            if let Some(score) = logits.get_mut(token as usize) {
                *score = f32::NEG_INFINITY;
            }
        }
        prob::log_softmax(&logits)
    }
    // ANCHOR_END: suppress

    /// 지시 토큰들을 디코더에 넣고, 그다음 토큰의 로그 확률을 받는다.
    fn prime(
        &self,
        encoded: &Encoded,
        vocabulary: &Vocabulary,
        prompt: &[u32],
    ) -> Result<(State, Vec<f32>)> {
        let mut state = self.start();
        let mut logits = Vec::new();
        for &token in prompt {
            logits = self.step(encoded, &mut state, token)?;
        }
        if logits.is_empty() {
            return Err(invalid("지시 토큰이 없습니다"));
        }
        Ok((state, Self::log_probabilities(logits, vocabulary, true)))
    }

    // ANCHOR: greedy
    /// 탐욕 디코딩. 걸음마다 가장 높은 토큰 하나를 고른다(12장).
    pub fn greedy(
        &self,
        encoded: &Encoded,
        vocabulary: &Vocabulary,
        prompt: &[u32],
        limit: usize,
    ) -> Result<Transcript> {
        let end = vocabulary.end()?;
        let (mut state, mut scores) = self.prime(encoded, vocabulary, prompt)?;
        let mut transcript = Transcript {
            tokens: Vec::new(),
            log_probability: 0.0,
        };
        while transcript.tokens.len() < limit && state.position < self.max_tokens() {
            let best = (0..scores.len()).fold(0, |best, token| {
                if scores[token] > scores[best] {
                    token
                } else {
                    best
                }
            });
            transcript.log_probability += scores[best];
            if best as u32 == end {
                break;
            }
            transcript.tokens.push(best as u32);
            let logits = self.step(encoded, &mut state, best as u32)?;
            scores = Self::log_probabilities(logits, vocabulary, false);
        }
        Ok(transcript)
    }
    // ANCHOR_END: greedy

    // ANCHOR: beam
    /// 빔 서치. 걸음마다 후보를 `width`개 들고 간다.
    ///
    /// 탐욕 디코딩은 걸음마다 가장 좋아 보이는 토큰 하나만 따라간다. 그
    /// 선택이 뒤에서 나쁜 결과로 이어져도 되돌아갈 수 없다. 빔 서치는
    /// 후보(빔)마다 다음 토큰을 모두 펼쳐 보고, **지금까지의 로그 확률의
    /// 합**이 큰 `width`개만 남긴다.
    ///
    /// 끝 토큰의 처리가 까다롭다. 끝 토큰은 **그 걸음의 상위 `width`개 안에
    /// 들었을 때만** 받아들여, 다 쓴 글로 따로 모은다. 그러지 않고 끝 토큰이
    /// 보일 때마다 모으면 쓰다 만 글이 잔뜩 쌓인다. 다 쓴 글은 토큰 하나당의
    /// 평균 로그 확률로 견준다(합으로 견주면 짧은 글이 늘 이긴다). 다 쓴
    /// 글이 `width`개 모였고, 아직 쓰고 있는 후보 가운데 가장 좋은 것도 그
    /// 가운데 가장 나쁜 것을 넘지 못하면 멈춘다.
    pub fn beam(
        &self,
        encoded: &Encoded,
        vocabulary: &Vocabulary,
        prompt: &[u32],
        width: usize,
        limit: usize,
    ) -> Result<Transcript> {
        struct Beam {
            state: State,
            scores: Vec<f32>,
            transcript: Transcript,
        }
        // 끝 토큰까지 센 길이로 나눈 평균.
        let average = |transcript: &Transcript| {
            transcript.log_probability / (transcript.tokens.len() + 1) as f32
        };
        let end = vocabulary.end()? as usize;
        let (state, scores) = self.prime(encoded, vocabulary, prompt)?;
        let empty = Transcript {
            tokens: Vec::new(),
            log_probability: 0.0,
        };
        let mut beams = vec![Beam {
            state,
            scores,
            transcript: empty,
        }];
        let mut finished: Vec<Transcript> = Vec::new();
        for _ in 0..limit {
            // 살아 있는 후보마다 가장 좋은 다음 토큰들을 모아 한 줄로 세운다.
            // (후보의 번호, 토큰, 합친 로그 확률). 후보 하나에서 `2 × width`개
            // 넘게 뽑힐 일은 없다.
            let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
            for (index, beam) in beams.iter().enumerate() {
                let mut order: Vec<usize> = (0..beam.scores.len()).collect();
                let count = (2 * width).min(order.len());
                order.select_nth_unstable_by(count - 1, |&a, &b| {
                    beam.scores[b].total_cmp(&beam.scores[a])
                });
                for &token in &order[..count] {
                    let total = beam.transcript.log_probability + beam.scores[token];
                    candidates.push((index, token, total));
                }
            }
            candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.1.cmp(&b.1)));
            let mut next = Vec::with_capacity(width);
            for (rank, &(index, token, total)) in candidates.iter().enumerate() {
                let mut transcript = beams[index].transcript.clone();
                transcript.log_probability = total;
                if token == end {
                    // 상위 `width`개 밖의 끝 토큰은 버린다.
                    if rank < width {
                        finished.push(transcript);
                    }
                    continue;
                }
                let mut state = beams[index].state.clone();
                if state.position >= self.max_tokens() {
                    continue;
                }
                transcript.tokens.push(token as u32);
                let logits = self.step(encoded, &mut state, token as u32)?;
                next.push(Beam {
                    state,
                    scores: Self::log_probabilities(logits, vocabulary, false),
                    transcript,
                });
                if next.len() == width {
                    break;
                }
            }
            beams = next;
            // 다 쓴 글은 좋은 것 `width`개만 쥔다.
            finished.sort_by(|a, b| average(b).total_cmp(&average(a)));
            finished.truncate(width);
            let Some(best) = beams.first() else {
                break;
            };
            if let Some(worst) = finished.get(width - 1)
                && average(&best.transcript) <= average(worst)
            {
                break;
            }
        }
        // 끝까지 다 쓴 것이 없으면 쓰다 만 것 가운데서 고른다.
        if finished.is_empty() {
            finished.extend(beams.into_iter().map(|beam| beam.transcript));
        }
        finished
            .into_iter()
            .max_by(|a, b| average(a).total_cmp(&average(b)))
            .ok_or_else(|| invalid("후보가 없습니다"))
    }
    // ANCHOR_END: beam

    // ANCHOR: language
    /// 소리가 어느 언어인지 알아낸다. 시작 토큰 하나만 넣고, 언어를 뜻하는
    /// 지시 토큰들 가운데 점수가 가장 높은 것을 고른다. 이름(`en`, `ko`)과
    /// 확률을 돌려준다.
    pub fn detect_language(
        &self,
        encoded: &Encoded,
        vocabulary: &Vocabulary,
    ) -> Result<(String, f32)> {
        let mut state = self.start();
        let start = vocabulary.special("<|startoftranscript|>")?;
        let logits = self.step(encoded, &mut state, start)?;
        // 언어 토큰은 시작 토큰 바로 뒤부터 `<|translate|>` 앞까지다.
        let last = vocabulary.special("<|translate|>")?;
        let languages = &logits[start as usize + 1..last as usize];
        let probabilities = prob::softmax(languages);
        let best = (0..probabilities.len())
            .max_by(|&a, &b| probabilities[a].total_cmp(&probabilities[b]))
            .ok_or_else(|| invalid("언어 토큰이 없습니다"))?;
        let name = vocabulary
            .special_name(start + 1 + best as u32)
            .ok_or_else(|| invalid("언어 토큰의 이름이 없습니다"))?;
        Ok((
            name.trim_matches(['<', '|', '>']).to_string(),
            probabilities[best],
        ))
    }
    // ANCHOR_END: language
}
