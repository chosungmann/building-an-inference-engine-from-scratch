//! 글을 소리로 바꾸는 모델 (37장).
//!
//! OuteTTS-0.1-350M이다. 언어 모델이 글을 읽고 **오디오 토큰**을 쓴다.
//! 오디오 토큰은 소리의 짧은 토막(75분의 1초) 하나를 나타내는 번호다.
//! 그 번호들을 오디오 디코더가 소리로 바꾼다.
//!
//! 언어 모델의 구조는 2부의 Llama 그대로이고, 36장처럼 출력 층이 토큰
//! 벡터의 표와 따로다.

use std::path::Path;

use badak_core::packed::Matrix;
use badak_core::{Tensor, ops};
use badak_formats::Result;
use badak_formats::safetensors::SafeTensors;

use crate::llama::cache::KvCache;
use crate::llama::model::LlamaModel;
use crate::sampling::Sampler;
use crate::tokenizer::TokenId;

/// OuteTTS의 언어 모델.
pub struct OuteTts {
    /// 2부의 언어 모델.
    pub text: LlamaModel,
    /// 출력 층.
    head: Matrix,
}

// ANCHOR: prompt
/// 읽을 글을 모델이 학습한 틀에 넣는다. 낱말 사이를 표시 토큰으로 가르고,
/// 끝에 "여기서부터 소리"라는 표시를 둔다.
///
/// 글은 기준 구현이 하는 대로 다듬는다. 소문자로 바꾸고, 낱말을 가르는
/// 기호(`-`, `_`, `/`, `,`, `.`, `\`)는 빈칸으로 바꾸고, 그 밖에 글자가 아닌
/// 것은 지운다(`it's`는 `its`가 된다). 숫자를 낱말로 푸는 일은 하지 않는다.
/// 숫자는 지워진다.
pub fn prompt(text: &str) -> String {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .filter_map(|character| match character {
            '-' | '_' | '/' | ',' | '.' | '\\' => Some(' '),
            _ if character.is_ascii_lowercase() || character.is_whitespace() => Some(character),
            _ => None,
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    format!(
        "<|im_start|>\n<|text_start|>{}<|text_end|>\n<|audio_start|>\n",
        words.join("<|text_sep|>")
    )
}
// ANCHOR_END: prompt

impl OuteTts {
    /// 디렉터리에서 `config.json`과 `model.safetensors`를 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let file = SafeTensors::open(directory.join("model.safetensors"))?;
        Ok(Self {
            text: LlamaModel::load(directory)?,
            head: file.tensor("lm_head.weight")?.into(),
        })
    }

    fn logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let count = hidden.shape()[0];
        let row = hidden.slice(0, count.saturating_sub(1)..count)?;
        let normed = self.text.final_norm(&row)?;
        Ok(ops::linear(&normed, &self.head, None)?.to_vec::<f32>()?)
    }

    // ANCHOR: generate
    /// 글의 토큰들을 읽고 오디오 토큰이 섞인 글을 쓴다. 토큰은 12장의
    /// 샘플러가 고른다(온도가 0이면 점수가 가장 높은 것). `stop`이 나오거나
    /// `limit`개를 쓰면 멈춘다. 돌려주는 것은 쓴 토큰들과, 글을 읽은 직후의
    /// 점수다.
    pub fn generate(
        &self,
        ids: &[TokenId],
        stop: TokenId,
        limit: usize,
        sampler: &mut Sampler,
        mut each: impl FnMut(TokenId),
    ) -> Result<(Vec<TokenId>, Vec<f32>)> {
        let mut cache = KvCache::new(&self.text.config);
        let hidden = self.text.hidden_cached(ids, &mut cache)?;
        let first = self.logits(&hidden)?;
        // 반복 벌점은 읽은 글과 쓴 글 모두를 본다.
        let mut history = ids.to_vec();
        let mut logits = first.clone();
        let mut written = Vec::new();
        while written.len() < limit {
            let next = sampler.sample(&logits, &history);
            written.push(next);
            history.push(next);
            if next == stop {
                break;
            }
            each(next);
            let hidden = self.text.hidden_cached(&[next], &mut cache)?;
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
    fn the_prompt_keeps_only_lowercase_words() {
        assert_eq!(
            prompt("Hello, World! It's 5 a-b."),
            "<|im_start|>\n<|text_start|>hello<|text_sep|>world<|text_sep|>its<|text_sep|>a<|text_sep|>b<|text_end|>\n<|audio_start|>\n"
        );
    }
}

/// 오디오 디코더(WavTokenizer). 오디오 코드를 받아 토막마다의 벡터를,
/// 그리고 토막마다의 스펙트럼을 낸다. 스펙트럼을 소리로 바꾸는 마지막
/// 단계(거꾸로 가는 푸리에 변환)는 `badak_audio`에 있다.
pub mod decoder {
    use std::path::Path;

    use badak_core::{nn, simd};
    use badak_formats::safetensors::SafeTensors;
    use badak_formats::{Error, Result};

    use crate::diffusion::layers::{attend, layer_norm};

    fn invalid(message: String) -> Error {
        Error::Parse {
            format: "WavTokenizer",
            offset: 0,
            message,
        }
    }

    /// 1차원 합성곱 하나. 필터는 `(출력, 입력 × 창)`의 행렬로 본다.
    struct Conv {
        weight: Vec<f32>,
        bias: Vec<f32>,
        inputs: usize,
        size: usize,
        /// 31장의 SME 커널이 읽는 순서로 다시 늘어놓은 가중치(빠른 길).
        #[cfg(target_arch = "aarch64")]
        packed: Option<simd::sme::Packed>,
    }

    impl Conv {
        /// 빠른 길을 켠다.
        fn pack(&mut self) {
            #[cfg(target_arch = "aarch64")]
            {
                let cols = self.inputs * self.size;
                self.packed = simd::sme::Packed::new(&self.weight, self.bias.len(), cols);
            }
        }

        fn load(file: &SafeTensors, name: &str) -> Result<Self> {
            let weight = file.tensor(&format!("{name}.weight"))?;
            let (inputs, size) = match *weight.shape() {
                [_, inputs, size] => (inputs, size),
                [_, inputs] => (inputs, 1),
                _ => return Err(invalid(format!("{name}의 형상이 맞지 않습니다"))),
            };
            Ok(Self {
                weight: weight.to_vec::<f32>()?,
                bias: file.tensor(&format!("{name}.bias"))?.to_vec::<f32>()?,
                inputs,
                size,
                #[cfg(target_arch = "aarch64")]
                packed: None,
            })
        }

        // ANCHOR: conv1d
        /// `x`는 `(토막 수, 입력 채널)`이다. 앞뒤에 0을 패딩해 길이를 지킨다.
        fn apply(&self, x: &[f32]) -> Vec<f32> {
            let (inputs, size, outputs) = (self.inputs, self.size, self.bias.len());
            let frames = x.len() / inputs;
            let cols = inputs * size;
            // 창 하나는 (채널, 창 안의 자리)의 순서다. 필터가 놓인 순서와 같다.
            let mut windows = vec![0.0; frames * cols];
            for (frame, window) in windows.chunks_exact_mut(cols).enumerate() {
                for tap in 0..size {
                    let Some(at) = (frame + tap).checked_sub(size / 2) else {
                        continue;
                    };
                    if at >= frames {
                        continue;
                    }
                    for (channel, &value) in x[at * inputs..][..inputs].iter().enumerate() {
                        window[channel * size + tap] = value;
                    }
                }
            }
            let mut y = vec![0.0; frames * outputs];
            #[cfg(target_arch = "aarch64")]
            if let Some(packed) = &self.packed {
                // SME 커널은 스레드 하나로 돈다. 토막들을 나눠 스레드에
                // 맡긴다(34.8절, 35.5절).
                let pool = badak_core::threads::global().pool();
                let strip = frames.div_ceil(pool.threads()).next_multiple_of(16);
                pool.run_chunks(&mut y, strip * outputs, pool.threads(), &|index, out| {
                    let count = out.len() / outputs;
                    let input = &windows[index * strip * cols..][..count * cols];
                    simd::sme::linear(input, count, packed, out);
                });
            } else {
                simd::linear(&windows, frames, &self.weight, outputs, cols, &mut y);
            }
            #[cfg(not(target_arch = "aarch64"))]
            simd::linear(&windows, frames, &self.weight, outputs, cols, &mut y);
            for row in y.chunks_exact_mut(outputs) {
                row.iter_mut()
                    .zip(&self.bias)
                    .for_each(|(value, b)| *value += b);
            }
            y
        }
        // ANCHOR_END: conv1d
    }

    /// 그룹 정규화(35.6절)를 `(토막 수, 채널)`에. 묶음은 32개다.
    fn group_norm(x: &mut [f32], weight: &[f32], bias: &[f32], swish: bool) {
        let channels = weight.len();
        let per_group = channels / 32;
        let frames = x.len() / channels;
        let (mut sums, mut squares) = (vec![0.0_f64; channels], vec![0.0_f64; channels]);
        for row in x.chunks_exact(channels) {
            for ((sum, square), &value) in sums.iter_mut().zip(&mut squares).zip(row) {
                *sum += f64::from(value);
                *square += f64::from(value) * f64::from(value);
            }
        }
        let count = (frames * per_group) as f64;
        let (mut scale, mut shift) = (vec![0.0_f32; channels], vec![0.0_f32; channels]);
        for group in 0..32 {
            let range = group * per_group..(group + 1) * per_group;
            let mean = sums[range.clone()].iter().sum::<f64>() / count;
            let variance = squares[range.clone()].iter().sum::<f64>() / count - mean * mean;
            let inverse = 1.0 / (variance.max(0.0) + 1e-6).sqrt();
            for channel in range {
                let a = inverse * f64::from(weight[channel]);
                scale[channel] = a as f32;
                shift[channel] = (f64::from(bias[channel]) - mean * a) as f32;
            }
        }
        for row in x.chunks_exact_mut(channels) {
            for ((value, a), c) in row.iter_mut().zip(&scale).zip(&shift) {
                *value = *value * a + c;
                if swish {
                    *value = nn::silu(*value);
                }
            }
        }
    }

    struct Resnet {
        norm1: (Vec<f32>, Vec<f32>),
        conv1: Conv,
        norm2: (Vec<f32>, Vec<f32>),
        conv2: Conv,
    }

    struct Next {
        /// 채널마다 따로 하는 합성곱. `(채널, 창)`.
        depthwise: Vec<f32>,
        depthwise_bias: Vec<f32>,
        norm: (Vec<f32>, Vec<f32>),
        up: Conv,
        down: Conv,
        gamma: Vec<f32>,
    }

    /// 디코더의 몸통.
    pub struct Backbone {
        codebook: Vec<f32>,
        embed: Conv,
        resnets: Vec<Resnet>,
        attention_norm: (Vec<f32>, Vec<f32>),
        attention: [Conv; 4],
        last_norm: (Vec<f32>, Vec<f32>),
        norm: (Vec<f32>, Vec<f32>),
        blocks: Vec<Next>,
        final_norm: (Vec<f32>, Vec<f32>),
        /// 벡터를 주파수마다의 크기와 위상으로 바꾸는 선형 층.
        head: Conv,
        code_width: usize,
        fast: bool,
    }

    impl Backbone {
        /// `reference/ch37_convert.py`가 만든 safetensors 파일을 읽는다.
        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            let file = SafeTensors::open(path)?;
            let vector =
                |name: &str| -> Result<Vec<f32>> { Ok(file.tensor(name)?.to_vec::<f32>()?) };
            let pair = |name: &str, a: &str, b: &str| -> Result<(Vec<f32>, Vec<f32>)> {
                Ok((
                    vector(&format!("{name}.{a}"))?,
                    vector(&format!("{name}.{b}"))?,
                ))
            };
            let codebook = file.tensor("codebook")?;
            let code_width = codebook.shape()[1];
            let mut resnets = Vec::new();
            for index in [0, 1, 3, 4] {
                let name = format!("backbone.pos_net.{index}");
                resnets.push(Resnet {
                    norm1: pair(&format!("{name}.norm1"), "weight", "bias")?,
                    conv1: Conv::load(&file, &format!("{name}.conv1"))?,
                    norm2: pair(&format!("{name}.norm2"), "weight", "bias")?,
                    conv2: Conv::load(&file, &format!("{name}.conv2"))?,
                });
            }
            let mut blocks = Vec::new();
            while file
                .names()
                .any(|name| name == format!("backbone.convnext.{}.gamma", blocks.len()))
            {
                let name = format!("backbone.convnext.{}", blocks.len());
                blocks.push(Next {
                    depthwise: vector(&format!("{name}.dwconv.weight"))?,
                    depthwise_bias: vector(&format!("{name}.dwconv.bias"))?,
                    norm: pair(&format!("{name}.norm"), "scale", "shift")?,
                    up: Conv::load(&file, &format!("{name}.pwconv1"))?,
                    down: Conv::load(&file, &format!("{name}.pwconv2"))?,
                    gamma: vector(&format!("{name}.gamma"))?,
                });
            }
            let attention = "backbone.pos_net.2";
            Ok(Self {
                codebook: codebook.to_vec::<f32>()?,
                embed: Conv::load(&file, "backbone.embed")?,
                resnets,
                attention_norm: pair(&format!("{attention}.norm"), "weight", "bias")?,
                attention: [
                    Conv::load(&file, &format!("{attention}.q"))?,
                    Conv::load(&file, &format!("{attention}.k"))?,
                    Conv::load(&file, &format!("{attention}.v"))?,
                    Conv::load(&file, &format!("{attention}.proj_out"))?,
                ],
                last_norm: pair("backbone.pos_net.5", "weight", "bias")?,
                norm: pair("backbone.norm", "scale", "shift")?,
                blocks,
                final_norm: pair("backbone.final_layer_norm", "weight", "bias")?,
                head: Conv::load(&file, "head.out")?,
                fast: false,
                code_width,
            })
        }

        // ANCHOR: spectrum
        /// 토막마다의 벡터를 토막마다의 스펙트럼으로 바꾼다. 토막마다
        /// 1,282개이고 앞의 641개는 주파수마다의 크기의 로그, 뒤의 641개는
        /// 위상이다. 이것을 소리로 바꾸는 일은 `badak_audio::Istft`가 한다.
        pub fn spectrum(&self, hidden: &[f32]) -> Vec<f32> {
            self.head.apply(hidden)
        }
        // ANCHOR_END: spectrum

        /// 빠른 길을 켠다. 합성곱과 선형 층을 31장의 SME 커널로 계산한다.
        pub fn fast(mut self) -> Self {
            let mut layers = vec![&mut self.embed, &mut self.head];
            layers.extend(self.attention.iter_mut());
            for block in &mut self.resnets {
                layers.extend([&mut block.conv1, &mut block.conv2]);
            }
            for block in &mut self.blocks {
                layers.extend([&mut block.up, &mut block.down]);
            }
            layers.into_iter().for_each(Conv::pack);
            self.fast = true;
            self
        }

        fn resnet(block: &Resnet, x: &[f32]) -> Vec<f32> {
            let mut hidden = x.to_vec();
            group_norm(&mut hidden, &block.norm1.0, &block.norm1.1, true);
            let mut hidden = block.conv1.apply(&hidden);
            group_norm(&mut hidden, &block.norm2.0, &block.norm2.1, true);
            let mut hidden = block.conv2.apply(&hidden);
            hidden.iter_mut().zip(x).for_each(|(a, b)| *a += b);
            hidden
        }

        // ANCHOR: decode
        /// 오디오 코드들을 토막마다의 벡터로 바꾼다. `(코드 수, 폭)`.
        pub fn forward(&self, codes: &[u32]) -> Result<Vec<f32>> {
            let width = self.final_norm.0.len();
            let mut features = Vec::with_capacity(codes.len() * self.code_width);
            for &code in codes {
                let row = self
                    .codebook
                    .get(code as usize * self.code_width..(code as usize + 1) * self.code_width)
                    .ok_or_else(|| invalid(format!("오디오 코드 {code}가 코드북 밖입니다")))?;
                features.extend_from_slice(row);
            }
            let mut x = self.embed.apply(&features);

            // 잔차 블록 둘, 어텐션, 잔차 블록 둘, 그룹 정규화.
            x = Self::resnet(&self.resnets[0], &x);
            x = Self::resnet(&self.resnets[1], &x);
            let mut normed = x.clone();
            group_norm(
                &mut normed,
                &self.attention_norm.0,
                &self.attention_norm.1,
                false,
            );
            let [query, key, value, out] = &self.attention;
            let mixed = attend(
                &query.apply(&normed),
                &key.apply(&normed),
                &value.apply(&normed),
                width,
                1,
                false,
            );
            let mixed = out.apply(&mixed);
            x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
            x = Self::resnet(&self.resnets[2], &x);
            x = Self::resnet(&self.resnets[3], &x);
            group_norm(&mut x, &self.last_norm.0, &self.last_norm.1, false);
            layer_norm(&mut x, &self.norm.0, &self.norm.1, 1e-6);

            // ConvNeXt 블록들.
            let frames = codes.len();
            for block in &self.blocks {
                let size = block.depthwise.len() / width;
                let mut hidden = vec![0.0; x.len()];
                for frame in 0..frames {
                    for tap in 0..size {
                        let Some(at) = (frame + tap).checked_sub(size / 2) else {
                            continue;
                        };
                        if at >= frames {
                            continue;
                        }
                        let source = &x[at * width..][..width];
                        let target = &mut hidden[frame * width..][..width];
                        for (channel, (t, s)) in target.iter_mut().zip(source).enumerate() {
                            *t += s * block.depthwise[channel * size + tap];
                        }
                    }
                }
                for row in hidden.chunks_exact_mut(width) {
                    row.iter_mut()
                        .zip(&block.depthwise_bias)
                        .for_each(|(value, b)| *value += b);
                }
                layer_norm(&mut hidden, &block.norm.0, &block.norm.1, 1e-6);
                let mut wide = block.up.apply(&hidden);
                // 값이 블록마다 (토막 수 × 2,304)개다. 스레드에 나누고, 빠른
                // 길에서는 지수 함수를 근사한 판을 쓴다(32장, 36.9절).
                let pool = badak_core::threads::global().pool();
                let piece = wide.len().div_ceil(4 * pool.threads()).max(1);
                let fast = self.fast;
                pool.run_chunks(&mut wide, piece, pool.threads(), &|_, chunk| {
                    if fast {
                        nn::gelu_in_place(chunk);
                    } else {
                        chunk.iter_mut().for_each(|value| *value = nn::gelu(*value));
                    }
                });
                let narrow = block.down.apply(&wide);
                for (row, add) in x.chunks_exact_mut(width).zip(narrow.chunks_exact(width)) {
                    for ((value, a), gamma) in row.iter_mut().zip(add).zip(&block.gamma) {
                        *value += a * gamma;
                    }
                }
            }
            layer_norm(&mut x, &self.final_norm.0, &self.final_norm.1, 1e-6);
            Ok(x)
        }
        // ANCHOR_END: decode
    }
}

// ANCHOR: voice
/// 오디오 코드를 소리로 바꾸는 것 전체. 디코더와 거꾸로 가는 푸리에 변환.
pub struct Voice {
    backbone: decoder::Backbone,
    istft: badak_audio::Istft,
}

/// 오디오 코드 하나가 차지하는 표본의 수. 24kHz에서 75분의 1초다.
pub const SAMPLES_PER_CODE: usize = 320;
/// 표본 속도.
pub const SAMPLE_RATE: u32 = 24_000;

impl Voice {
    /// `reference/ch37_convert.py`가 만든 safetensors 파일을 읽는다.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            backbone: decoder::Backbone::load(path)?,
            istft: badak_audio::Istft::new(4 * SAMPLES_PER_CODE, SAMPLES_PER_CODE),
        })
    }

    /// 빠른 길을 켠다.
    pub fn fast(mut self) -> Self {
        self.backbone = self.backbone.fast();
        self
    }

    /// 디코더의 몸통.
    pub fn backbone(&self) -> &decoder::Backbone {
        &self.backbone
    }

    /// 오디오 코드 전체를 한 번에 소리로 바꾼다. 코드 하나가 표본 320개다.
    pub fn decode(&self, codes: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.backbone.forward(codes)?;
        Ok(self.istft.synthesize(&self.backbone.spectrum(&hidden)))
    }

    /// 코드 `from..until`의 소리만 만든다. 스트리밍에 쓴다.
    ///
    /// 디코더는 앞뒤의 코드를 본다(합성곱, 어텐션). 그래서 조각만 따로 풀면
    /// 가장자리의 소리가 전체를 한 번에 풀었을 때와 달라진다. 앞의 코드
    /// `before`개와 뒤의 코드 `after`개(있으면)를 함께 넣어 풀고, 가운데만
    /// 돌려준다.
    pub fn decode_part(
        &self,
        codes: &[u32],
        from: usize,
        until: usize,
        before: usize,
        after: usize,
    ) -> Result<Vec<f32>> {
        let until = until.min(codes.len());
        let (low, high) = (
            from.saturating_sub(before),
            until.saturating_add(after).min(codes.len()),
        );
        let audio = self.decode(&codes[low..high])?;
        let start = (from - low) * SAMPLES_PER_CODE;
        Ok(audio[start..start + (until - from) * SAMPLES_PER_CODE].to_vec())
    }
}
// ANCHOR_END: voice
