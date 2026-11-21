//! 비전 트랜스포머 ViT (32장).
//!
//! 사진을 작은 조각(patch)들로 자르고, 조각 하나하나를 토큰처럼 다룬다.
//! 그다음은 2부의 트랜스포머와 같다. 어텐션과 MLP를 번갈아 쌓는다. 다른
//! 점은 셋이다. 어텐션에 마스크가 없고(모든 조각이 모든 조각을 본다),
//! 위치를 학습된 벡터로 더하고, 정규화가 층 정규화다.

use std::path::Path;

use badak_core::{Tensor, ops, prob, simd};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

fn invalid(message: String) -> Error {
    Error::Parse {
        format: "ViT",
        offset: 0,
        message,
    }
}

// ANCHOR: model
/// 블록 하나의 가중치.
struct Block {
    norm1: (Tensor, Tensor),
    /// 질의·키·값을 한꺼번에 만드는 선형 층. `(3 × 폭, 폭)`.
    qkv: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
}

/// ViT.
pub struct Vit {
    /// 조각 하나를 벡터로 바꾸는 선형 층. `(폭, 3 × 조각 × 조각)`.
    patch: (Tensor, Tensor),
    /// 맨 앞에 끼워 넣는, 분류에 쓸 토큰. `(폭)`.
    class_token: Tensor,
    /// 자리마다 더하는 벡터. `(토큰 수, 폭)`.
    positions: Tensor,
    blocks: Vec<Block>,
    norm: (Tensor, Tensor),
    head: (Tensor, Tensor),
    /// 조각의 한 변(픽셀).
    patch_size: usize,
    /// 사진의 한 변(픽셀).
    image_size: usize,
    heads: usize,
    /// 31장의 SME 커널과 근사한 지수 함수를 쓸지. [`Vit::fast`]가 켠다.
    fast: bool,
    /// `fast`일 때 쓰는, 다시 늘어놓은 가중치. 가중치의 값이 놓인 주소로
    /// 찾는다(모델을 옮겨도 그 주소는 그대로다).
    #[cfg(target_arch = "aarch64")]
    packed: Vec<(usize, badak_core::simd::sme::Packed)>,
}

/// 순전파의 중간 결과. 정답과 단계마다 대조할 때 쓴다.
pub struct Trace {
    /// 조각들을 벡터로 바꾼 것. `(조각 수, 폭)`.
    pub patches: Tensor,
    /// 분류 토큰을 끼우고 위치를 더한 것. `(토큰 수, 폭)`.
    pub embedded: Tensor,
    /// 블록마다의 출력.
    pub blocks: Vec<Tensor>,
    /// 마지막 정규화를 지난 것.
    pub normed: Tensor,
    /// 이름 1,000가지의 점수.
    pub logits: Tensor,
}
// ANCHOR_END: model

/// 층 정규화의 `eps`. timm의 ViT가 쓰는 값이다.
const EPS: f32 = 1e-6;

impl Vit {
    // ANCHOR: load
    /// timm이 저장한 safetensors 파일에서 가중치를 읽는다. `heads`는
    /// 어텐션 헤드의 수다. 파일에 적혀 있지 않아서 받는다.
    pub fn load(path: impl AsRef<Path>, heads: usize) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let pair = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                file.tensor(&format!("{name}.weight"))?,
                file.tensor(&format!("{name}.bias"))?,
            ))
        };
        // 조각을 벡터로 바꾸는 층은 파일에 합성곱의 필터 (폭, 3, 조각, 조각)로
        // 들어 있다. 스트라이드가 창의 크기와 같은 합성곱은 조각마다 한 번씩
        // 하는 선형 층이다(27.5절). 필터를 행렬로 펴서 쓴다.
        let (filter, patch_bias) = pair("patch_embed.proj")?;
        let &[width, 3, patch_size, columns] = filter.shape() else {
            return Err(invalid(format!(
                "조각 층의 형상이 {:?}입니다",
                filter.shape()
            )));
        };
        if patch_size != columns || patch_size == 0 || heads == 0 || width % heads != 0 {
            return Err(invalid(
                "조각이 정사각형이 아니거나 헤드의 수가 맞지 않습니다".into(),
            ));
        }
        let patch = (
            filter.reshape(&[width, 3 * patch_size * patch_size])?,
            patch_bias,
        );

        let positions = file.tensor("pos_embed")?;
        let &[1, tokens, _] = positions.shape() else {
            return Err(invalid(
                "위치 벡터의 형상이 (1, 토큰 수, 폭)이 아닙니다".into(),
            ));
        };
        let side = (tokens - 1).isqrt();
        if side * side + 1 != tokens {
            return Err(invalid(format!(
                "토큰 수 {tokens}가 조각의 정사각 격자 + 1이 아닙니다"
            )));
        }
        let mut blocks = Vec::new();
        while file
            .names()
            .any(|name| name == format!("blocks.{}.norm1.weight", blocks.len()))
        {
            let name = format!("blocks.{}", blocks.len());
            blocks.push(Block {
                norm1: pair(&format!("{name}.norm1"))?,
                qkv: pair(&format!("{name}.attn.qkv"))?,
                proj: pair(&format!("{name}.attn.proj"))?,
                norm2: pair(&format!("{name}.norm2"))?,
                fc1: pair(&format!("{name}.mlp.fc1"))?,
                fc2: pair(&format!("{name}.mlp.fc2"))?,
            });
        }
        Ok(Self {
            patch,
            class_token: file.tensor("cls_token")?.reshape(&[width])?,
            positions: positions.reshape(&[tokens, width])?,
            blocks,
            norm: pair("norm")?,
            head: pair("head")?,
            patch_size,
            image_size: side * patch_size,
            heads,
            fast: false,
            #[cfg(target_arch = "aarch64")]
            packed: Vec::new(),
        })
    }
    // ANCHOR_END: load

    // ANCHOR: fast
    /// 빠른 길을 켠다. 선형 층을 31장의 SME 커널로 계산하고(쓸 수 있는
    /// 기계에서), GELU와 소프트맥스의 지수 함수를 근사로 바꾼다. 결과가
    /// 느린 길과 비트까지 같지는 않다.
    pub fn fast(mut self) -> Result<Self> {
        self.fast = true;
        #[cfg(target_arch = "aarch64")]
        if badak_core::simd::sme::available() {
            let mut weights = vec![&self.patch.0, &self.head.0];
            for block in &self.blocks {
                weights.extend([&block.qkv.0, &block.proj.0, &block.fc1.0, &block.fc2.0]);
            }
            for weight in weights {
                let &[rows, cols] = weight.shape() else {
                    return Err(invalid("가중치가 행렬이 아닙니다".into()));
                };
                // 파일을 빌려 쓰거나 힙에 있는, 이미 이어진 값들이다.
                let values = weight.as_slice::<f32>()?;
                let packed = badak_core::simd::sme::Packed::new(values, rows, cols)
                    .ok_or_else(|| invalid("가중치를 늘어놓지 못했습니다".into()))?;
                self.packed.push((values.as_ptr() as usize, packed));
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

    /// GELU. 빠른 길에서는 지수 함수를 근사한 버전을 쓴다.
    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        if !self.fast {
            return Ok(ops::gelu(x)?);
        }
        let mut values = x.to_vec::<f32>()?;
        badak_core::nn::gelu_in_place(&mut values);
        Ok(Tensor::from_vec(values, x.shape())?)
    }
    // ANCHOR_END: fast

    /// 사진의 한 변(픽셀).
    pub fn image_size(&self) -> usize {
        self.image_size
    }

    /// 블록의 수.
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// 토큰 하나의 폭.
    pub fn width(&self) -> usize {
        self.class_token.len()
    }

    /// 토큰의 수. 조각들과 분류 토큰 하나.
    pub fn tokens(&self) -> usize {
        self.positions.shape()[0]
    }

    // ANCHOR: patches
    /// 사진을 조각으로 잘라 조각마다 벡터 하나로 편다.
    ///
    /// `image`는 `(3, 한 변, 한 변)`이다. 결과는 `(조각 수, 3 × 조각 ×
    /// 조각)`이고, 조각 안의 순서는 필터와 같은 (채널, 세로, 가로)다.
    fn cut(&self, image: &[f32]) -> Result<Tensor> {
        let (side, patch) = (self.image_size, self.patch_size);
        if image.len() != 3 * side * side {
            return Err(invalid(format!(
                "사진의 값이 {}개입니다. (3, {side}, {side})이어야 합니다",
                image.len()
            )));
        }
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
        Ok(Tensor::from_vec(flat, &[grid * grid, 3 * patch * patch])?)
    }
    // ANCHOR_END: patches

    // ANCHOR: attention
    /// 셀프 어텐션. `x`는 `(토큰 수, 폭)`이다.
    ///
    /// 9장의 어텐션에서 마스크와 위치 회전을 뺀 것이다. 모든 토큰이
    /// 모든 토큰을 본다.
    fn attention(&self, block: &Block, x: &Tensor) -> Result<Tensor> {
        let (tokens, width) = (self.tokens(), self.width());
        let head = width / self.heads;
        // 선형 층 하나가 질의, 키, 값을 이어 붙인 (토큰 수, 3 × 폭)을 낸다.
        let qkv = self.linear(x, &block.qkv)?;
        let qkv = qkv.as_slice::<f32>()?;
        let part = |token: usize, which: usize, index: usize| {
            &qkv[token * 3 * width + which * width + index * head..][..head]
        };
        let scale = 1.0 / (head as f32).sqrt();
        let mut mixed = vec![0.0; tokens * width];
        let (mut queries, mut keys) = (vec![0.0; tokens * head], vec![0.0; tokens * head]);
        // 값은 (헤드의 폭, 토큰 수)로 뒤집어 둔다. 선형 층의 커널이 가중치를
        // (행, 열)로 받기 때문이다.
        let mut values = vec![0.0; head * tokens];
        let (mut scores, mut out) = (vec![0.0; tokens * tokens], vec![0.0; tokens * head]);
        for index in 0..self.heads {
            for token in 0..tokens {
                queries[token * head..][..head].copy_from_slice(part(token, 0, index));
                keys[token * head..][..head].copy_from_slice(part(token, 1, index));
                for (lane, &value) in part(token, 2, index).iter().enumerate() {
                    values[lane * tokens + token] = value;
                }
            }
            // 점수[i][j] = 질의 i · 키 j. "키들을 가중치로 하는 선형 층"이다.
            simd::linear(&queries, tokens, &keys, tokens, head, &mut scores);
            for row in scores.chunks_exact_mut(tokens) {
                row.iter_mut().for_each(|score| *score *= scale);
                if self.fast {
                    badak_core::nn::softmax_fast_in_place(row);
                } else {
                    prob::softmax_in_place(row);
                }
            }
            // 출력[i] = Σⱼ 점수[i][j] × 값 j.
            simd::linear(&scores, tokens, &values, head, tokens, &mut out);
            for token in 0..tokens {
                mixed[token * width + index * head..][..head]
                    .copy_from_slice(&out[token * head..][..head]);
            }
        }
        let mixed = Tensor::from_vec(mixed, &[tokens, width])?;
        self.linear(&mixed, &block.proj)
    }
    // ANCHOR_END: attention

    // ANCHOR: forward
    /// 사진 한 장을 분류한다. `image`는 정규화한 `(3, 한 변, 한 변)`이다.
    pub fn forward(&self, image: &[f32]) -> Result<Trace> {
        // 1. 조각마다 벡터로.
        let patches = self.linear(&self.cut(image)?, &self.patch)?;

        // 2. 분류 토큰을 맨 앞에 끼우고 위치를 더한다.
        let mut tokens = self.class_token.to_vec::<f32>()?;
        tokens.extend_from_slice(patches.contiguous().as_slice::<f32>()?);
        let tokens = Tensor::from_vec(tokens, &[self.tokens(), self.width()])?;
        let embedded = ops::add(&tokens, &self.positions)?;

        // 3. 블록들. 정규화하고 어텐션, 정규화하고 MLP. 둘 다 건너뛰는 길에
        //    더한다(9장).
        let mut x = embedded.clone();
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let normed = ops::layer_norm(&x, &block.norm1.0, &block.norm1.1, EPS)?;
            x = ops::add(&x, &self.attention(block, &normed)?)?;
            let normed = ops::layer_norm(&x, &block.norm2.0, &block.norm2.1, EPS)?;
            let hidden = self.gelu(&self.linear(&normed, &block.fc1)?)?;
            x = ops::add(&x, &self.linear(&hidden, &block.fc2)?)?;
            outputs.push(x.clone());
        }

        // 4. 정규화하고, 분류 토큰 하나만 꺼내 점수로 바꾼다.
        let normed = ops::layer_norm(&x, &self.norm.0, &self.norm.1, EPS)?;
        let class = normed.slice(0, 0..1)?;
        let logits = self.linear(&class, &self.head)?;
        Ok(Trace {
            patches,
            embedded,
            blocks: outputs,
            normed,
            logits: logits.reshape(&[self.head.1.len()])?,
        })
    }
    // ANCHOR_END: forward
}
