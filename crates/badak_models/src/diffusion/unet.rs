//! 잡음을 짐작하는 U-Net (35장).
//!
//! 잡음이 섞인 잠재 그림 `(64, 64, 4)`, 단계 번호, 글의 벡터들을 받아
//! 그 그림에 섞인 잡음 `(64, 64, 4)`을 짐작한다. 내려가며 그림을 줄이고
//! 채널을 늘렸다가(64 → 32 → 16 → 8), 올라오며 되돌린다. 내려갈 때의
//! 결과를 올라올 때 같은 크기의 자리에 이어 붙인다.

use std::path::Path;

use badak_formats::Result;
use badak_formats::safetensors::SafeTensors;

use super::layers::{GroupNorm, Kernel, Map, activate, attend, invalid, layer_norm, pair};

/// 그룹 정규화의 묶음 수.
const GROUPS: usize = 32;

// ANCHOR: resnet
/// 잔차 블록. 정규화, SiLU, 합성곱을 두 번 하고 들어온 것에 더한다.
/// 그 사이에 단계 번호의 벡터를 채널마다 더한다.
pub(crate) struct Resnet {
    norm1: GroupNorm,
    conv1: Kernel,
    /// 단계의 벡터를 이 블록의 채널 수로 바꾸는 선형 층. VAE에는 없다.
    time: Option<Kernel>,
    norm2: GroupNorm,
    conv2: Kernel,
    /// 들어온 것과 나가는 것의 채널이 다를 때 쓰는 `1 × 1` 합성곱.
    shortcut: Option<Kernel>,
}

impl Resnet {
    pub(crate) fn load(file: &SafeTensors, name: &str, eps: f32) -> Result<Self> {
        let has = |part: &str| {
            let wanted = format!("{name}.{part}.weight");
            file.names().any(|candidate| candidate == wanted)
        };
        let optional = |part: &str| -> Result<Option<Kernel>> {
            has(part)
                .then(|| Kernel::load(file, &format!("{name}.{part}")))
                .transpose()
        };
        Ok(Self {
            norm1: GroupNorm::load(file, &format!("{name}.norm1"), GROUPS, eps)?,
            conv1: Kernel::load(file, &format!("{name}.conv1"))?,
            time: optional("time_emb_proj")?,
            norm2: GroupNorm::load(file, &format!("{name}.norm2"), GROUPS, eps)?,
            conv2: Kernel::load(file, &format!("{name}.conv2"))?,
            shortcut: optional("conv_shortcut")?,
        })
    }

    pub(crate) fn kernels(&mut self) -> Vec<&mut Kernel> {
        let mut kernels = vec![&mut self.conv1, &mut self.conv2];
        kernels.extend(self.time.as_mut());
        kernels.extend(self.shortcut.as_mut());
        kernels
    }

    /// `time`은 SiLU까지 지난 단계의 벡터다.
    pub(crate) fn forward(&self, x: &Map, time: Option<&[f32]>, fast: bool) -> Result<Map> {
        let mut hidden = self.conv1.conv(&self.norm1.apply(x, true, fast), 1)?;
        if let (Some(layer), Some(time)) = (&self.time, time) {
            let shift = layer.linear(time)?;
            for row in hidden.data.chunks_exact_mut(hidden.channels) {
                row.iter_mut()
                    .zip(&shift)
                    .for_each(|(value, s)| *value += s);
            }
        }
        let mut hidden = self.conv2.conv(&self.norm2.apply(&hidden, true, fast), 1)?;
        match &self.shortcut {
            Some(shortcut) => hidden.add(&shortcut.conv(x, 1)?),
            None => hidden.add(x),
        }
        Ok(hidden)
    }
}
// ANCHOR_END: resnet

/// 트랜스포머 블록 하나. 그림의 자리들이 토큰이다.
struct Block {
    norm1: (Vec<f32>, Vec<f32>),
    /// 셀프 어텐션: 그림의 자리들이 서로를 본다.
    own: [Kernel; 4],
    norm2: (Vec<f32>, Vec<f32>),
    /// 교차 어텐션: 그림의 자리들이 글의 토큰들을 본다.
    cross: [Kernel; 4],
    norm3: (Vec<f32>, Vec<f32>),
    fc1: Kernel,
    fc2: Kernel,
}

/// 어텐션 자리. 정규화하고, 트랜스포머 블록을 지나고, 들어온 것에 더한다.
struct Attention {
    norm: GroupNorm,
    proj_in: Kernel,
    block: Block,
    proj_out: Kernel,
}

impl Attention {
    fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        let block = format!("{name}.transformer_blocks.0");
        let four = |part: &str| -> Result<[Kernel; 4]> {
            Ok([
                Kernel::load(file, &format!("{block}.{part}.to_q"))?,
                Kernel::load(file, &format!("{block}.{part}.to_k"))?,
                Kernel::load(file, &format!("{block}.{part}.to_v"))?,
                Kernel::load(file, &format!("{block}.{part}.to_out.0"))?,
            ])
        };
        Ok(Self {
            norm: GroupNorm::load(file, &format!("{name}.norm"), GROUPS, 1e-6)?,
            proj_in: Kernel::load(file, &format!("{name}.proj_in"))?,
            block: Block {
                norm1: pair(file, &format!("{block}.norm1"))?,
                own: four("attn1")?,
                norm2: pair(file, &format!("{block}.norm2"))?,
                cross: four("attn2")?,
                norm3: pair(file, &format!("{block}.norm3"))?,
                fc1: Kernel::load(file, &format!("{block}.ff.net.0.proj"))?,
                fc2: Kernel::load(file, &format!("{block}.ff.net.2"))?,
            },
            proj_out: Kernel::load(file, &format!("{name}.proj_out"))?,
        })
    }

    fn kernels(&mut self) -> Vec<&mut Kernel> {
        let block = &mut self.block;
        let mut kernels = vec![
            &mut self.proj_in,
            &mut self.proj_out,
            &mut block.fc1,
            &mut block.fc2,
        ];
        kernels.extend(block.own.iter_mut());
        kernels.extend(block.cross.iter_mut());
        kernels
    }

    // ANCHOR: transformer
    /// `context`는 글의 벡터들 `(77, 768)`이다.
    fn forward(&self, x: &Map, context: &[f32], heads: usize, fast: bool) -> Result<Map> {
        let width = x.channels;
        let block = &self.block;
        let mut tokens = self.proj_in.linear(&self.norm.apply(x, false, fast).data)?;
        let eps = 1e-5;

        // 1. 셀프 어텐션. 키와 값이 그림 자신이다.
        let mut normed = tokens.clone();
        layer_norm(&mut normed, &block.norm1.0, &block.norm1.1, eps);
        let [query, key, value, out] = &block.own;
        let mixed = attend(
            &query.linear(&normed)?,
            &key.linear(&normed)?,
            &value.linear(&normed)?,
            width,
            heads,
            fast,
        );
        let mixed = out.linear(&mixed)?;
        tokens.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);

        // 2. 교차 어텐션. 키와 값이 글이다. 글이 그림에 들어오는 유일한
        //    길이다.
        let mut normed = tokens.clone();
        layer_norm(&mut normed, &block.norm2.0, &block.norm2.1, eps);
        let [query, key, value, out] = &block.cross;
        let mixed = attend(
            &query.linear(&normed)?,
            &key.linear(context)?,
            &value.linear(context)?,
            width,
            heads,
            fast,
        );
        let mixed = out.linear(&mixed)?;
        tokens.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);

        // 3. MLP. 첫 층이 폭의 여덟 배를 내고, 반은 값, 반은 GELU를 지나
        //    그 값에 곱해지는 문이 된다(GEGLU).
        let mut normed = tokens.clone();
        layer_norm(&mut normed, &block.norm3.0, &block.norm3.1, eps);
        let wide = block.fc1.linear(&normed)?;
        let half = block.fc2.inputs;
        let mut gated = Vec::with_capacity(wide.len() / 2);
        for row in wide.chunks_exact(2 * half) {
            let (values, gates) = row.split_at(half);
            let start = gated.len();
            gated.extend_from_slice(gates);
            if fast {
                badak_core::nn::gelu_in_place(&mut gated[start..]);
            } else {
                gated[start..]
                    .iter_mut()
                    .for_each(|gate| *gate = badak_core::nn::gelu(*gate));
            }
            gated[start..]
                .iter_mut()
                .zip(values)
                .for_each(|(gate, value)| *gate *= value);
        }
        let hidden = block.fc2.linear(&gated)?;
        tokens.iter_mut().zip(&hidden).for_each(|(a, b)| *a += b);

        let mut out = Map::new(x.height, x.width, width, self.proj_out.linear(&tokens)?);
        out.add(x);
        Ok(out)
    }
    // ANCHOR_END: transformer
}

/// 내려가거나 올라가는 한 단.
struct Stage {
    resnets: Vec<Resnet>,
    /// 잔차 블록마다 하나씩. 어텐션이 없는 단은 비어 있다.
    attentions: Vec<Attention>,
    /// 크기를 바꾸는 합성곱. 마지막 단에는 없다.
    resize: Option<Kernel>,
}

impl Stage {
    fn load(file: &SafeTensors, name: &str, resize: &str) -> Result<Self> {
        let has = |key: String| file.names().any(|candidate| candidate == key);
        let mut resnets = Vec::new();
        while has(format!("{name}.resnets.{}.conv1.weight", resnets.len())) {
            let resnet = format!("{name}.resnets.{}", resnets.len());
            resnets.push(Resnet::load(file, &resnet, 1e-5)?);
        }
        let mut attentions = Vec::new();
        while has(format!(
            "{name}.attentions.{}.proj_in.weight",
            attentions.len()
        )) {
            let attention = format!("{name}.attentions.{}", attentions.len());
            attentions.push(Attention::load(file, &attention)?);
        }
        let resize = format!("{name}.{resize}.0.conv");
        let resize = has(format!("{resize}.weight"))
            .then(|| Kernel::load(file, &resize))
            .transpose()?;
        Ok(Self {
            resnets,
            attentions,
            resize,
        })
    }

    fn kernels(&mut self) -> Vec<&mut Kernel> {
        let mut kernels: Vec<&mut Kernel> = Vec::new();
        for resnet in &mut self.resnets {
            kernels.extend(resnet.kernels());
        }
        for attention in &mut self.attentions {
            kernels.extend(attention.kernels());
        }
        kernels.extend(self.resize.as_mut());
        kernels
    }
}

// ANCHOR: unet
/// U-Net.
pub struct Unet {
    conv_in: Kernel,
    time1: Kernel,
    time2: Kernel,
    down: Vec<Stage>,
    middle: Stage,
    up: Vec<Stage>,
    norm_out: GroupNorm,
    conv_out: Kernel,
    heads: usize,
    fast: bool,
}
// ANCHOR_END: unet

impl Unet {
    /// safetensors 파일에서 읽는다. `heads`는 어텐션 헤드의 수다.
    pub fn load(path: impl AsRef<Path>, heads: usize) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let has = |key: String| file.names().any(|candidate| candidate == key);
        let mut down = Vec::new();
        while has(format!("down_blocks.{}.resnets.0.conv1.weight", down.len())) {
            let name = format!("down_blocks.{}", down.len());
            down.push(Stage::load(&file, &name, "downsamplers")?);
        }
        let mut up = Vec::new();
        while has(format!("up_blocks.{}.resnets.0.conv1.weight", up.len())) {
            let name = format!("up_blocks.{}", up.len());
            up.push(Stage::load(&file, &name, "upsamplers")?);
        }
        if down.is_empty() || down.len() != up.len() {
            return Err(invalid("내려가는 단과 올라가는 단의 수가 다릅니다".into()));
        }
        Ok(Self {
            conv_in: Kernel::load(&file, "conv_in")?,
            time1: Kernel::load(&file, "time_embedding.linear_1")?,
            time2: Kernel::load(&file, "time_embedding.linear_2")?,
            down,
            middle: Stage::load(&file, "mid_block", "none")?,
            up,
            norm_out: GroupNorm::load(&file, "conv_norm_out", GROUPS, 1e-5)?,
            conv_out: Kernel::load(&file, "conv_out")?,
            heads,
            fast: false,
        })
    }

    fn kernels(&mut self) -> Vec<&mut Kernel> {
        let mut kernels = vec![
            &mut self.conv_in,
            &mut self.time1,
            &mut self.time2,
            &mut self.conv_out,
        ];
        for stage in self.down.iter_mut().chain(&mut self.up) {
            kernels.extend(stage.kernels());
        }
        kernels.extend(self.middle.kernels());
        kernels
    }

    /// 빠른 길을 켠다. 모든 선형 층과 합성곱을 31장의 SME 커널로, 지수
    /// 함수를 근사로 계산한다. 가중치를 다시 늘어놓으므로 메모리를 그만큼
    /// 더 쓴다.
    pub fn fast(mut self) -> Result<Self> {
        self.fast = true;
        for kernel in self.kernels() {
            kernel.pack()?;
        }
        Ok(self)
    }

    /// 합성곱과 선형 층의 가중치의 수. 정규화의 가중치는 세지 않는다.
    pub fn parameters(&mut self) -> usize {
        self.kernels()
            .iter()
            .map(|kernel| kernel.parameters())
            .sum()
    }

    // ANCHOR: time
    /// 단계 번호를 벡터로 바꾼다. 9장의 "자리를 사인과 코사인으로 적기"와
    /// 같은 방법으로 320개의 값을 만들고, 선형 층 둘을 지난다.
    fn time(&self, step: usize) -> Result<Vec<f32>> {
        let half = self.time1.inputs / 2;
        let mut waves = vec![0.0; 2 * half];
        for index in 0..half {
            let frequency = (-(10000.0_f64.ln()) * index as f64 / half as f64).exp();
            let angle = step as f64 * frequency;
            waves[index] = angle.cos() as f32;
            waves[half + index] = angle.sin() as f32;
        }
        let mut hidden = self.time1.linear(&waves)?;
        activate(&mut hidden, false);
        let mut time = self.time2.linear(&hidden)?;
        // 잔차 블록들은 이 벡터에 SiLU를 적용한 것을 쓴다. 미리 해 둔다.
        activate(&mut time, false);
        Ok(time)
    }
    // ANCHOR_END: time

    // ANCHOR: forward
    /// 잡음을 짐작한다. `sample`은 잡음이 섞인 잠재 그림, `step`은 단계
    /// 번호, `context`는 글의 벡터들이다.
    pub fn forward(&self, sample: &Map, step: usize, context: &[f32]) -> Result<Map> {
        let (heads, fast) = (self.heads, self.fast);
        let time = self.time(step)?;
        let mut x = self.conv_in.conv(sample, 1)?;

        // 내려간다. 지나온 결과를 모두 쌓아 둔다.
        let mut skips = vec![x.clone()];
        for stage in &self.down {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                x = resnet.forward(&x, Some(&time), fast)?;
                if let Some(attention) = stage.attentions.get(index) {
                    x = attention.forward(&x, context, heads, fast)?;
                }
                skips.push(x.clone());
            }
            if let Some(resize) = &stage.resize {
                x = resize.conv(&x, 2)?;
                skips.push(x.clone());
            }
        }

        // 가장 작은 크기(8 × 8)에서.
        x = self.middle.resnets[0].forward(&x, Some(&time), fast)?;
        x = self.middle.attentions[0].forward(&x, context, heads, fast)?;
        x = self.middle.resnets[1].forward(&x, Some(&time), fast)?;

        // 올라온다. 쌓아 둔 것을 뒤에서부터 꺼내 채널 방향으로 붙인다.
        for stage in &self.up {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                let skip = skips
                    .pop()
                    .ok_or_else(|| invalid("쌓아 둔 결과가 모자랍니다".into()))?;
                x = resnet.forward(&x.concat(&skip), Some(&time), fast)?;
                if let Some(attention) = stage.attentions.get(index) {
                    x = attention.forward(&x, context, heads, fast)?;
                }
            }
            if let Some(resize) = &stage.resize {
                x = resize.conv(&x.upsample(), 1)?;
            }
        }
        self.conv_out.conv(&self.norm_out.apply(&x, true, fast), 1)
    }
    // ANCHOR_END: forward
}
