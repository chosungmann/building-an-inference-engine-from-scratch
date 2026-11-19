//! 잠재 그림을 픽셀로 푸는 VAE 디코더 (35장).
//!
//! U-Net은 `512 × 512`의 그림이 아니라 `64 × 64 × 4`의 잠재 그림에서
//! 일한다. 가로와 세로가 8분의 1이다. 다 걷어 낸 잠재 그림을 이 디코더가
//! 한 번에 `512 × 512 × 3`의 픽셀로 푼다.

use std::path::Path;

use badak_formats::Result;
use badak_formats::safetensors::SafeTensors;

use super::layers::{GroupNorm, Kernel, Map, attend};
use super::unet::Resnet;

/// 잠재 그림에 곱해져 있는 값. 디코더에 넣기 전에 이 값으로 나눈다.
pub const SCALE: f32 = 0.18215;
const EPS: f32 = 1e-6;

// ANCHOR: vae
/// VAE의 디코더.
pub struct Decoder {
    post_quant: Kernel,
    conv_in: Kernel,
    middle: [Resnet; 2],
    /// 가장 작은 크기에서 한 번 하는 어텐션. 헤드는 하나다.
    norm: GroupNorm,
    attention: [Kernel; 4],
    /// 단마다 잔차 블록들과, 크기를 두 배로 늘린 뒤의 합성곱(마지막 단에는
    /// 없다).
    up: Vec<(Vec<Resnet>, Option<Kernel>)>,
    norm_out: GroupNorm,
    conv_out: Kernel,
    fast: bool,
}
// ANCHOR_END: vae

impl Decoder {
    /// safetensors 파일에서 디코더 쪽만 읽는다.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = SafeTensors::open(path)?;
        let has = |key: String| file.names().any(|candidate| candidate == key);
        let resnet = |name: String| Resnet::load(&file, &name, EPS);
        let mut up = Vec::new();
        while has(format!(
            "decoder.up_blocks.{}.resnets.0.conv1.weight",
            up.len()
        )) {
            let name = format!("decoder.up_blocks.{}", up.len());
            let mut resnets = Vec::new();
            while has(format!("{name}.resnets.{}.conv1.weight", resnets.len())) {
                resnets.push(resnet(format!("{name}.resnets.{}", resnets.len()))?);
            }
            let resize = format!("{name}.upsamplers.0.conv");
            let resize = has(format!("{resize}.weight"))
                .then(|| Kernel::load(&file, &resize))
                .transpose()?;
            up.push((resnets, resize));
        }
        let attention = "decoder.mid_block.attentions.0";
        Ok(Self {
            post_quant: Kernel::load(&file, "post_quant_conv")?,
            conv_in: Kernel::load(&file, "decoder.conv_in")?,
            middle: [
                resnet("decoder.mid_block.resnets.0".into())?,
                resnet("decoder.mid_block.resnets.1".into())?,
            ],
            norm: GroupNorm::load(&file, &format!("{attention}.group_norm"), 32, EPS)?,
            attention: [
                Kernel::load(&file, &format!("{attention}.query"))?,
                Kernel::load(&file, &format!("{attention}.key"))?,
                Kernel::load(&file, &format!("{attention}.value"))?,
                Kernel::load(&file, &format!("{attention}.proj_attn"))?,
            ],
            up,
            norm_out: GroupNorm::load(&file, "decoder.conv_norm_out", 32, EPS)?,
            conv_out: Kernel::load(&file, "decoder.conv_out")?,
            fast: false,
        })
    }

    /// 빠른 길을 켠다.
    pub fn fast(mut self) -> Result<Self> {
        self.fast = true;
        let mut kernels = vec![&mut self.post_quant, &mut self.conv_in, &mut self.conv_out];
        kernels.extend(self.attention.iter_mut());
        for resnet in &mut self.middle {
            kernels.extend(resnet.kernels());
        }
        for (resnets, resize) in &mut self.up {
            for resnet in resnets {
                kernels.extend(resnet.kernels());
            }
            kernels.extend(resize.as_mut());
        }
        for kernel in kernels {
            kernel.pack()?;
        }
        Ok(self)
    }

    // ANCHOR: decode
    /// 잠재 그림 `(h, w, 4)`를 픽셀 `(8h, 8w, 3)`로 푼다. 값은 대략 −1에서
    /// 1 사이다.
    pub fn decode(&self, latent: &Map) -> Result<Map> {
        let fast = self.fast;
        let mut scaled = latent.clone();
        scaled.data.iter_mut().for_each(|value| *value /= SCALE);
        let mut x = self.conv_in.conv(&self.post_quant.conv(&scaled, 1)?, 1)?;

        x = self.middle[0].forward(&x, None, fast)?;
        let [query, key, value, out] = &self.attention;
        let normed = self.norm.apply(&x, false, fast).data;
        let mixed = attend(
            &query.linear(&normed)?,
            &key.linear(&normed)?,
            &value.linear(&normed)?,
            x.channels,
            1,
            fast,
        );
        let mixed = Map::new(x.height, x.width, x.channels, out.linear(&mixed)?);
        x.add(&mixed);
        x = self.middle[1].forward(&x, None, fast)?;

        for (resnets, resize) in &self.up {
            for resnet in resnets {
                x = resnet.forward(&x, None, fast)?;
            }
            if let Some(resize) = resize {
                x = resize.conv(&x.upsample(), 1)?;
            }
        }
        self.conv_out.conv(&self.norm_out.apply(&x, true, fast), 1)
    }
    // ANCHOR_END: decode
}
