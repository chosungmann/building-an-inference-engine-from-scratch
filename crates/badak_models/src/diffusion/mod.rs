//! 글에서 그림을 만드는 확산 모델 (35장).
//!
//! Stable Diffusion 1.5의 세 부품과 그것들을 잇는 반복문이다.
//!
//! 1. [`TextEncoder`]: 글을 벡터 77개로 바꾼다. 한 번 실행한다.
//! 2. [`Unet`]: 잡음이 섞인 잠재 그림에서 잡음을 짐작한다. 단계마다 두
//!    번 실행한다(글을 주고 한 번, 주지 않고 한 번).
//! 3. [`Decoder`]: 다 걷어 낸 잠재 그림을 픽셀로 푼다. 한 번 실행한다.

pub mod clip;
pub mod layers;
pub mod scheduler;
pub mod unet;
pub mod vae;

pub use clip::TextEncoder;
pub use layers::Map;
pub use scheduler::{Schedule, ddim_step, guide};
pub use unet::Unet;
pub use vae::Decoder;

use badak_formats::Result;

// ANCHOR: settings
/// 그림 한 장을 만드는 설정.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// 잡음을 걷어 내는 스텝의 수.
    pub steps: usize,
    /// 가이던스의 세기. 1 이하이면 글 없이 짐작하는 실행을 건너뛴다.
    pub guidance: f32,
}
// ANCHOR_END: settings

// ANCHOR: denoise
/// 잡음 `latent`에서 시작해 스텝마다 잡음을 걷어 낸다. `progress`는
/// 스텝이 끝날 때마다 (스텝 번호, 단계 번호, 그때의 잠재 그림)으로 불린다.
///
/// `context`는 글의 벡터들, `empty`는 빈 글의 벡터들이다.
pub fn denoise(
    unet: &Unet,
    mut latent: Map,
    context: &[f32],
    empty: &[f32],
    settings: Settings,
    mut progress: impl FnMut(usize, usize, &Map),
) -> Result<Map> {
    let schedule = Schedule::stable_diffusion();
    let timesteps = schedule.timesteps(settings.steps, 1);
    for (index, &step) in timesteps.iter().enumerate() {
        // 1. 잡음을 짐작한다. 글을 주고 한 번, 주지 않고 한 번.
        let conditional = unet.forward(&latent, step, context)?;
        let noise = if settings.guidance > 1.0 {
            let unconditional = unet.forward(&latent, step, empty)?;
            guide(&unconditional.data, &conditional.data, settings.guidance)
        } else {
            conditional.data
        };
        // 2. 그만큼 걷어 내고 다음 단계로 간다.
        let previous = timesteps.get(index + 1).copied();
        // 이 모델의 설정(`set_alpha_to_one`이 거짓)에서는 맨 끝 스텝도
        // 단계 0의 ᾱ를 쓴다.
        let final_alpha = schedule.alphas[0];
        ddim_step(
            &schedule,
            step,
            previous,
            &mut latent.data,
            &noise,
            final_alpha,
        );
        progress(index, step, &latent);
    }
    Ok(latent)
}
// ANCHOR_END: denoise

/// 디코더가 낸 값(대략 −1에서 1)을 0에서 255의 바이트로 바꾼다.
pub fn to_bytes(image: &Map) -> Vec<u8> {
    image
        .data
        .iter()
        .map(|&value| ((value / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

// ANCHOR: noise
/// 정규 분포를 따르는 잡음 `count`개. 시작하는 잠재 그림이다.
///
/// 12장의 난수 생성기가 내는 고른 난수 둘을 정규 분포의 값 둘로 바꾼다
/// (박스-뮬러 변환). 번호가 같으면 늘 같은 잡음이 나온다. PyTorch의
/// 생성기와는 다른 수열이므로, 같은 번호로 같은 그림이 나오지는 않는다.
pub fn noise(seed: u64, count: usize) -> Vec<f32> {
    let mut random = badak_core::random::Pcg64::new(seed);
    let mut values = Vec::with_capacity(count + 1);
    while values.len() < count {
        // 로그에 0이 들어가지 않게 (0, 1]로 옮긴다.
        let radius = (-2.0 * (1.0 - random.next_f64()).ln()).sqrt();
        let angle = std::f64::consts::TAU * random.next_f64();
        values.push((radius * angle.cos()) as f32);
        values.push((radius * angle.sin()) as f32);
    }
    values.truncate(count);
    values
}
// ANCHOR_END: noise

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_reproducible_and_roughly_standard_normal() {
        let values = noise(7, 16_384);
        assert_eq!(values, noise(7, 16_384));
        assert_ne!(values[..8], noise(8, 8)[..]);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
        assert!(mean.abs() < 0.03, "{mean}");
        assert!((variance - 1.0).abs() < 0.05, "{variance}");
        assert_eq!(noise(1, 3).len(), 3);
    }

    #[test]
    fn bytes_are_clamped_and_rounded() {
        let image = Map::new(1, 1, 4, vec![-3.0, -1.0, 0.0, 1.5]);
        assert_eq!(to_bytes(&image), [0, 0, 128, 255]);
    }
}
