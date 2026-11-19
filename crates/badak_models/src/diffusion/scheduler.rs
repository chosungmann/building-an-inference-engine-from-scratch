//! 잡음을 걷어 내는 순서 (35장).
//!
//! 모델은 "이 그림에 섞인 잡음이 무엇인가"만 답한다. 그 답으로 그림을
//! 얼마나 고칠지, 어느 단계들을 밟을지는 스케줄러가 정한다.

// ANCHOR: schedule
/// 학습할 때 쓴 잡음의 일정. 단계 `t`(0부터 999)마다, 원래 그림이 얼마나
/// 남아 있는지(`alphas[t]`, ᾱ)를 적어 둔 표다. 0에서 1에 가깝고(거의
/// 그림) 999에서 0에 가깝다(거의 잡음).
pub struct Schedule {
    /// 단계마다의 ᾱ.
    pub alphas: Vec<f64>,
}

impl Schedule {
    /// Stable Diffusion 1.x의 일정. β의 제곱근이 `0.00085`의 제곱근에서
    /// `0.012`의 제곱근까지 고르게 늘어난다(`scaled_linear`).
    pub fn stable_diffusion() -> Self {
        let (steps, first, last) = (1000, 0.00085_f64.sqrt(), 0.012_f64.sqrt());
        let mut product = 1.0;
        let alphas = (0..steps)
            .map(|step| {
                let root = first + (last - first) * step as f64 / (steps - 1) as f64;
                product *= 1.0 - root * root;
                product
            })
            .collect();
        Self { alphas }
    }

    /// 학습할 때의 단계 수.
    pub fn len(&self) -> usize {
        self.alphas.len()
    }

    /// 비었는가.
    pub fn is_empty(&self) -> bool {
        self.alphas.is_empty()
    }

    /// 1,000단계 가운데 `count`개를 고르게 골라 큰 것부터 돌려준다.
    /// `offset`은 고른 단계들에 더하는 값이다(이 모델의 설정은 1).
    pub fn timesteps(&self, count: usize, offset: usize) -> Vec<usize> {
        let stride = self.len() / count.max(1);
        (0..count)
            .rev()
            .map(|index| index * stride + offset)
            .collect()
    }
}
// ANCHOR_END: schedule

// ANCHOR: ddim
/// DDIM의 한 걸음. 단계 `t`의 그림 `sample`과 모델이 짐작한 잡음
/// `noise`로, 단계 `previous`의 그림을 만든다(`previous`가 `None`이면
/// 맨 끝, 잡음이 없는 그림).
///
/// 1. 잡음을 모두 걷어 낸 그림을 짐작한다.
///    `x₀ = (x − √(1 − ᾱₜ) · ε) / √ᾱₜ`
/// 2. 그 그림에 다음 단계만큼의 잡음을 **같은 방향으로** 다시 섞는다.
///    `x' = √ᾱ' · x₀ + √(1 − ᾱ') · ε`
///
/// 무작위가 없다. 같은 잡음에서 시작하면 늘 같은 그림이 나온다.
pub fn ddim_step(
    schedule: &Schedule,
    t: usize,
    previous: Option<usize>,
    sample: &mut [f32],
    noise: &[f32],
    final_alpha: f64,
) {
    let alpha = schedule.alphas[t];
    let next = previous.map_or(final_alpha, |step| schedule.alphas[step]);
    let (keep, mix) = (alpha.sqrt(), (1.0 - alpha).sqrt());
    let (next_keep, next_mix) = (next.sqrt(), (1.0 - next).sqrt());
    for (x, &epsilon) in sample.iter_mut().zip(noise) {
        let (value, epsilon) = (f64::from(*x), f64::from(epsilon));
        let clean = (value - mix * epsilon) / keep;
        *x = (next_keep * clean + next_mix * epsilon) as f32;
    }
}
// ANCHOR_END: ddim

// ANCHOR: guidance
/// 분류기 없는 가이던스. 글 없이 짐작한 잡음에서 글을 주고 짐작한 잡음
/// 쪽으로 `scale`배만큼 간다. 1이면 글을 준 짐작 그대로다.
pub fn guide(unconditional: &[f32], conditional: &[f32], scale: f32) -> Vec<f32> {
    unconditional
        .iter()
        .zip(conditional)
        .map(|(u, c)| u + scale * (c - u))
        .collect()
}
// ANCHOR_END: guidance

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_runs_from_image_to_noise() {
        let schedule = Schedule::stable_diffusion();
        assert_eq!(schedule.len(), 1000);
        assert!((schedule.alphas[0] - 0.99915).abs() < 1e-9);
        assert!(schedule.alphas[999] < 0.005 && schedule.alphas[999] > 0.004);
        assert!(schedule.alphas.windows(2).all(|pair| pair[1] < pair[0]));
    }

    #[test]
    fn timesteps_are_evenly_spaced_and_descending() {
        let schedule = Schedule::stable_diffusion();
        assert_eq!(schedule.timesteps(4, 1), [751, 501, 251, 1]);
        assert_eq!(schedule.timesteps(20, 1)[..3], [951, 901, 851]);
    }

    #[test]
    fn a_perfect_guess_recovers_the_image_in_one_step() {
        // 그림 x₀에 잡음 ε을 단계 500만큼 섞고, 그 ε을 정확히 알려 주면
        // 마지막 걸음(ᾱ' = 1)이 x₀을 돌려준다.
        let schedule = Schedule::stable_diffusion();
        let (clean, epsilon) = ([0.5_f32, -1.0], [0.3_f32, 2.0]);
        let alpha = schedule.alphas[500];
        let mut sample: Vec<f32> = clean
            .iter()
            .zip(&epsilon)
            .map(|(&x, &e)| {
                (alpha.sqrt() * f64::from(x) + (1.0 - alpha).sqrt() * f64::from(e)) as f32
            })
            .collect();
        ddim_step(&schedule, 500, None, &mut sample, &epsilon, 1.0);
        for (value, wanted) in sample.iter().zip(clean) {
            assert!((value - wanted).abs() < 1e-5);
        }
    }

    #[test]
    fn guidance_extrapolates_past_the_conditional_guess() {
        assert_eq!(guide(&[1.0], &[2.0], 1.0), [2.0]);
        assert_eq!(guide(&[1.0], &[2.0], 0.0), [1.0]);
        assert_eq!(guide(&[1.0], &[2.0], 7.5), [8.5]);
    }
}
