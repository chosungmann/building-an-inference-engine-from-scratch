//! 표본 속도 바꾸기 (37장).
//!
//! 음성 합성 모델은 24kHz의 소리를 내고 33장의 Whisper는 16kHz만 받는다.
//! 만든 소리를 받아써 보려면 표본 속도를 바꿔야 한다.

use std::f64::consts::PI;

/// 한쪽으로 보는 표본의 수(낮추는 비율이 1일 때).
const TAPS: f64 = 8.0;

// ANCHOR: resample
/// 표본 속도를 `from`에서 `to`로 바꾼다.
///
/// 새 표본 하나는 옛 표본들 사이의 어느 자리에 놓인다. 그 자리의 값을
/// 둘레의 옛 표본들을 `sinc` 모양의 비중으로 섞어 만든다. 36장의 란초스
/// 필터와 같은 생각이고(거기서는 픽셀, 여기서는 표본), 비중을 매끄럽게
/// 0으로 줄이는 창으로 한 창을 쓴다.
///
/// 속도를 낮출 때는 새 속도의 절반보다 높은 떨림을 먼저 지워야 한다.
/// 지우지 않으면 그 떨림이 낮은 떨림으로 둔갑해 섞여 든다. `sinc`를
/// 낮추는 비율만큼 넓히면 그 일을 함께 한다.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let step = f64::from(from) / f64::from(to);
    // 낮출 때만 넓힌다.
    let stretch = step.max(1.0);
    let reach = TAPS * stretch;
    let count = (samples.len() as f64 / step).floor() as usize;
    (0..count)
        .map(|index| {
            let center = index as f64 * step;
            let first = (center - reach).ceil().max(0.0) as usize;
            let last = ((center + reach).floor() as usize).min(samples.len() - 1);
            let (mut sum, mut total) = (0.0, 0.0);
            for (source, &sample) in samples.iter().enumerate().take(last + 1).skip(first) {
                let distance = (source as f64 - center) / stretch;
                let sinc = if distance == 0.0 {
                    1.0
                } else {
                    (PI * distance).sin() / (PI * distance)
                };
                let window = 0.5 + 0.5 * (PI * distance / TAPS).cos();
                sum += f64::from(sample) * sinc * window;
                total += sinc * window;
            }
            // 비중의 합으로 나눠 가장자리에서도 크기가 유지되게 한다.
            (sum / total) as f32
        })
        .collect()
}
// ANCHOR_END: resample

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hertz: f64, rate: u32, count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| (2.0 * PI * hertz * index as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn loudest(samples: &[f32]) -> f32 {
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        middle.iter().fold(0.0, |peak, value| peak.max(value.abs()))
    }

    #[test]
    fn a_low_tone_survives_and_keeps_its_pitch() {
        // 440Hz를 24kHz에서 16kHz로. 길이는 3분의 2, 값은 16kHz로 만든 것과 같다.
        let resampled = resample(&tone(440.0, 24_000, 2400), 24_000, 16_000);
        assert_eq!(resampled.len(), 1600);
        let direct = tone(440.0, 16_000, 1600);
        for index in 200..1400 {
            assert!((resampled[index] - direct[index]).abs() < 2e-3, "{index}");
        }
    }

    #[test]
    fn a_tone_above_the_new_limit_is_removed() {
        // 11kHz는 16kHz의 절반(8kHz)보다 높다. 남으면 5kHz로 둔갑한다.
        let resampled = resample(&tone(11_000.0, 24_000, 2400), 24_000, 16_000);
        assert!(loudest(&resampled) < 0.02, "{}", loudest(&resampled));
    }

    #[test]
    fn the_same_rate_and_empty_input_pass_through() {
        assert_eq!(resample(&[0.5, -0.5], 16_000, 16_000), [0.5, -0.5]);
        assert!(resample(&[], 24_000, 16_000).is_empty());
    }
}
