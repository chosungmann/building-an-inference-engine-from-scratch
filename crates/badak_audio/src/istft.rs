//! 스펙트럼에서 소리로: 거꾸로 가는 짧은 시간 푸리에 변환 (37장).
//!
//! 33장은 소리를 토막 내어 토막마다 푸리에 변환을 했다. 여기서는 그
//! 반대다. 토막마다의 스펙트럼을 받아 토막마다의 표본으로 되돌리고,
//! 겹치는 토막들을 더해 소리 한 줄로 만든다.

use crate::fft::{Bluestein, Complex, Split};
use crate::mel::hann;

// ANCHOR: istft
/// 거꾸로 가는 짧은 시간 푸리에 변환.
pub struct Istft {
    /// 토막의 길이(표본).
    size: usize,
    /// 토막 사이의 간격(표본).
    hop: usize,
    window: Vec<f64>,
    /// 길이가 `작은 홀수 × 2의 거듭제곱`이면 줄로 갈라 변환한다.
    split: Option<Split>,
    transform: Bluestein,
}

impl Istft {
    /// 토막의 길이와 간격을 정한다.
    pub fn new(size: usize, hop: usize) -> Self {
        Self {
            size,
            hop,
            window: hann(size),
            split: Split::new(size, 7),
            transform: Bluestein::new(size),
        }
    }

    /// 주파수의 수. 실수 신호의 스펙트럼은 절반만 있으면 된다.
    pub fn bins(&self) -> usize {
        self.size / 2 + 1
    }

    /// 토막 하나의 스펙트럼을 창을 곱한 표본 `size`개로 바꿔 `out`에 적는다.
    ///
    /// `row`의 앞 절반은 주파수마다의 크기의 로그, 뒤 절반은 위상이다.
    /// 크기는 100에서 자른다(모델이 터무니없이 큰 값을 냈을 때의 안전장치다).
    ///
    /// 거꾸로 가는 변환은 앞으로 가는 변환으로 계산한다. 스펙트럼의
    /// 켤레를 변환하고 다시 켤레를 취해 길이로 나누면 된다. 실수 신호의
    /// 스펙트럼은 뒤 절반이 앞 절반의 켤레이므로 그렇게 채운다.
    pub fn frame(&self, row: &[f32], out: &mut [f64], scratch: &mut Vec<Complex>) {
        let bins = self.bins();
        let mut values = vec![Complex::default(); self.size];
        for bin in 0..bins {
            let magnitude = f64::from(row[bin].exp().min(100.0));
            let phase = f64::from(row[bins + bin]);
            // 켤레를 넣는다: 허수부의 부호를 뒤집는다.
            let value = Complex {
                re: magnitude * phase.cos(),
                im: -magnitude * phase.sin(),
            };
            values[bin] = value;
            if bin > 0 && bin < bins - 1 {
                values[self.size - bin] = value.conjugate();
            }
        }
        match &self.split {
            Some(split) => split.transform(&mut values, scratch),
            None => self.transform.transform(&mut values, scratch),
        }
        for ((sample, value), window) in out.iter_mut().zip(&values).zip(&self.window) {
            // 결과의 켤레의 실수부는 실수부 그대로다.
            *sample = value.re / self.size as f64 * window;
        }
    }

    /// 토막들의 스펙트럼을 소리로 바꾼다. `rows`는 토막마다 `2 × bins`개다.
    ///
    /// 토막들을 `hop`만큼 어긋나게 겹쳐 더하고, 자리마다 겹친 창의 제곱의
    /// 합으로 나눈다(기준 구현의 규칙이다. 토막에 창을 한 번만 곱했으므로
    /// 소리의 크기에 일정한 배수가 남는데, 모델이 그 규칙으로 학습되었다). 앞뒤의 `(size − hop) / 2`개는 잘라 낸다. 그러면 토막
    /// 하나가 정확히 표본 `hop`개가 된다.
    pub fn synthesize(&self, rows: &[f32]) -> Vec<f32> {
        let width = 2 * self.bins();
        let frames = rows.len() / width;
        if frames == 0 {
            return Vec::new();
        }
        let length = (frames - 1) * self.hop + self.size;
        let (mut signal, mut envelope) = (vec![0.0_f64; length], vec![0.0_f64; length]);
        let mut frame = vec![0.0_f64; self.size];
        let mut scratch = Vec::new();
        for (index, row) in rows.chunks_exact(width).enumerate() {
            self.frame(row, &mut frame, &mut scratch);
            let start = index * self.hop;
            for sample in 0..self.size {
                signal[start + sample] += frame[sample];
                envelope[start + sample] += self.window[sample] * self.window[sample];
            }
        }
        let pad = (self.size - self.hop) / 2;
        (pad..length - pad)
            .map(|sample| (signal[sample] / envelope[sample]) as f32)
            .collect()
    }
}
// ANCHOR_END: istft

#[cfg(test)]
mod tests {
    use super::*;

    /// 정의대로 계산한 토막 하나. 느리지만 틀릴 곳이 없다.
    fn slow_frame(row: &[f32], size: usize) -> Vec<f64> {
        let bins = size / 2 + 1;
        let window = hann(size);
        (0..size)
            .map(|sample| {
                let mut sum = 0.0;
                for bin in 0..bins {
                    let magnitude = f64::from(row[bin].exp().min(100.0));
                    let angle = std::f64::consts::TAU * (bin * sample) as f64 / size as f64
                        + f64::from(row[bins + bin]);
                    let weight = if bin == 0 || bin == bins - 1 {
                        1.0
                    } else {
                        2.0
                    };
                    sum += weight * magnitude * angle.cos();
                }
                sum / size as f64 * window[sample]
            })
            .collect()
    }

    fn spectrum(frames: usize, bins: usize) -> Vec<f32> {
        (0..frames * 2 * bins)
            .map(|index| ((index * 37 % 101) as f32 / 50.0 - 1.0) * 1.5)
            .collect()
    }

    #[test]
    fn the_fast_frame_matches_the_definition() {
        // 2의 거듭제곱이 아닌 길이.
        let istft = Istft::new(20, 5);
        let rows = spectrum(1, istft.bins());
        let (mut fast, mut scratch) = (vec![0.0; 20], Vec::new());
        istft.frame(&rows, &mut fast, &mut scratch);
        for (a, b) in fast.iter().zip(slow_frame(&rows, 20)) {
            assert!((a - b).abs() < 1e-9, "{a} {b}");
        }
    }

    #[test]
    fn every_frame_becomes_hop_samples() {
        let istft = Istft::new(16, 4);
        let rows = spectrum(7, istft.bins());
        assert_eq!(istft.synthesize(&rows).len(), 7 * 4);
        assert!(istft.synthesize(&[]).is_empty());
    }

    #[test]
    fn a_single_tone_comes_back_as_a_sine_wave() {
        // 3번 주파수만 크기 1. 가운데의 표본들은 cos(2π·3·t/16)의 꼴이어야
        // 한다. 크기에는 4/3이 곱해진다. 토막에는 창을 한 번 곱했는데 나눌
        // 때는 창의 **제곱**의 합으로 나누기 때문이다(간격이 길이의 4분의
        // 1인 한 창에서 창의 합은 2, 제곱의 합은 1.5다). 이 디코더가 그렇게
        // 학습되었으므로 그대로 따른다.
        let istft = Istft::new(16, 4);
        let bins = istft.bins();
        let mut rows = vec![-30.0_f32; 12 * 2 * bins];
        for frame in 0..12 {
            rows[frame * 2 * bins + 3] = 0.0;
            // 토막이 4표본씩 밀리므로 위상도 그만큼 민다.
            rows[frame * 2 * bins + bins + 3] =
                (std::f64::consts::TAU * 3.0 * (frame * 4) as f64 / 16.0) as f32;
            rows[frame * 2 * bins + bins..(frame + 1) * 2 * bins]
                .iter_mut()
                .enumerate()
                .filter(|(bin, _)| *bin != 3)
                .for_each(|(_, phase)| *phase = 0.0);
        }
        let audio = istft.synthesize(&rows);
        // 잘라 낸 6표본만큼 밀려 있다.
        for (index, &sample) in audio.iter().enumerate().skip(8).take(24) {
            let t = (index + 6) as f64;
            let wanted = 4.0 / 3.0 * 2.0 / 16.0 * (std::f64::consts::TAU * 3.0 * t / 16.0).cos();
            assert!(
                (f64::from(sample) - wanted).abs() < 1e-5,
                "{index}: {sample} {wanted}"
            );
        }
    }
}
