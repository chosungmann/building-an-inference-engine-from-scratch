//! 멜 스펙트로그램.
//!
//! 소리를 짧은 토막으로 끊어 토막마다 푸리에 변환을 하고, 주파수들을 사람의
//! 귀가 구별하는 간격으로 묶은 뒤, 세기를 로그로 바꾼다. 가로가 시간, 세로가
//! 소리의 높이, 값이 세기인 그림이 된다.

use std::f64::consts::PI;

use crate::fft::{Bluestein, Complex};

// ANCHOR: window
/// 한(Hann) 창. 토막의 양 끝을 0으로 부드럽게 줄인다.
///
/// 소리를 칼로 자르듯 끊으면 끊긴 자리가 급한 변화처럼 보여, 원래 없던 높은
/// 주파수가 변환에 나타난다. 토막에 이 창을 곱해 양 끝을 눌러 준다.
/// 길이 `n`을 한 주기로 보는 꼴(끝 값이 처음 값과 이어진다)이다.
pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f64 / n as f64).cos())
        .collect()
}
// ANCHOR_END: window

// ANCHOR: mel_scale
/// 헤르츠를 멜로 바꾼다. 1,000Hz 아래는 비례하고 그 위는 로그다.
///
/// 사람은 100Hz와 200Hz의 차이는 또렷이 듣지만 5,000Hz와 5,100Hz의 차이는
/// 거의 듣지 못한다. 멜은 "같은 간격이 같은 차이로 들리게" 만든 눈금이다.
/// 눈금을 정하는 식은 여럿이다. 이것은 Whisper가 쓰는 꼴(Slaney)이다.
pub fn hertz_to_mel(hertz: f64) -> f64 {
    const BREAK: f64 = 1000.0;
    const LINEAR: f64 = 200.0 / 3.0;
    if hertz < BREAK {
        hertz / LINEAR
    } else {
        // 1,000Hz 위에서는 6.4배마다 27멜이다.
        BREAK / LINEAR + (hertz / BREAK).ln() * 27.0 / 6.4_f64.ln()
    }
}

/// 멜을 헤르츠로 바꾼다. [`hertz_to_mel`]의 역이다.
pub fn mel_to_hertz(mel: f64) -> f64 {
    const BREAK: f64 = 1000.0;
    const LINEAR: f64 = 200.0 / 3.0;
    if mel < BREAK / LINEAR {
        mel * LINEAR
    } else {
        BREAK * ((mel - BREAK / LINEAR) * 6.4_f64.ln() / 27.0).exp()
    }
}
// ANCHOR_END: mel_scale

// ANCHOR: filters
/// 멜 필터 뭉치. `(mels, n_fft / 2 + 1)` 행렬이고, 행 하나가 주파수 칸들을
/// 묶는 세모꼴 가중치다.
///
/// 0Hz부터 표본 속도의 절반까지를 멜 눈금에서 같은 간격으로 나눈 점
/// `mels + 2`개를 잡는다. `i`번째 필터는 `i`번째 점에서 0, `i + 1`번째 점에서
/// 꼭대기, `i + 2`번째 점에서 다시 0인 세모다. 높은 쪽 필터일수록 넓으므로
/// 넓이가 같아지도록 높이를 낮춘다.
pub fn mel_filters(sample_rate: u32, n_fft: usize, mels: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let top = hertz_to_mel(f64::from(sample_rate) / 2.0);
    let points: Vec<f64> = (0..mels + 2)
        .map(|index| mel_to_hertz(top * index as f64 / (mels + 1) as f64))
        .collect();
    let mut filters = vec![0.0_f32; mels * bins];
    for (mel, row) in filters.chunks_exact_mut(bins).enumerate() {
        let (low, peak, high) = (points[mel], points[mel + 1], points[mel + 2]);
        let height = 2.0 / (high - low);
        for (bin, weight) in row.iter_mut().enumerate() {
            let hertz = bin as f64 * f64::from(sample_rate) / n_fft as f64;
            let rising = (hertz - low) / (peak - low);
            let falling = (high - hertz) / (high - peak);
            *weight = (rising.min(falling).max(0.0) * height) as f32;
        }
    }
    filters
}
// ANCHOR_END: filters

/// 로그 멜 스펙트로그램. `(mels, frames)`이고 행 하나가 멜 칸 하나다.
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogram {
    /// 멜 칸의 수.
    pub mels: usize,
    /// 토막의 수.
    pub frames: usize,
    /// 값들. 멜 칸마다 토막들이 이어져 있다.
    pub data: Vec<f32>,
}

// ANCHOR: settings
/// 스펙트로그램을 만드는 방법. 모델을 학습한 사람이 정한 것을 따라야 한다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogMel {
    /// 소리의 표본 속도(Hz).
    pub sample_rate: u32,
    /// 토막 하나의 표본 수.
    pub n_fft: usize,
    /// 토막을 옮기는 간격(표본).
    pub hop: usize,
    /// 멜 칸의 수.
    pub mels: usize,
    /// 입력의 길이(표본). 짧으면 0으로 채우고 길면 자른다.
    pub samples: usize,
}

impl LogMel {
    /// Whisper의 설정. 16kHz, 25밀리초짜리 토막을 10밀리초마다, 멜 칸 80개,
    /// 30초.
    pub const WHISPER: Self = Self {
        sample_rate: 16_000,
        n_fft: 400,
        hop: 160,
        mels: 80,
        samples: 480_000,
    };

    /// 토막의 수.
    pub fn frames(&self) -> usize {
        self.samples / self.hop
    }
}
// ANCHOR_END: settings

impl LogMel {
    // ANCHOR: spectrogram
    /// 소리를 로그 멜 스펙트로그램으로 바꾼다.
    pub fn spectrogram(&self, samples: &[f32]) -> MelSpectrogram {
        let (n_fft, bins, frames) = (self.n_fft, self.n_fft / 2 + 1, self.frames());
        // 1. 길이를 맞춘다. 모자라면 0(침묵)을 붙인다.
        let mut audio = vec![0.0_f64; self.samples];
        for (slot, &sample) in audio.iter_mut().zip(samples) {
            *slot = f64::from(sample);
        }
        // 2. 토막마다: 창을 곱하고, 변환하고, 세기를 구한다. `t`번째 토막의
        //    가운데가 표본 `t × hop`이다. 처음과 끝의 토막은 소리 밖으로
        //    나가는데, 그 자리는 소리를 거울처럼 뒤집어 채운다.
        let half = (n_fft / 2) as isize;
        let last = self.samples as isize - 1;
        let mirrored = |index: isize| -> f64 {
            let index = if index < 0 { -index } else { index };
            let index = if index > last {
                2 * last - index
            } else {
                index
            };
            audio[index.clamp(0, last) as usize]
        };
        let window = hann(n_fft);
        let plan = Bluestein::new(n_fft);
        let filters = mel_filters(self.sample_rate, n_fft, self.mels);
        // 토막마다의 계산은 서로 무관하다. 스레드에 나눈다. 스레드마다 변환에
        // 쓸 자리를 따로 쥔다.
        thread_local! {
            static BUFFERS: std::cell::RefCell<(Vec<Complex>, Vec<Complex>, Vec<f32>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        let mut by_frame = vec![0.0_f32; frames * self.mels];
        let pool = badak_core::threads::global();
        let work = |index: usize, column: &mut [f32]| {
            BUFFERS.with_borrow_mut(|(frame, scratch, power)| {
                frame.resize(n_fft, Complex::default());
                power.resize(bins, 0.0);
                let start = (index * self.hop) as isize - half;
                for (offset, slot) in frame.iter_mut().enumerate() {
                    *slot = Complex {
                        re: mirrored(start + offset as isize) * window[offset],
                        im: 0.0,
                    };
                }
                plan.transform(frame, scratch);
                // 입력이 실수이면 뒤쪽 절반은 앞쪽 절반의 켤레다. 앞쪽만 쓴다.
                for (slot, value) in power.iter_mut().zip(frame.iter()) {
                    *slot = value.power() as f32;
                }
                // 3. 주파수 칸들을 멜 칸으로 묶는다.
                for (slot, weights) in column.iter_mut().zip(filters.chunks_exact(bins)) {
                    let energy: f32 = weights.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
                    // 4. 로그. 0의 로그를 피하려고 아주 작은 값으로 받친다.
                    *slot = energy.max(1e-10).log10();
                }
            });
        };
        pool.pool()
            .run_chunks(&mut by_frame, self.mels, pool.for_vectors(frames), &work);
        // (토막, 멜 칸)으로 계산한 것을 (멜 칸, 토막)으로 옮긴다.
        let mut data = vec![0.0_f32; self.mels * frames];
        for (index, column) in by_frame.chunks_exact(self.mels).enumerate() {
            for (mel, &value) in column.iter().enumerate() {
                data[mel * frames + index] = value;
            }
        }
        // 5. 가장 큰 값에서 8(여덟 자릿수) 아래는 잘라 내고, 0 근처의 값이
        //    되도록 옮기고 줄인다.
        let peak = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut data {
            *value = (value.max(peak - 8.0) + 4.0) / 4.0;
        }
        MelSpectrogram {
            mels: self.mels,
            frames,
            data,
        }
    }
    // ANCHOR_END: spectrogram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_and_scale_have_the_expected_shape() {
        let window = hann(400);
        assert_eq!(window[0], 0.0);
        assert!((window[200] - 1.0).abs() < 1e-12);
        assert!((window[100] - window[300]).abs() < 1e-12);
        // 1,000Hz가 15멜이고, 그 아래는 비례한다.
        assert!((hertz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        assert!((hertz_to_mel(200.0) - 3.0).abs() < 1e-12);
        assert!((hertz_to_mel(6400.0) - 42.0).abs() < 1e-9);
        for hertz in [0.0, 50.0, 999.0, 1000.0, 2500.0, 8000.0] {
            assert!((mel_to_hertz(hertz_to_mel(hertz)) - hertz).abs() < 1e-6);
        }
    }

    #[test]
    fn filters_are_triangles_of_equal_area() {
        let (rate, n_fft, mels) = (16_000, 400, 80);
        let filters = mel_filters(rate, n_fft, mels);
        let bins = n_fft / 2 + 1;
        assert_eq!(filters.len(), mels * bins);
        for row in filters.chunks_exact(bins) {
            assert!(row.iter().all(|&weight| weight >= 0.0));
            // 0이 아닌 칸들이 한 덩어리이고, 올라갔다 내려온다.
            let active: Vec<usize> = (0..bins).filter(|&bin| row[bin] > 0.0).collect();
            assert!(!active.is_empty());
            assert_eq!(active.last().unwrap() - active[0] + 1, active.len());
            let top = active
                .iter()
                .copied()
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap();
            assert!(row[active[0]..=top].is_sorted());
            assert!(row[top..=*active.last().unwrap()].iter().rev().is_sorted());
        }
        // 낮은 쪽 필터는 좁고 높고, 높은 쪽 필터는 넓고 낮다.
        let peak = |mel: usize| {
            filters[mel * bins..][..bins]
                .iter()
                .copied()
                .fold(0.0, f32::max)
        };
        assert!(peak(0) > 4.0 * peak(79));
    }

    #[test]
    fn a_tone_lights_up_one_band() {
        let settings = LogMel {
            samples: 16_000,
            ..LogMel::WHISPER
        };
        assert_eq!(settings.frames(), 100);
        // 1초짜리 440Hz.
        let tone: Vec<f32> = (0..16_000)
            .map(|t| (2.0 * PI * 440.0 * t as f64 / 16_000.0).sin() as f32 * 0.5)
            .collect();
        let picture = settings.spectrogram(&tone);
        assert_eq!((picture.mels, picture.frames), (80, 100));
        // 가운데 토막에서 가장 밝은 멜 칸이 440Hz가 든 칸이다.
        let column: Vec<f32> = (0..80).map(|mel| picture.data[mel * 100 + 50]).collect();
        let brightest = (0..80)
            .max_by(|&a, &b| column[a].total_cmp(&column[b]))
            .unwrap();
        let low = mel_to_hertz(hertz_to_mel(8000.0) * brightest as f64 / 81.0);
        let high = mel_to_hertz(hertz_to_mel(8000.0) * (brightest + 2) as f64 / 81.0);
        assert!(low < 440.0 && 440.0 < high, "{low} ~ {high}");
        // 침묵은 가장 큰 값에서 8 아래로 잘린 바닥이다.
        let silence = settings.spectrogram(&[]);
        assert!(silence.data.iter().all(|&value| value == silence.data[0]));
        assert!((silence.data[0] - (-10.0 + 4.0) / 4.0).abs() < 1e-6);
    }
}
