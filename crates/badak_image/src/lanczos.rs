//! 란초스 필터로 크기 바꾸기 (36장).
//!
//! 27장의 쌍선형 방법은 새 픽셀의 비중을 삼각형으로 정했다. 란초스는
//! 더 넓은 범위(양쪽으로 세 픽셀)를 보고, 비중이 물결 모양이어서 음수도
//! 있다. 가장자리가 덜 뭉개진다. 36장의 모델이 이 방법으로 줄인 사진으로
//! 학습되었다. Pillow의 `LANCZOS`와 같은 바이트를 내는 것이 기준이다.

use crate::{CHANNELS, Image};

/// Pillow가 계수를 정수로 바꿀 때 쓰는 소수점 아래 비트 수.
const PRECISION_BITS: u32 = 32 - 8 - 2;
/// 란초스 필터가 보는 범위(픽셀).
const SUPPORT: f64 = 3.0;

// ANCHOR: kernel
/// 란초스 필터의 비중. `sinc(x) × sinc(x / 3)`이고 3 밖에서는 0이다.
fn lanczos(x: f64) -> f64 {
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            let angle = x * std::f64::consts::PI;
            angle.sin() / angle
        }
    };
    if (-SUPPORT..SUPPORT).contains(&x) {
        sinc(x) * sinc(x / SUPPORT)
    } else {
        0.0
    }
}
// ANCHOR_END: kernel

/// 한 축에서, 새 픽셀마다 (첫 옛 픽셀의 번호, 비중들)을 정한다. 27장의
/// `coefficients`와 같은 계산이고 필터만 다르다. 음수 비중은 0에서 먼
/// 쪽으로 반올림한다(Pillow의 `normalize_coeffs_8bpc`).
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = SUPPORT * filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let last = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (first..last)
                .map(|source| lanczos((source as f64 - center + 0.5) / filter_scale))
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed = weights
                .iter()
                .map(|&weight| {
                    let weight = if total != 0.0 { weight / total } else { weight };
                    let scaled = weight * f64::from(1_u32 << PRECISION_BITS);
                    if scaled < 0.0 {
                        (scaled - 0.5) as i32
                    } else {
                        (scaled + 0.5) as i32
                    }
                })
                .collect();
            (first, fixed)
        })
        .collect()
}

fn clip(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// 가로로 크기를 바꾼다.
fn columns(image: &Image, width: usize) -> Image {
    let half = 1 << (PRECISION_BITS - 1);
    let taps = coefficients(image.width, width);
    let mut pixels = vec![0_u8; image.height * width * CHANNELS];
    for (source, target) in image
        .pixels
        .chunks_exact(image.width * CHANNELS)
        .zip(pixels.chunks_exact_mut(width * CHANNELS))
    {
        for (pixel, (first, weights)) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().zip(&taps)
        {
            let mut sums = [half; CHANNELS];
            for (tap, &weight) in weights.iter().enumerate() {
                let values = &source[(first + tap) * CHANNELS..][..CHANNELS];
                for (sum, &value) in sums.iter_mut().zip(values) {
                    *sum += i32::from(value) * weight;
                }
            }
            for (value, sum) in pixel.iter_mut().zip(sums) {
                *value = clip(sum);
            }
        }
    }
    Image {
        width,
        height: image.height,
        pixels,
    }
}

/// 세로로 크기를 바꾼다.
fn rows(image: &Image, height: usize) -> Image {
    let half = 1 << (PRECISION_BITS - 1);
    let taps = coefficients(image.height, height);
    let line = image.width * CHANNELS;
    let mut pixels = vec![0_u8; height * line];
    let mut sums = vec![0_i32; line];
    for (target, (first, weights)) in pixels.chunks_exact_mut(line).zip(&taps) {
        sums.fill(half);
        for (tap, &weight) in weights.iter().enumerate() {
            let source = &image.pixels[(first + tap) * line..][..line];
            for (sum, &value) in sums.iter_mut().zip(source) {
                *sum += i32::from(value) * weight;
            }
        }
        for (value, &sum) in target.iter_mut().zip(&sums) {
            *value = clip(sum);
        }
    }
    Image {
        width: image.width,
        height,
        pixels,
    }
}

// ANCHOR: resize
/// 란초스 필터로 크기를 바꾼다. 한 축씩 두 번에 나눠 하고, 크기가 같은
/// 축은 건드리지 않는다. 보통은 가로를 먼저 하지만, 세로로 줄어드는 픽셀
/// 수가 가로의 두 배보다 많으면 세로를 먼저 한다(27장에서 본 Pillow의
/// 순서다. 순서가 다르면 중간의 반올림이 달라진다).
pub fn resize(image: &Image, width: usize, height: usize) -> Image {
    let across = |image: &Image| {
        if width == image.width {
            image.clone()
        } else {
            columns(image, width)
        }
    };
    let down = |image: &Image| {
        if height == image.height {
            image.clone()
        } else {
            rows(image, height)
        }
    };
    let shrink = |from: usize, to: usize| from as i64 - to as i64;
    let (vertical, horizontal) = (shrink(image.height, height), shrink(image.width, width));
    if vertical > 0 && vertical > horizontal * 2 {
        across(&down(image))
    } else {
        down(&across(image))
    }
}
// ANCHOR_END: resize

// ANCHOR: preprocess
/// 사진을 SmolVLM의 입력으로 바꾼다. 긴 변이 `longest`픽셀이 되게 늘리거나
/// 줄이고, 다시 `side × side`의 정사각형으로 줄인다(가로세로의 비율은
/// 지키지 않는다). 값은 −1에서 1 사이로 옮기고 `(3, side, side)`의
/// 순서로 놓는다.
///
/// 두 번에 나눠 줄이는 것은 transformers의 전처리가 그렇게 하기
/// 때문이다. 사진을 여러 조각으로 나누는 기능을 끄면 첫 단계는 쓸모가
/// 없지만, 결과의 바이트를 맞추려면 따라야 한다.
pub fn squash(image: &Image, longest: usize, side: usize) -> Vec<f32> {
    // 줄인 변의 길이는 소수점 아래를 버리고, 홀수이면 1을 더한다.
    let aspect = image.width as f64 / image.height as f64;
    let even = |length: f64| {
        let length = length as usize;
        (length + length % 2).max(1)
    };
    let (width, height) = if image.width >= image.height {
        (longest, even(longest as f64 / aspect))
    } else {
        (even(longest as f64 * aspect), longest)
    };
    let large = resize(image, width, height);
    let small = resize(&large, side, side);
    let mut planes = vec![0.0; 3 * side * side];
    for (cell, pixel) in small.pixels.as_chunks::<3>().0.iter().enumerate() {
        for (channel, &value) in pixel.iter().enumerate() {
            planes[channel * side * side + cell] = (f32::from(value) / 255.0 - 0.5) / 0.5;
        }
    }
    planes
}
// ANCHOR_END: preprocess

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_is_one_at_the_center_and_zero_at_whole_pixels() {
        assert_eq!(lanczos(0.0), 1.0);
        for x in [1.0, 2.0, -1.0, -2.0] {
            assert!(lanczos(x).abs() < 1e-12, "{x}");
        }
        assert_eq!(lanczos(3.0), 0.0);
        assert_eq!(lanczos(-3.5), 0.0);
        // 중심에서 한 픽셀 반쯤 떨어진 곳은 음수다.
        assert!(lanczos(1.5) < 0.0);
    }

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        for (input, output) in [(640, 512), (100, 512), (512, 512), (2048, 512)] {
            for (_, weights) in coefficients(input, output) {
                let total: i64 = weights.iter().map(|&weight| i64::from(weight)).sum();
                assert!((total - (1_i64 << PRECISION_BITS)).abs() <= weights.len() as i64);
            }
        }
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let image = Image {
            width: 9,
            height: 7,
            pixels: vec![200; 9 * 7 * 3],
        };
        let resized = resize(&image, 4, 12);
        assert_eq!((resized.width, resized.height), (4, 12));
        assert!(
            resized
                .pixels
                .iter()
                .all(|&value| (199..=201).contains(&value))
        );
    }
}
