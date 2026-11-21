//! 이미지 전처리 (27장).
//!
//! 사진 파일에서 모델의 입력까지는 네 단계다. 파일을 픽셀로 풀고, 크기를
//! 줄이고, 가운데를 잘라 내고, 숫자의 범위와 레이아웃을 모델이 학습할 때 본
//! 것과 같게 맞춘다. 첫 단계만 외부 크레이트에 맡기고 나머지는 직접 한다.
//!
//! 모델은 학습할 때 본 것과 "거의 같은" 입력에도 다른 답을 낼 수 있다.
//! 그래서 이 크레이트의 기준은 "그럴듯한 결과"가 아니라 학습에 쓴 도구
//! (Pillow)와 같은 픽셀이다.

pub mod lanczos;
pub mod png;

use std::fmt;
use std::path::Path;

/// 전처리에서 생기는 오류.
#[derive(Debug)]
pub enum Error {
    /// 파일을 읽지 못했다.
    Io(std::io::Error),
    /// 파일을 픽셀로 풀지 못했다.
    Decode(String),
    /// 풀 수는 있지만 이 크레이트가 다루지 않는 종류다.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "파일을 읽지 못했습니다: {error}"),
            Self::Decode(message) => write!(f, "사진을 풀지 못했습니다: {message}"),
            Self::Unsupported(what) => write!(f, "아직 지원하지 않습니다: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// 이 크레이트의 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;

// ANCHOR: image
/// 채널마다 8비트인 RGB 사진.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 픽셀들. 위에서 아래로, 왼쪽에서 오른쪽으로, 픽셀마다 빨강·초록·파랑의
    /// 순서다. 길이는 `너비 × 높이 × 3`이다.
    pub pixels: Vec<u8>,
}

/// 색의 채널 수.
pub const CHANNELS: usize = 3;
// ANCHOR_END: image

impl Image {
    /// PNG 파일로 쓴다. 압축하지 않는다([`png`]).
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        Ok(std::fs::write(path, png::encode(self))?)
    }
}

// ANCHOR: decode
impl Image {
    /// JPEG 파일을 연다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::decode_jpeg(&std::fs::read(path)?)
    }

    /// JPEG 바이트를 픽셀로 푼다. 이 일만 외부 크레이트가 한다.
    pub fn decode_jpeg(bytes: &[u8]) -> Result<Self> {
        let mut decoder = jpeg_decoder::Decoder::new(bytes);
        let pixels = decoder
            .decode()
            .map_err(|error| Error::Decode(error.to_string()))?;
        let info = decoder
            .info()
            .ok_or_else(|| Error::Decode("사진의 크기를 알 수 없습니다".into()))?;
        let pixels = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => pixels,
            // 흑백 사진은 세 채널에 같은 값을 넣는다.
            jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&gray| [gray; 3]).collect(),
            other => return Err(Error::Unsupported(format!("픽셀 형식 {other:?}"))),
        };
        Ok(Self {
            width: usize::from(info.width),
            height: usize::from(info.height),
            pixels,
        })
    }
}
// ANCHOR_END: decode

// ANCHOR: resampling
/// 크기를 바꿀 때 새 픽셀의 값을 정하는 방법. 둘 다 "쌍선형"이라고 불린다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    /// 새 픽셀이 덮는 넓이 안의 옛 픽셀을 모두 본다. 줄이는 비율이 클수록
    /// 많은 픽셀을 섞는다. Pillow의 방법이고, 결과의 바이트까지 같다.
    Area,
    /// 새 픽셀의 중심에 가장 가까운 옛 픽셀 둘만 본다. 줄이는 비율이 커도
    /// 둘이다. 그 사이의 픽셀은 버려진다.
    TwoTaps,
}
// ANCHOR_END: resampling

/// Pillow가 계수를 정수로 바꿀 때 쓰는 소수점 아래 비트 수.
const PRECISION_BITS: u32 = 32 - 8 - 2;

// ANCHOR: coefficients
/// 한 축에서, 새 픽셀마다 "어느 옛 픽셀들을 어떤 비중으로 섞을지"를 정한다.
/// 새 픽셀마다 (첫 옛 픽셀의 번호, 비중들)을 돌려준다.
///
/// Pillow의 `precompute_coeffs`와 `normalize_coeffs_8bpc`를 옮긴 것이다.
/// 비중은 삼각형 모양이다. 새 픽셀의 중심에서 가장 크고, 멀어질수록 줄어
/// `support`만큼 떨어지면 0이 된다. 줄일 때는 삼각형이 줄이는 비율만큼
/// 넓어진다.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let last = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (first..last)
                .map(|source| {
                    let distance = ((source as f64 - center + 0.5) / filter_scale).abs();
                    (1.0 - distance).max(0.0)
                })
                .collect();
            let total: f64 = weights.iter().sum();
            // 비중의 합을 1로 맞춘 뒤 정수로 바꾼다. 계산을 정수로 하면
            // 어느 기계에서나 같은 바이트가 나온다.
            let fixed = weights
                .iter()
                .map(|&weight| {
                    let weight = if total != 0.0 { weight / total } else { weight };
                    (0.5 + weight * f64::from(1_u32 << PRECISION_BITS)) as i32
                })
                .collect();
            (first, fixed)
        })
        .collect()
}

/// 정수로 더한 값을 8비트로 되돌린다.
fn clip(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}
// ANCHOR_END: coefficients

impl Image {
    // ANCHOR: resize
    /// 크기를 바꾼다.
    pub fn resize(&self, width: usize, height: usize, resampling: Resampling) -> Self {
        match resampling {
            Resampling::Area => self.resize_area(width, height),
            Resampling::TwoTaps => self.resize_two_taps(width, height),
        }
    }

    /// 한 축씩 두 번에 나눠 줄인다. 한 번 줄일 때마다 결과를 8비트로
    /// 되돌린다. 보통은 가로를 먼저 하지만, 세로로 줄어드는 픽셀 수가 가로의
    /// 두 배보다 많으면 세로를 먼저 한다. Pillow가 하는 순서 그대로이고,
    /// 순서가 다르면 중간의 반올림이 달라져 결과의 바이트가 달라진다.
    fn resize_area(&self, width: usize, height: usize) -> Self {
        let shrink = |from: usize, to: usize| from as i64 - to as i64;
        let (down, across) = (shrink(self.height, height), shrink(self.width, width));
        if down > 0 && down > across * 2 {
            self.resize_rows(height).resize_columns(width)
        } else {
            self.resize_columns(width).resize_rows(height)
        }
    }

    /// 가로로 줄인다. 줄마다 따로 계산한다.
    fn resize_columns(&self, width: usize) -> Self {
        let half = 1 << (PRECISION_BITS - 1);
        let columns = coefficients(self.width, width);
        let mut pixels = vec![0_u8; self.height * width * CHANNELS];
        for (source, target) in self
            .pixels
            .chunks_exact(self.width * CHANNELS)
            .zip(pixels.chunks_exact_mut(width * CHANNELS))
        {
            for (pixel, (first, weights)) in target
                .as_chunks_mut::<CHANNELS>()
                .0
                .iter_mut()
                .zip(&columns)
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
        Self {
            width,
            height: self.height,
            pixels,
        }
    }

    /// 세로로 줄인다. 새 줄 하나가 옛 줄 여럿을 비중대로 섞은 것이다.
    fn resize_rows(&self, height: usize) -> Self {
        let half = 1 << (PRECISION_BITS - 1);
        let rows = coefficients(self.height, height);
        let line = self.width * CHANNELS;
        let mut pixels = vec![0_u8; height * line];
        let mut sums = vec![0_i32; line];
        for (target, (first, weights)) in pixels.chunks_exact_mut(line).zip(&rows) {
            sums.fill(half);
            for (tap, &weight) in weights.iter().enumerate() {
                let source = &self.pixels[(first + tap) * line..][..line];
                for (sum, &value) in sums.iter_mut().zip(source) {
                    *sum += i32::from(value) * weight;
                }
            }
            for (value, &sum) in target.iter_mut().zip(&sums) {
                *value = clip(sum);
            }
        }
        Self {
            width: self.width,
            height,
            pixels,
        }
    }
    // ANCHOR_END: resize

    // ANCHOR: two_taps
    /// 새 픽셀의 중심이 옛 사진의 어디에 떨어지는지를 보고, 그 둘레의 옛 픽셀
    /// 넷(가로로 둘, 세로로 둘)을 거리에 따라 섞는다.
    fn resize_two_taps(&self, width: usize, height: usize) -> Self {
        // 한 축에서 (왼쪽 픽셀, 오른쪽 픽셀, 오른쪽의 비중).
        let taps = |input: usize, output: usize| -> Vec<(usize, usize, f32)> {
            let scale = input as f32 / output as f32;
            (0..output)
                .map(|index| {
                    let position =
                        ((index as f32 + 0.5) * scale - 0.5).clamp(0.0, (input - 1) as f32);
                    let low = position.floor() as usize;
                    (low, (low + 1).min(input - 1), position - low as f32)
                })
                .collect()
        };
        let (columns, rows) = (taps(self.width, width), taps(self.height, height));
        let at = |row: usize, column: usize| {
            &self.pixels[(row * self.width + column) * CHANNELS..][..CHANNELS]
        };
        let mut pixels = Vec::with_capacity(width * height * CHANNELS);
        for &(top, bottom, down) in &rows {
            for &(left, right, across) in &columns {
                for channel in 0..CHANNELS {
                    let value = |row: usize, column: usize| f32::from(at(row, column)[channel]);
                    let upper = value(top, left) * (1.0 - across) + value(top, right) * across;
                    let lower =
                        value(bottom, left) * (1.0 - across) + value(bottom, right) * across;
                    pixels.push((upper * (1.0 - down) + lower * down).round() as u8);
                }
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
    // ANCHOR_END: two_taps

    // ANCHOR: crop
    /// 가운데의 `width × height`를 잘라 낸다.
    pub fn center_crop(&self, width: usize, height: usize) -> Self {
        let (width, height) = (width.min(self.width), height.min(self.height));
        let (left, top) = ((self.width - width) / 2, (self.height - height) / 2);
        let pixels = (top..top + height)
            .flat_map(|row| {
                let start = (row * self.width + left) * CHANNELS;
                &self.pixels[start..start + width * CHANNELS]
            })
            .copied()
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }
    // ANCHOR_END: crop
}

// ANCHOR: preprocess
/// 사진을 모델의 입력으로 바꾸는 방법. 모델마다 다르고, 모델을 학습한
/// 사람이 정한 것을 그대로 따라야 한다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Preprocess {
    /// 짧은 변을 이 길이로 맞춘다.
    pub resize: usize,
    /// 그 뒤에 가운데에서 잘라 낼 정사각형의 한 변.
    pub crop: usize,
    /// 크기를 바꾸는 방법.
    pub resampling: Resampling,
    /// 채널마다 뺄 값. 픽셀을 0과 1 사이로 바꾼 뒤에 뺀다.
    pub mean: [f32; CHANNELS],
    /// 채널마다 나눌 값.
    pub std: [f32; CHANNELS],
}

impl Preprocess {
    /// ImageNet으로 학습한 분류 모델들이 흔히 쓰는 설정.
    pub const IMAGENET: Self = Self {
        resize: 256,
        crop: 224,
        resampling: Resampling::Area,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };

    /// 짧은 변이 `resize`가 되는 크기. 긴 변은 비율대로 줄이고 가장 가까운
    /// 정수로 맞춘다.
    pub fn resized(&self, width: usize, height: usize) -> (usize, usize) {
        let scale = self.resize as f64 / width.min(height) as f64;
        let scaled =
            |extent: usize| ((extent as f64 * scale).round_ties_even() as usize).max(self.resize);
        (scaled(width), scaled(height))
    }

    /// 사진을 줄이고 잘라 낸다. 숫자로 바꾸기 직전의 사진이다.
    pub fn shape(&self, image: &Image) -> Image {
        let (width, height) = self.resized(image.width, image.height);
        image
            .resize(width, height, self.resampling)
            .center_crop(self.crop, self.crop)
    }

    /// 사진을 모델의 입력으로 바꾼다. 결과는 `(3, crop, crop)`의 순서로 놓인
    /// `f32`들이다. 빨강 전체, 초록 전체, 파랑 전체의 순서다.
    pub fn apply(&self, image: &Image) -> Vec<f32> {
        let shaped = self.shape(image);
        let plane = shaped.width * shaped.height;
        let mut values = vec![0.0_f32; CHANNELS * plane];
        for (index, pixel) in shaped.pixels.as_chunks::<CHANNELS>().0.iter().enumerate() {
            for channel in 0..CHANNELS {
                let scaled = f32::from(pixel[channel]) / 255.0;
                values[channel * plane + index] = (scaled - self.mean[channel]) / self.std[channel];
            }
        }
        values
    }
}
// ANCHOR_END: preprocess

// ANCHOR: letterbox
impl Image {
    /// 사진을 비율을 지켜 한 변이 `side`인 정사각형 안에 들어가게 줄이고,
    /// 남는 자리를 `fill`로 채운다 (32장). 사진은 왼쪽 위에 놓는다.
    ///
    /// 결과는 `(3, side, side)`의 순서로 놓인 `f32`들이고 값은 0~255
    /// 그대로다. 채널은 **파랑, 초록, 빨강**의 순서다. YOLOX가 그렇게
    /// 학습되었다. 줄인 비율도 함께 돌려준다. 모델이 낸 상자를 원래 사진의
    /// 좌표로 되돌릴 때 쓴다.
    ///
    /// 분류 모델처럼 가운데를 잘라 내지 않는 까닭은, 잘라 낸 곳에 있던
    /// 물체는 찾을 수 없기 때문이다. 비율을 바꿔 늘이지 않는 까닭은 물체의
    /// 모양이 달라지기 때문이다.
    pub fn letterbox(&self, side: usize, fill: u8) -> (Vec<f32>, f32) {
        let ratio = (side as f32 / self.width as f32).min(side as f32 / self.height as f32);
        let width = ((self.width as f32 * ratio) as usize).clamp(1, side);
        let height = ((self.height as f32 * ratio) as usize).clamp(1, side);
        let small = self.resize(width, height, Resampling::Area);
        let plane = side * side;
        let mut values = vec![f32::from(fill); CHANNELS * plane];
        for (row, line) in small.pixels.chunks_exact(width * CHANNELS).enumerate() {
            for (column, pixel) in line.as_chunks::<CHANNELS>().0.iter().enumerate() {
                // 빨강·초록·파랑을 파랑·초록·빨강의 자리에.
                for (channel, &value) in pixel.iter().enumerate() {
                    let target = (CHANNELS - 1 - channel) * plane + row * side + column;
                    values[target] = f32::from(value);
                }
            }
        }
        (values, ratio)
    }
}
// ANCHOR_END: letterbox

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: usize, height: usize) -> Image {
        let pixels = (0..height)
            .flat_map(|row| {
                (0..width).flat_map(move |column| {
                    [
                        (column * 255 / (width - 1)) as u8,
                        (row * 255 / (height - 1)) as u8,
                        128,
                    ]
                })
            })
            .collect();
        Image {
            width,
            height,
            pixels,
        }
    }

    #[test]
    fn coefficients_sum_to_one() {
        for (input, output) in [(2380, 319), (1912, 256), (300, 256), (256, 256), (100, 256)] {
            for (first, weights) in coefficients(input, output) {
                assert!(first + weights.len() <= input);
                let total: i64 = weights.iter().map(|&weight| i64::from(weight)).sum();
                // 정수로 바꾸면서 생긴 반올림만큼만 어긋난다.
                assert!((total - (1_i64 << PRECISION_BITS)).abs() <= weights.len() as i64);
            }
        }
        // 줄이는 비율이 크면 한 픽셀이 많은 픽셀을 본다.
        assert!(coefficients(2560, 256)[100].1.len() >= 19);
        // 크기가 같으면 자기 자신만 본다.
        let same = coefficients(8, 8);
        assert!(same.iter().enumerate().all(|(index, (first, weights))| {
            weights
                .iter()
                .enumerate()
                .all(|(tap, &weight)| (weight != 0) == (first + tap == index))
        }));
    }

    #[test]
    fn resizing_to_the_same_size_changes_nothing() {
        let image = gradient(17, 9);
        assert_eq!(image.resize(17, 9, Resampling::Area), image);
        assert_eq!(image.resize(17, 9, Resampling::TwoTaps), image);
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let image = Image {
            width: 40,
            height: 30,
            pixels: vec![77; 40 * 30 * CHANNELS],
        };
        for resampling in [Resampling::Area, Resampling::TwoTaps] {
            let small = image.resize(7, 5, resampling);
            assert_eq!((small.width, small.height), (7, 5));
            assert!(small.pixels.iter().all(|&value| value == 77));
        }
    }

    #[test]
    fn area_averages_what_two_taps_skips() {
        // 세로 줄무늬를 4분의 1로 줄인다. 네 칸마다 가운데 두 칸이 희고 양쪽
        // 두 칸이 검다. 새 픽셀의 중심은 늘 흰 두 칸의 사이에 떨어진다.
        // 둘만 보는 쪽은 흰색만 보고, 넓이를 보는 쪽은 검은 칸도 섞는다.
        let pixels = (0..64 * 64)
            .flat_map(|index| {
                [if matches!((index % 64) % 4, 1 | 2) {
                    255
                } else {
                    0
                }; 3]
            })
            .collect();
        let stripes = Image {
            width: 64,
            height: 64,
            pixels,
        };
        let taps = stripes.resize(16, 16, Resampling::TwoTaps);
        assert!(taps.pixels.iter().all(|&value| value == 255));
        let area = stripes.resize(16, 16, Resampling::Area);
        assert!(
            area.pixels
                .iter()
                .all(|&value| (100..=160).contains(&value))
        );
    }

    #[test]
    fn center_crop_takes_the_middle() {
        let image = gradient(10, 6);
        let crop = image.center_crop(4, 2);
        assert_eq!((crop.width, crop.height), (4, 2));
        assert_eq!(crop.pixels[..3], image.pixels[(2 * 10 + 3) * 3..][..3]);
        // 사진보다 크게 자를 수는 없다.
        let whole = image.center_crop(100, 100);
        assert_eq!(whole, image);
    }

    #[test]
    fn preprocess_normalizes_and_separates_channels() {
        let image = Image {
            width: 300,
            height: 400,
            pixels: [255, 0, 124].repeat(300 * 400),
        };
        let settings = Preprocess::IMAGENET;
        assert_eq!(settings.resized(300, 400), (256, 341));
        assert_eq!(settings.resized(4032, 3024), (341, 256));
        let values = settings.apply(&image);
        assert_eq!(values.len(), 3 * 224 * 224);
        let plane = 224 * 224;
        assert!((values[0] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((values[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-6);
        assert!((values[2 * plane] - (124.0 / 255.0 - 0.406) / 0.225).abs() < 1e-6);
    }

    #[test]
    fn broken_files_are_errors() {
        assert!(Image::decode_jpeg(b"not a jpeg").is_err());
        assert!(Image::decode_jpeg(&[]).is_err());
        assert!(Image::open("/nonexistent/file.jpg").is_err());
    }
}
