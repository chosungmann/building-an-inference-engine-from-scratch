//! 수를 더 적은 비트로 나타내기: 반정밀도 실수와 정수 양자화 (19장).
//!
//! 이 모듈은 값을 바꾸는 함수만 담는다. 바꾼 값으로 빠르게 계산하는 커널은
//! 20장의 일이다.

// ANCHOR: f16_to_f32
/// 16비트 실수(IEEE 754 binary16)의 비트를 `f32`로 바꾼다. 값은 정확히 보존된다.
///
/// 16비트는 부호 1비트, 지수 5비트, 가수 10비트다.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from(half >> 10) & 0x1f;
    let mantissa = u32::from(half & 0x03ff);
    let bits = match exponent {
        // 지수가 0이면 가수가 곧 값이다. 단위는 2의 -24제곱이다.
        0 => (mantissa as f32 * TWO_TO_MINUS_24).to_bits() | sign,
        // 지수가 모두 1이면 무한대이거나 수가 아니다(NaN).
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        // 그 밖에는 지수의 기준만 옮기고(15 → 127) 가수를 왼쪽으로 붙인다.
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// 2의 -24제곱. 16비트 실수가 나타내는 가장 작은 양수다.
const TWO_TO_MINUS_24: f32 = f32::from_bits(0x3380_0000);
// ANCHOR_END: f16_to_f32

// ANCHOR: f16_from_f32
/// `f32`를 가장 가까운 16비트 실수의 비트로 바꾼다.
///
/// 두 값의 한가운데이면 마지막 비트가 0인 쪽을 고른다(IEEE 754의 기본
/// 반올림). 너무 큰 값은 무한대가 되고 너무 작은 값은 0이 된다.
pub fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 128 {
        // 무한대와 NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    if exponent > 15 {
        // 16비트 실수의 가장 큰 값(65504)보다 한참 크다.
        return sign | 0x7c00;
    }
    // `kept`는 남길 비트, `rest`는 버릴 비트, `midpoint`는 버릴 비트의 한가운데.
    let (kept, rest, midpoint) = if exponent >= -14 {
        // 보통의 수: 가수 23비트 가운데 위의 10비트를 남긴다.
        let kept = (((exponent + 15) as u32) << 10) | (mantissa >> 13);
        (kept, mantissa & 0x1fff, 0x1000)
    } else if exponent >= -25 {
        // 아주 작은 수: 지수를 더 내릴 수 없으므로 가수를 오른쪽으로 민다.
        // 유효 숫자가 그만큼 줄어든다.
        let significand = 0x0080_0000 | mantissa;
        let shift = (-1 - exponent) as u32;
        (
            significand >> shift,
            significand & ((1 << shift) - 1),
            1 << (shift - 1),
        )
    } else {
        // 가장 작은 양수의 절반보다도 작다.
        return sign;
    };
    let round_up = rest > midpoint || (rest == midpoint && kept & 1 == 1);
    // 올림으로 가수가 넘치면 지수가 하나 오른다. 비트를 이어 붙여 두었으므로
    // 덧셈 한 번으로 그렇게 된다. 가장 큰 값에서 넘치면 무한대가 된다.
    sign | (kept + u32::from(round_up)) as u16
}
// ANCHOR_END: f16_from_f32

// ANCHOR: bf16
/// BF16의 비트를 `f32`로 바꾼다. 아래 16비트를 0으로 채우면 된다(9.3절).
pub fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits(u32::from(half) << 16)
}

/// `f32`를 가장 가까운 BF16의 비트로 바꾼다. 한가운데이면 마지막 비트가
/// 0인 쪽을 고른다.
pub fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // 반올림하다가 NaN이 다른 값이 되지 않게 한다.
        return (bits >> 16) as u16 | 0x0040;
    }
    // 버릴 16비트가 한가운데(0x8000)를 넘으면 올린다. 정확히 한가운데이면
    // 남길 부분의 마지막 비트가 1일 때만 올린다.
    let last_kept = (bits >> 16) & 1;
    ((bits + 0x7fff + last_kept) >> 16) as u16
}
// ANCHOR_END: bf16

// ANCHOR: half_types
/// 16비트 실수(IEEE 754 binary16) 하나. 비트를 그대로 들고 있다.
///
/// Rust에 16비트 실수 형이 없어서 만든 형이다. 이 형으로는 계산하지 않는다.
/// 담아 두기만 하고, 계산할 때 [`to_f32`](Self::to_f32)로 바꾼다.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F16(pub u16);

/// BF16 실수 하나. `f32`의 위쪽 16비트를 그대로 들고 있다.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bf16(pub u16);

impl F16 {
    /// 가장 가까운 16비트 실수로 바꾼다.
    pub fn from_f32(value: f32) -> Self {
        Self(f16_from_f32(value))
    }

    /// 같은 값의 `f32`로 바꾼다.
    pub fn to_f32(self) -> f32 {
        f16_to_f32(self.0)
    }
}

impl Bf16 {
    /// 가장 가까운 BF16으로 바꾼다.
    pub fn from_f32(value: f32) -> Self {
        Self(bf16_from_f32(value))
    }

    /// 같은 값의 `f32`로 바꾼다.
    pub fn to_f32(self) -> f32 {
        bf16_to_f32(self.0)
    }
}
// ANCHOR_END: half_types

macro_rules! impl_half {
    ($type:ident) => {
        impl $type {
            /// 리틀 엔디언 바이트 둘에서 읽는다.
            pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
                Self(u16::from_le_bytes(bytes))
            }

            /// 리틀 엔디언 바이트 둘로 적는다.
            pub fn to_le_bytes(self) -> [u8; 2] {
                self.0.to_le_bytes()
            }
        }

        impl std::fmt::Debug for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self.to_f32())
            }
        }
    };
}
impl_half!(F16);
impl_half!(Bf16);

// ANCHOR: quantizer
/// 실수를 정수 눈금에 맞추는 규칙.
///
/// 실수 `x`를 정수 `q = round(x / scale) + zero_point`로 바꾸고, 정수를
/// `scale × (q - zero_point)`로 되돌린다. 정수는 `low..=high`를 벗어나지
/// 못한다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantizer {
    /// 스케일 한 칸의 크기.
    pub scale: f32,
    /// 실수 0에 대응하는 정수.
    pub zero_point: i32,
    /// 쓸 수 있는 가장 작은 정수.
    pub low: i32,
    /// 쓸 수 있는 가장 큰 정수.
    pub high: i32,
}

impl Quantizer {
    /// 0을 가운데 두는 눈금. 절댓값이 `limit`인 실수가 가장 큰 정수가 된다.
    ///
    /// `bits`비트이면 정수는 `-(2^(bits-1) - 1)..=2^(bits-1) - 1`이다. 8비트라면
    /// -127부터 127까지로, -128은 쓰지 않는다.
    pub fn symmetric(limit: f32, bits: u32) -> Self {
        let high = (1_i32 << (bits - 1)) - 1;
        Self {
            scale: positive(limit / high as f32),
            zero_point: 0,
            low: -high,
            high,
        }
    }

    /// 값들의 가장 큰 절댓값에 맞춘 대칭 눈금.
    pub fn symmetric_for(values: &[f32], bits: u32) -> Self {
        let limit = values
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        Self::symmetric(limit, bits)
    }

    /// `min..=max` 구간 전체에 정수 `0..=2^bits - 1`을 고르게 펴는 눈금.
    ///
    /// 구간이 0을 포함하도록 넓히고, 0이 정확히 한 정수에 대응하게 한다.
    pub fn asymmetric(min: f32, max: f32, bits: u32) -> Self {
        let (min, max) = (min.min(0.0), max.max(0.0));
        let high = ((1_i64 << bits) - 1) as i32;
        let scale = positive((max - min) / high as f32);
        Self {
            scale,
            zero_point: ((-min / scale).round_ties_even() as i32).clamp(0, high),
            low: 0,
            high,
        }
    }

    /// 값들의 최솟값과 최댓값에 맞춘 비대칭 눈금.
    pub fn asymmetric_for(values: &[f32], bits: u32) -> Self {
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if min > max {
            return Self::asymmetric(0.0, 0.0, bits);
        }
        Self::asymmetric(min, max, bits)
    }

    /// 실수를 정수로 바꾼다. 범위를 벗어나면 끝값으로 자른다.
    pub fn quantize(&self, value: f32) -> i32 {
        let code = (value / self.scale).round_ties_even() as i32 + self.zero_point;
        code.clamp(self.low, self.high)
    }

    /// 정수를 실수로 되돌린다.
    pub fn dequantize(&self, code: i32) -> f32 {
        (code - self.zero_point) as f32 * self.scale
    }

    /// 정수로 바꿨다가 되돌린 값. 원래 값과의 차이가 양자화 오차다.
    pub fn round_trip(&self, value: f32) -> f32 {
        self.dequantize(self.quantize(value))
    }
}

/// 스케일이 0이 되지 않게 한다. 값이 모두 0인 경우다.
fn positive(scale: f32) -> f32 {
    if scale > 0.0 && scale.is_finite() {
        scale
    } else {
        1.0
    }
}
// ANCHOR_END: quantizer

// ANCHOR: fake
/// 스케일을 어느 범위마다 따로 정할지.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// 값 전체에 스케일 하나.
    Tensor,
    /// 연달아 놓인 값 몇 개마다 스케일 하나. 행렬의 행마다 따로 정하려면 열의
    /// 수를, 작은 덩어리마다 따로 정하려면 덩어리의 크기를 넣는다.
    Blocks(usize),
}

/// 값들을 대칭 눈금의 정수로 바꿨다가 되돌려 놓는다.
///
/// 결과는 여전히 `f32`이지만 정수 `2^bits - 1`가지로 나타낼 수 있는 값만
/// 남는다. 저장 공간은 줄지 않는다. 양자화가 계산 결과를 얼마나 바꾸는지
/// 보려고 쓰는 방법이다.
pub fn fake_quantize(values: &mut [f32], bits: u32, granularity: Granularity) {
    let size = match granularity {
        Granularity::Tensor => values.len(),
        Granularity::Blocks(size) => size,
    };
    for block in values.chunks_mut(size.max(1)) {
        let quantizer = Quantizer::symmetric_for(block, bits);
        for value in block {
            *value = quantizer.round_trip(*value);
        }
    }
}
// ANCHOR_END: fake

// ANCHOR: sqnr
/// 신호 대 양자화 잡음비(SQNR)를 데시벨로 돌려준다.
///
/// 원래 값의 제곱의 합을 오차의 제곱의 합으로 나누고 `10 × log10`을 취한다.
/// 10dB가 오르면 오차의 크기(제곱의 합)가 10분의 1이 된 것이다. 오차가
/// 없으면 무한대다.
pub fn sqnr_db(original: &[f32], restored: &[f32]) -> f64 {
    assert_eq!(original.len(), restored.len(), "길이가 같아야 합니다");
    let mut signal = 0.0_f64;
    let mut noise = 0.0_f64;
    for (&a, &b) in original.iter().zip(restored) {
        signal += f64::from(a) * f64::from(a);
        let error = f64::from(a) - f64::from(b);
        noise += error * error;
    }
    10.0 * (signal / noise).log10()
}
// ANCHOR_END: sqnr

#[cfg(test)]
mod tests {
    use super::*;

    /// NumPy(`float16`)와 PyTorch(`bfloat16`)로 바꾼 결과. `f32`의 비트와
    /// 16비트 결과의 비트다.
    const F16_CASES: [(u32, u16); 25] = [
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        (0x3f800000, 0x3c00),
        (0xc0200000, 0xc100),
        (0x3dcccccd, 0x2e66),
        (0x40490fdb, 0x4248),
        (0x477fe000, 0x7bff),
        (0x477feffd, 0x7bff),
        (0x477ff000, 0x7c00),
        (0x47c35000, 0x7c00),
        (0xc7c35000, 0xfc00),
        (0x38800000, 0x0400),
        (0x387ba882, 0x03ef),
        (0x33800000, 0x0001),
        (0x33000000, 0x0000),
        (0x33000001, 0x0001),
        (0x2edbe6ff, 0x0000),
        (0x3f801000, 0x3c00),
        (0x3f801008, 0x3c01),
        (0x3f803000, 0x3c02),
        (0x3eaaaaab, 0x3555),
        (0x3727c5ac, 0x00a8),
        (0xba49d9d3, 0x924f),
        (0x45001000, 0x6800),
        (0x45003000, 0x6802),
    ];
    const BF16_CASES: [(u32, u16); 12] = [
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        (0x3f800000, 0x3f80),
        (0xc0200000, 0xc020),
        (0x3dcccccd, 0x3dcd),
        (0x40490fdb, 0x4049),
        (0x477feffd, 0x4780),
        (0x387ba882, 0x387c),
        (0x2edbe6ff, 0x2edc),
        (0x3f801008, 0x3f80),
        (0x3eaaaaab, 0x3eab),
        (0xba49d9d3, 0xba4a),
    ];

    // ANCHOR: f16_tests
    #[test]
    fn f16_matches_numpy() {
        for (bits, half) in F16_CASES {
            assert_eq!(f16_from_f32(f32::from_bits(bits)), half, "{bits:#010x}");
        }
    }

    /// 16비트 실수 65,536개를 모두 `f32`로 바꿨다가 되돌려 본다.
    #[test]
    fn every_f16_survives_a_round_trip() {
        let mut previous = f32::NEG_INFINITY;
        for half in 0..=u16::MAX {
            let value = f16_to_f32(half);
            if value.is_nan() {
                assert!(f16_to_f32(f16_from_f32(value)).is_nan());
                continue;
            }
            assert_eq!(f16_from_f32(value), half, "{half:#06x}");
            // 양수 쪽에서는 비트가 커지면 값도 커져야 한다.
            if half < 0x7c00 {
                assert!(value > previous || half == 0, "{half:#06x}");
                previous = value;
            }
        }
    }

    /// 이웃한 두 16비트 실수의 사이에서 가까운 쪽으로 가는지 본다.
    #[test]
    fn f16_rounds_to_the_nearest() {
        for half in 0..0x7bff_u16 {
            let (low, high) = (f16_to_f32(half), f16_to_f32(half + 1));
            let (low, high) = (f64::from(low), f64::from(high));
            let near_low = (low + (high - low) * 0.25) as f32;
            let near_high = (low + (high - low) * 0.75) as f32;
            let middle = (low + (high - low) * 0.5) as f32;
            assert_eq!(f16_from_f32(near_low), half);
            assert_eq!(f16_from_f32(near_high), half + 1);
            // 한가운데에서는 마지막 비트가 0인 쪽.
            let even = if half & 1 == 0 { half } else { half + 1 };
            assert_eq!(f16_from_f32(middle), even, "{half:#06x}");
            assert_eq!(f16_from_f32(-near_high), (half + 1) | 0x8000);
        }
    }
    // ANCHOR_END: f16_tests

    #[test]
    fn f16_handles_the_edges() {
        assert_eq!(f16_from_f32(f32::INFINITY), 0x7c00);
        assert_eq!(f16_from_f32(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_to_f32(f16_from_f32(f32::NAN)).is_nan());
        assert_eq!(f16_from_f32(f32::MAX), 0x7c00);
        assert_eq!(f16_from_f32(f32::MIN_POSITIVE), 0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2.0_f32.powi(-14));
    }

    #[test]
    fn half_types_wrap_the_conversions() {
        assert_eq!(F16::from_f32(1.0), F16(0x3c00));
        assert_eq!(F16(0xc100).to_f32(), -2.5);
        assert_eq!(Bf16::from_f32(-2.5), Bf16(0xc020));
        assert_eq!(Bf16::from_le_bytes([0x80, 0x3f]).to_f32(), 1.0);
        assert_eq!(F16(0x3c00).to_le_bytes(), [0x00, 0x3c]);
        assert_eq!(format!("{:?}", F16(0x3c00)), "1.0");
    }

    #[test]
    fn bf16_matches_pytorch_and_round_trips() {
        for (bits, half) in BF16_CASES {
            assert_eq!(bf16_from_f32(f32::from_bits(bits)), half, "{bits:#010x}");
        }
        for half in 0..=u16::MAX {
            let value = bf16_to_f32(half);
            if value.is_nan() {
                assert!(bf16_to_f32(bf16_from_f32(value)).is_nan());
            } else {
                assert_eq!(bf16_from_f32(value), half);
            }
        }
        // 한가운데: 1과 그다음 BF16(1 + 2^-7)의 사이.
        assert_eq!(bf16_from_f32(1.0 + 2.0_f32.powi(-8)), 0x3f80);
        assert_eq!(bf16_from_f32(1.0 + 3.0 * 2.0_f32.powi(-8)), 0x3f82);
        assert_eq!(bf16_from_f32(f32::MAX), 0x7f80);
    }

    // ANCHOR: quantizer_tests
    #[test]
    fn symmetric_quantizer_keeps_zero_and_bounds_the_error() {
        for bits in [2, 4, 8, 16] {
            let quantizer = Quantizer::symmetric(3.0, bits);
            let high = (1 << (bits - 1)) - 1;
            assert_eq!((quantizer.low, quantizer.high), (-high, high));
            assert_eq!(quantizer.round_trip(0.0), 0.0);
            assert_eq!(quantizer.quantize(3.0), high);
            assert_eq!(quantizer.quantize(-3.0), -high);
            // 범위 안에서는 오차가 눈금 반 칸을 넘지 않는다.
            for step in -300..=300 {
                let value = step as f32 * 0.01;
                let error = (quantizer.round_trip(value) - value).abs();
                assert!(
                    error <= quantizer.scale * 0.5 * 1.0001,
                    "{bits}비트, {value}"
                );
            }
            // 범위 밖은 끝값으로 잘린다.
            assert_eq!(quantizer.quantize(100.0), high);
        }
    }

    #[test]
    fn asymmetric_quantizer_uses_the_whole_range() {
        let quantizer = Quantizer::asymmetric(-1.0, 3.0, 8);
        assert_eq!((quantizer.low, quantizer.high), (0, 255));
        assert_eq!(quantizer.quantize(-1.0), 0);
        assert_eq!(quantizer.quantize(3.0), 255);
        // 0은 오차 없이 되돌아온다.
        assert_eq!(quantizer.round_trip(0.0), 0.0);
        // 한쪽으로 치우친 값: 대칭 눈금보다 칸이 절반 가까이 작다.
        let one_sided = Quantizer::asymmetric(0.0, 3.0, 8);
        assert!(one_sided.scale < Quantizer::symmetric(3.0, 8).scale * 0.51);
        // 0을 포함하지 않는 구간은 0까지 넓힌다.
        assert_eq!(Quantizer::asymmetric(1.0, 3.0, 8).zero_point, 0);
    }
    // ANCHOR_END: quantizer_tests

    #[test]
    fn fake_quantize_works_per_block() {
        // 큰 값 하나가 섞인 경우. 덩어리마다 스케일을 따로 정하면 그 값이 든
        // 덩어리만 거칠어진다.
        let original: Vec<f32> = (0..64)
            .map(|i| {
                if i == 5 {
                    100.0
                } else {
                    (i as f32 - 32.0) * 0.01
                }
            })
            .collect();
        let mut whole = original.clone();
        fake_quantize(&mut whole, 4, Granularity::Tensor);
        let mut blocks = original.clone();
        fake_quantize(&mut blocks, 4, Granularity::Blocks(32));
        assert!(whole.iter().filter(|&&value| value != 0.0).count() == 1);
        assert_eq!(blocks[5], 100.0);
        assert!(sqnr_db(&original[32..], &blocks[32..]) > 15.0);
        assert!(sqnr_db(&original, &original).is_infinite());
        // 값이 모두 0이어도 멈추지 않는다.
        let mut zeros = vec![0.0_f32; 8];
        fake_quantize(&mut zeros, 4, Granularity::Tensor);
        assert_eq!(zeros, vec![0.0; 8]);
        fake_quantize(&mut [], 4, Granularity::Tensor);
    }
}
