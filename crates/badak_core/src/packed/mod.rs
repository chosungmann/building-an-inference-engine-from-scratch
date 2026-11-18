//! 값 32개씩 묶어 정수로 적은 가중치와, 그것으로 하는 계산 (20장).
//!
//! 묶음 하나는 눈금의 크기(16비트 실수) 하나와 정수 32개다. 바이트의 생김새는
//! llama.cpp의 `Q8_0`, `Q4_0` 형식과 같다.

use std::cell::Cell;
use std::ops::Range;

use crate::error::{Error, Result};
use crate::quant::{Bf16, f16_from_f32, f16_to_f32};
use crate::tensor::{DType, Tensor};
use crate::{ops, simd, threads};

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// 묶음 하나에 든 값의 수.
pub const BLOCK: usize = 32;

// ANCHOR: format
/// 묶음의 형식.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 값 하나에 8비트. 정수는 -127부터 127까지.
    Q8_0,
    /// 값 하나에 4비트. 정수는 -8부터 7까지.
    Q4_0,
}

impl Format {
    /// 묶음 하나의 바이트 수. 눈금의 크기 2바이트에 값들이 이어진다.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 2 + BLOCK,
            Self::Q4_0 => 2 + BLOCK / 2,
        }
    }

    /// 값 하나에 드는 비트. 눈금의 크기의 몫을 더한 것이다.
    pub fn bits_per_weight(self) -> f64 {
        self.block_bytes() as f64 * 8.0 / BLOCK as f64
    }
}
// ANCHOR_END: format

// ANCHOR: quantize_q8_0
/// 값 32개를 `Q8_0` 묶음 하나(34바이트)로 적는다.
///
/// llama.cpp가 파일을 만들 때 쓰는 계산을 그대로 따른다. 같은 값을 넣으면
/// 같은 바이트가 나온다.
fn quantize_block_q8_0(values: &[f32], block: &mut [u8]) {
    let largest = values.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
    let scale = largest / 127.0;
    let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    block[..2].copy_from_slice(&f16_from_f32(scale).to_le_bytes());
    for (code, &value) in block[2..].iter_mut().zip(values) {
        // 한가운데이면 0에서 먼 쪽으로 간다(C의 `roundf`).
        *code = (value * inverse).round() as i8 as u8;
    }
}
// ANCHOR_END: quantize_q8_0

// ANCHOR: quantize_q4_0
/// 값 32개를 `Q4_0` 묶음 하나(18바이트)로 적는다.
///
/// 정수 16가지(-8부터 7까지)를 모두 쓴다. 절댓값이 가장 큰 값이 -8에 오도록
/// 눈금의 크기에 부호를 준다. 그 값이 양수이면 눈금의 크기가 음수가 된다.
fn quantize_block_q4_0(values: &[f32], block: &mut [u8]) {
    let mut largest = 0.0_f32;
    let mut signed = 0.0_f32;
    for &value in values {
        if largest < value.abs() {
            largest = value.abs();
            signed = value;
        }
    }
    let scale = signed / -8.0;
    let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    block[..2].copy_from_slice(&f16_from_f32(scale).to_le_bytes());
    // 8을 더해 0부터 15까지로 옮기고, 0.5를 더한 뒤 소수를 버려 반올림한다.
    let code = |value: f32| ((value * inverse + 8.5) as i8).min(15) as u8;
    // 바이트 하나에 값 둘: 아래 4비트에 앞 절반의 값, 위 4비트에 뒤 절반의 값.
    for (index, byte) in block[2..].iter_mut().enumerate() {
        *byte = code(values[index]) | (code(values[index + BLOCK / 2]) << 4);
    }
}
// ANCHOR_END: quantize_q4_0

// ANCHOR: dequantize
/// 묶음 하나를 `f32` 32개로 되돌린다.
fn dequantize_block(format: Format, block: &[u8], values: &mut [f32]) {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    match format {
        Format::Q8_0 => {
            for (value, &code) in values.iter_mut().zip(&block[2..]) {
                *value = f32::from(code as i8) * scale;
            }
        }
        Format::Q4_0 => {
            let (front, back) = values.split_at_mut(BLOCK / 2);
            for ((front, back), &byte) in front.iter_mut().zip(back).zip(&block[2..]) {
                *front = (i32::from(byte & 0x0f) - 8) as f32 * scale;
                *back = (i32::from(byte >> 4) - 8) as f32 * scale;
            }
        }
    }
}
// ANCHOR_END: dequantize

// ANCHOR: q4_k
/// llama.cpp의 `Q4_K` 묶음 하나의 바이트 수와 값의 수.
pub const Q4_K_BYTES: usize = 144;
/// `Q4_K` 묶음 하나에 든 값의 수. 32개짜리 작은 묶음 여덟 개다.
pub const Q4_K_VALUES: usize = 256;

/// `Q4_K`로 적힌 바이트를 `f32`로 되돌린다. 읽기만 하고 만들지는 않는다.
///
/// 묶음 하나(144바이트)의 생김새는 이렇다.
///
/// - 2바이트: 눈금의 크기들의 눈금(16비트 실수)
/// - 2바이트: 최솟값들의 눈금(16비트 실수)
/// - 12바이트: 작은 묶음 여덟 개 각각의 눈금의 크기와 최솟값. 6비트씩이다.
/// - 128바이트: 값 256개. 4비트씩이고 0부터 15까지다.
///
/// 값은 `눈금 × 정수 − 최솟값`으로 되돌린다. 0을 가운데 두지 않는 비대칭
/// 눈금이다(19.4절).
pub fn dequantize_q4_k(bytes: &[u8], values: &mut [f32]) {
    let blocks = bytes.as_chunks::<Q4_K_BYTES>().0;
    for (block, values) in blocks.iter().zip(values.as_chunks_mut::<Q4_K_VALUES>().0) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (packed, codes) = (&block[4..16], &block[16..]);
        // `index`번째 작은 묶음의 (눈금의 크기, 최솟값). 6비트 둘이다. 앞의 넷은
        // 바이트의 아래 6비트에 있고, 뒤의 넷은 남는 비트들에 흩어져 있다.
        let scale_and_minimum = |index: usize| -> (u8, u8) {
            if index < 4 {
                (packed[index] & 63, packed[index + 4] & 63)
            } else {
                (
                    (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
                    (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
                )
            }
        };
        // 작은 묶음 둘(값 64개)이 32바이트를 함께 쓴다. 아래 4비트가 앞의
        // 묶음, 위 4비트가 뒤의 묶음이다.
        let halves = values.as_chunks_mut::<64>().0.iter_mut();
        for (pair, (values, codes)) in halves.zip(codes.as_chunks::<32>().0).enumerate() {
            let (first, second) = values.split_at_mut(32);
            let (sc, m) = scale_and_minimum(2 * pair);
            let (d1, m1) = (scale * f32::from(sc), minimum * f32::from(m));
            let (sc, m) = scale_and_minimum(2 * pair + 1);
            let (d2, m2) = (scale * f32::from(sc), minimum * f32::from(m));
            for ((first, second), &byte) in first.iter_mut().zip(second).zip(codes) {
                *first = d1 * f32::from(byte & 0x0f) - m1;
                *second = d2 * f32::from(byte >> 4) - m2;
            }
        }
    }
}
// ANCHOR_END: q4_k

// ANCHOR: q6_k
/// llama.cpp의 `Q6_K` 묶음 하나의 바이트 수. 값은 `Q4_K`와 같이 256개다.
pub const Q6_K_BYTES: usize = 210;

/// `Q6_K`로 적힌 바이트를 `f32`로 되돌린다. 읽기만 하고 만들지는 않는다 (21장).
///
/// 값 하나가 6비트 정수(-32부터 31까지)다. 아래 4비트와 위 2비트를 따로
/// 모아 두었다. 눈금은 값 16개마다 8비트 정수 하나이고, 그 눈금들의 눈금이
/// 16비트 실수 하나다. 0을 가운데 두는 대칭 눈금이다. 값 하나에 6.56비트.
pub fn dequantize_q6_k(bytes: &[u8], values: &mut [f32]) {
    let blocks = bytes.as_chunks::<Q6_K_BYTES>().0;
    for (block, values) in blocks.iter().zip(values.as_chunks_mut::<Q4_K_VALUES>().0) {
        let (low, rest) = block.split_at(128);
        let (high, rest) = rest.split_at(64);
        let (scales, d) = rest.split_at(16);
        let d = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
        // 값 128개씩 둘로 나뉜다. 절반 하나가 아래 4비트 64바이트, 위 2비트
        // 32바이트, 눈금 8개를 쓴다.
        for half in 0..2 {
            let (low, high) = (&low[half * 64..], &high[half * 32..]);
            let scales = &scales[half * 8..];
            let values = &mut values[half * 128..];
            for l in 0..32 {
                let scale = |index: usize| d * f32::from(scales[l / 16 + index] as i8);
                // 6비트 정수 넷. 위 2비트 넷이 바이트 하나에 모여 있다.
                let code = |nibble: u8, shift: u8| -> f32 {
                    f32::from((nibble | (((high[l] >> shift) & 3) << 4)) as i8 - 32)
                };
                values[l] = scale(0) * code(low[l] & 0x0f, 0);
                values[l + 32] = scale(2) * code(low[l + 32] & 0x0f, 2);
                values[l + 64] = scale(4) * code(low[l] >> 4, 4);
                values[l + 96] = scale(6) * code(low[l + 32] >> 4, 6);
            }
        }
    }
}
// ANCHOR_END: q6_k

// ANCHOR: activations
/// 입력 벡터들을 묶음마다 8비트 정수로 바꾼 것.
///
/// 가중치와 달리 저장하지 않고 곱하기 직전에 만든다. 그래서 눈금의 크기를
/// 16비트로 줄이지 않고 `f32`로 둔다.
#[derive(Default)]
pub struct Activations {
    /// 묶음마다의 눈금의 크기.
    pub scales: Vec<f32>,
    /// 값마다의 정수. -127부터 127까지.
    pub codes: Vec<i8>,
}

impl Activations {
    /// `f32` 값들을 32개씩 묶어 정수로 바꾼다. 길이는 32의 배수여야 한다.
    ///
    /// 묶음마다 가장 큰 절댓값을 127에 맞춘다. 한가운데는 짝수 쪽으로 간다.
    pub fn quantize(x: &[f32]) -> Self {
        let mut activations = Self::default();
        activations.fill(x);
        activations
    }

    /// [`quantize`](Self::quantize)와 같되, 이미 가진 자리에 다시 적는다 (22장).
    ///
    /// 벡터가 지난번보다 길지 않으면 힙에서 자리를 새로 얻지 않는다.
    pub fn fill(&mut self, x: &[f32]) {
        assert_eq!(x.len() % BLOCK, 0, "길이가 32의 배수여야 합니다");
        let Self { scales, codes } = self;
        scales.clear();
        scales.reserve(x.len() / BLOCK);
        // 길이만 맞춘다. 아래에서 모든 자리를 다시 적으므로 지울 필요가 없다.
        codes.resize(x.len(), 0);
        let blocks = x.as_chunks::<BLOCK>().0.iter();
        for (values, codes) in blocks.zip(codes.as_chunks_mut::<BLOCK>().0) {
            let largest = values.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
            let scale = largest / 127.0;
            let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            for (code, &value) in codes.iter_mut().zip(values) {
                *code = (value * inverse).round_ties_even() as i8;
            }
            scales.push(scale);
        }
    }
}
// ANCHOR_END: activations

// ANCHOR: dot_q8_0
/// `Q8_0`으로 적은 가중치 한 행과 정수로 바꾼 입력 벡터 하나의 내적.
///
/// 이 함수가 계산을 정의한다.
///
/// 1. 묶음마다 정수 32쌍을 곱해 **정수로** 더한다.
/// 2. 그 합을 `f32`로 바꾸고, 두 눈금의 크기의 곱을 곱해, `f32` 합에
///    더한다. 눈금끼리의 곱, 합과의 곱, 더하기를 각각 따로 반올림한다.
///
/// 정수의 덧셈은 순서를 바꿔도 결과가 같다. 그래서 1번을 어떤 벡터 명령으로
/// 계산하든 같은 값이 나오고, 구현끼리 맞춰야 하는 것은 2번뿐이다.
pub fn dot_q8_0(row: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let blocks = row.as_chunks::<{ Format::Q8_0.block_bytes() }>().0;
    let codes = codes.as_chunks::<BLOCK>().0;
    for ((block, codes), &x_scale) in blocks.iter().zip(codes).zip(scales) {
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut products = 0_i32;
        for (&w, &x) in block[2..].iter().zip(codes) {
            products += i32::from(w as i8) * i32::from(x);
        }
        sum += products as f32 * (w_scale * x_scale);
    }
    sum
}
// ANCHOR_END: dot_q8_0

// ANCHOR: dot_q4_0
/// `Q4_0`으로 적은 가중치 한 행과 정수로 바꾼 입력 벡터 하나의 내적.
///
/// 바이트 하나에서 4비트 값 둘을 꺼내는 것만 [`dot_q8_0`]과 다르다. 아래
/// 4비트가 앞 16개의 값이고 위 4비트가 뒤 16개의 값이다.
pub fn dot_q4_0(row: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let blocks = row.as_chunks::<{ Format::Q4_0.block_bytes() }>().0;
    let codes = codes.as_chunks::<BLOCK>().0;
    for ((block, codes), &x_scale) in blocks.iter().zip(codes).zip(scales) {
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let (front, back) = codes.split_at(BLOCK / 2);
        let mut products = 0_i32;
        for ((&byte, &x_front), &x_back) in block[2..].iter().zip(front).zip(back) {
            products += (i32::from(byte & 0x0f) - 8) * i32::from(x_front);
            products += (i32::from(byte >> 4) - 8) * i32::from(x_back);
        }
        sum += products as f32 * (w_scale * x_scale);
    }
    sum
}
// ANCHOR_END: dot_q4_0

// ANCHOR: kernel
/// 어느 구현으로 계산할지.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// 이 기계에서 쓸 수 있는 가장 빠른 구현.
    Fast,
    /// 벡터 명령을 직접 쓰지 않는 기준 구현.
    Portable,
}

/// 가중치 한 행과 정수로 바꾼 입력 벡터 하나의 내적을 계산하는 함수.
pub type Dot = fn(&[u8], &[i8], &[f32]) -> f32;

/// 이 기계에서 쓸 수 있는 가장 빠른 내적. 어느 것을 고르든 결과의 비트가
/// 같다.
pub fn fastest_dot(format: Format) -> Dot {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod")
        && std::arch::is_aarch64_feature_detected!("fp16")
    {
        // SAFETY: 바로 위에서 이 프로세서가 `dotprod`와 `fp16`을 지원함을 확인했다.
        return match format {
            Format::Q8_0 => |row, codes, scales| unsafe { neon::one::<false>(row, codes, scales) },
            Format::Q4_0 => |row, codes, scales| unsafe { neon::one::<true>(row, codes, scales) },
        };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: 바로 위에서 이 프로세서가 AVX2를 지원함을 확인했다.
        return match format {
            Format::Q8_0 => |row, codes, scales| unsafe { avx2::dot_q8_0(row, codes, scales) },
            Format::Q4_0 => |row, codes, scales| unsafe { avx2::dot_q4_0(row, codes, scales) },
        };
    }
    portable_dot(format)
}

/// 벡터 명령을 직접 쓰지 않는 기준 구현.
pub fn portable_dot(format: Format) -> Dot {
    match format {
        Format::Q8_0 => dot_q8_0,
        Format::Q4_0 => dot_q4_0,
    }
}
// ANCHOR_END: kernel

// ANCHOR: packed
/// 묶음으로 적은 가중치 행렬. `(rows, cols)`이고 행마다 묶음이 `cols ÷ 32`개다.
#[derive(Debug, Clone)]
pub struct Packed {
    format: Format,
    /// `[rows, cols]`.
    shape: [usize; 2],
    /// 묶음들의 바이트. `u8` 텐서에 담는다. 텐서는 직접 쥔 메모리일 수도,
    /// 파일을 비춘 메모리를 빌린 것일 수도 있다(22장).
    storage: Tensor,
}

impl Packed {
    /// `f32` 행렬을 묶음으로 적는다. 열의 수가 32의 배수여야 한다.
    pub fn quantize(format: Format, values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if values.len() != rows * cols || !cols.is_multiple_of(BLOCK) {
            // 열의 수가 32의 배수가 아니면 행을 묶음으로 나눌 수 없다.
            return Err(Error::Shape {
                operation: "묶음으로 양자화",
                left: vec![rows, cols],
                right: vec![BLOCK],
            });
        }
        let size = format.block_bytes();
        let mut bytes = vec![0_u8; values.len() / BLOCK * size];
        // 묶음마다 따로 계산하므로 18장의 풀에 나눈다.
        let threads = threads::global();
        let pool = threads.pool();
        let piece = 4096;
        pool.run_chunks(&mut bytes, piece * size, pool.threads(), &|index, bytes| {
            let values = &values[index * piece * BLOCK..];
            let values = values.as_chunks::<BLOCK>().0;
            for (block, values) in bytes.chunks_exact_mut(size).zip(values) {
                match format {
                    Format::Q8_0 => quantize_block_q8_0(values, block),
                    Format::Q4_0 => quantize_block_q4_0(values, block),
                }
            }
        });
        let storage = Tensor::from_vec(bytes, &[rows, cols / BLOCK * format.block_bytes()])?;
        Ok(Self {
            format,
            shape: [rows, cols],
            storage,
        })
    }

    /// 이미 묶음으로 적힌 바이트를 행렬로 받아들인다.
    pub fn from_bytes(format: Format, rows: usize, cols: usize, bytes: Vec<u8>) -> Result<Self> {
        let expected = rows * cols / BLOCK * format.block_bytes();
        if !cols.is_multiple_of(BLOCK) || bytes.len() != expected {
            return Err(Error::ElementCount {
                expected,
                actual: bytes.len(),
            });
        }
        let storage = Tensor::from_vec(bytes, &[rows, cols / BLOCK * format.block_bytes()])?;
        Ok(Self {
            format,
            shape: [rows, cols],
            storage,
        })
    }

    // ANCHOR: from_tensor
    /// 묶음의 바이트가 든 `u8` 텐서를 행렬로 받아들인다 (22장).
    ///
    /// 텐서의 형상은 `(행의 수, 행 하나의 바이트 수)`여야 한다. 텐서가
    /// 파일을 비춘 메모리를 빌린 것이면 이 행렬도 그 메모리를 그대로 쓴다.
    /// 복사가 없다.
    pub fn from_tensor(format: Format, storage: Tensor) -> Result<Self> {
        let &[rows, row_bytes] = storage.shape() else {
            return Err(Error::Shape {
                operation: "묶음의 바이트",
                left: storage.shape().to_vec(),
                right: vec![format.block_bytes()],
            });
        };
        if !row_bytes.is_multiple_of(format.block_bytes()) {
            return Err(Error::ElementCount {
                expected: row_bytes.next_multiple_of(format.block_bytes()),
                actual: row_bytes,
            });
        }
        // 자료형이 `u8`이고 연속인지 여기서 확인해 둔다.
        storage.as_slice::<u8>()?;
        Ok(Self {
            format,
            shape: [rows, row_bytes / format.block_bytes() * BLOCK],
            storage,
        })
    }

    /// 묶음의 바이트가 든 텐서. 파일에 적을 때 쓴다.
    pub fn storage(&self) -> &Tensor {
        &self.storage
    }
    // ANCHOR_END: from_tensor

    /// 묶음의 형식.
    pub fn format(&self) -> Format {
        self.format
    }

    /// 형상 `[rows, cols]`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 묶음들의 바이트.
    pub fn bytes(&self) -> &[u8] {
        self.storage
            .as_slice::<u8>()
            .expect("만들 때 연속인 u8 텐서임을 확인했다")
    }

    /// 행 하나가 차지하는 바이트 수.
    fn row_bytes(&self) -> usize {
        self.shape[1] / BLOCK * self.format.block_bytes()
    }

    /// 행 `range`를 `f32`로 되돌린다.
    pub fn dequantize_rows(&self, range: Range<usize>) -> Vec<f32> {
        let mut values = vec![0.0_f32; range.len() * self.shape[1]];
        self.dequantize_rows_into(range, &mut values);
        values
    }

    /// 행 `range`를 `f32`로 되돌려 `values`에 적는다.
    pub fn dequantize_rows_into(&self, range: Range<usize>, values: &mut [f32]) {
        assert_eq!(values.len(), range.len() * self.shape[1]);
        let size = self.format.block_bytes();
        let bytes = &self.bytes()[range.start * self.row_bytes()..range.end * self.row_bytes()];
        let blocks = bytes.chunks_exact(size);
        for (block, values) in blocks.zip(values.as_chunks_mut::<BLOCK>().0) {
            dequantize_block(self.format, block, values);
        }
    }
    // ANCHOR_END: packed

    // ANCHOR: packed_linear
    /// 입력 벡터 `count`개 각각에 이 행렬을 곱한다. `x`는 `(count, cols)`,
    /// `y`는 `(count, rows)`다.
    pub fn linear(&self, x: &[f32], count: usize, y: &mut [f32]) {
        let threads = threads::global().for_vectors(count);
        self.linear_with(Kernel::Fast, threads, x, count, y);
    }

    /// [`linear`](Self::linear)와 같되, 구현과 스레드의 수를 직접 정한다.
    pub fn linear_with(
        &self,
        kernel: Kernel,
        threads: usize,
        x: &[f32],
        count: usize,
        y: &mut [f32],
    ) {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        assert_eq!(x.len(), count * cols, "입력의 크기가 맞지 않습니다");
        assert_eq!(y.len(), count * rows, "출력의 크기가 맞지 않습니다");
        // 1. 입력을 정수로 바꾼다. 벡터마다 한 번이면 되고, 모든 행이 함께 쓴다.
        // 22장: 정수를 담을 자리는 스레드마다 하나를 두고 돌려 쓴다.
        thread_local! {
            static INPUT: Cell<Activations> = Cell::default();
        }
        let mut input = INPUT.take();
        input.fill(x);

        // 2. 행의 구간 하나를 모든 벡터와 곱한다.
        let piece = |range: Range<usize>, output: &mut [f32]| {
            let bytes = &self.bytes()[range.start * self.row_bytes()..range.end * self.row_bytes()];
            self.piece(kernel, bytes, range.len(), &input, count, output);
        };

        // 3. 18장의 방법 그대로 행을 스레드에 나눈다.
        if threads <= 1 || count * rows * cols < simd::MIN_PARALLEL_WORK {
            piece(0..rows, y);
        } else {
            let pool = threads::global().pool();
            let pieces = threads * simd::PIECES_PER_THREAD;
            simd::split_rows(pool, threads, pieces, count, rows, y, &piece);
        }
        // 다음 호출이 다시 쓰도록 돌려놓는다.
        INPUT.set(input);
    }

    /// 행 `width`개(`bytes`)를 입력 벡터 `count`개와 곱해 `(count, width)`
    /// 모양의 `output`을 채운다.
    fn piece(
        &self,
        kernel: Kernel,
        bytes: &[u8],
        width: usize,
        input: &Activations,
        count: usize,
        output: &mut [f32],
    ) {
        #[cfg(target_arch = "aarch64")]
        if kernel == Kernel::Fast
            && std::arch::is_aarch64_feature_detected!("dotprod")
            && std::arch::is_aarch64_feature_detected!("fp16")
        {
            let (codes, scales) = (&input.codes[..], &input.scales[..]);
            // SAFETY: 바로 위에서 이 프로세서가 `dotprod`와 `fp16`을 지원함을
            // 확인했다.
            unsafe {
                match self.format {
                    Format::Q8_0 => {
                        neon::piece::<false>(bytes, width, codes, scales, count, output)
                    }
                    Format::Q4_0 => neon::piece::<true>(bytes, width, codes, scales, count, output),
                }
            }
            return;
        }
        // 내적을 하나씩 계산한다. 행 하나가 캐시에 머무는 동안 벡터들이
        // 지나간다.
        let dot = match kernel {
            Kernel::Fast => fastest_dot(self.format),
            Kernel::Portable => portable_dot(self.format),
        };
        let (cols, blocks) = (self.shape[1], self.shape[1] / BLOCK);
        for (offset, row) in bytes.chunks_exact(self.row_bytes()).enumerate() {
            for vector in 0..count {
                let codes = &input.codes[vector * cols..][..cols];
                let scales = &input.scales[vector * blocks..][..blocks];
                output[vector * width + offset] = dot(row, codes, scales);
            }
        }
    }
    // ANCHOR_END: packed_linear
}

// ANCHOR: matrix
/// 선형 층의 가중치 행렬. 저장된 방식이 여럿이다.
#[derive(Debug, Clone)]
pub enum Matrix {
    /// 값마다 따로 적은 것. 텐서의 자료형이 `f32`이거나 BF16이다.
    Dense(Tensor),
    /// 값 32개씩 묶어 정수로 적은 것.
    Packed(Packed),
}

impl Matrix {
    /// 형상 `[rows, cols]`.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(tensor) => tensor.shape(),
            Self::Packed(packed) => packed.shape(),
        }
    }

    /// 값의 수.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// 값이 하나도 없는가.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 메모리에서 차지하는 바이트 수.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::Dense(tensor) => tensor.size_in_bytes(),
            Self::Packed(packed) => packed.bytes().len(),
        }
    }

    /// 값을 `f32` 텐서로 꺼낸다. 양자화된 값은 되돌려서 꺼낸다.
    pub fn to_f32(&self) -> Result<Tensor> {
        match self {
            Self::Dense(tensor) => ops::to_f32(tensor),
            Self::Packed(packed) => {
                Tensor::from_vec(packed.dequantize_rows(0..packed.shape[0]), packed.shape())
            }
        }
    }

    /// 표로 보고, 번호에 해당하는 행들을 `f32`로 뽑는다(임베딩).
    pub fn lookup(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(tensor) => ops::embedding(tensor, ids),
            Self::Packed(packed) => {
                let width = packed.shape[1];
                let mut shape = crate::Dims::from(ids.shape());
                shape.push(width);
                Tensor::build(&shape, |data| {
                    let mut outputs = data.chunks_exact_mut(width.max(1));
                    for id in ids.iter::<i32>()? {
                        let row = usize::try_from(id)
                            .ok()
                            .filter(|&row| row < packed.shape[0])
                            .ok_or(Error::OutOfBounds {
                                axis: 0,
                                index: id.max(0) as usize,
                                len: packed.shape[0],
                            })?;
                        if let Some(output) = outputs.next() {
                            packed.dequantize_rows_into(row..row + 1, output);
                        }
                    }
                    Ok(())
                })
            }
        }
    }
}

impl From<Tensor> for Matrix {
    fn from(tensor: Tensor) -> Self {
        Self::Dense(tensor)
    }
}

impl From<Packed> for Matrix {
    fn from(packed: Packed) -> Self {
        Self::Packed(packed)
    }
}
// ANCHOR_END: matrix

// ANCHOR: linear_weight
impl ops::LinearWeight for Tensor {
    fn dimensions(&self) -> Option<(usize, usize)> {
        match *self.shape() {
            [rows, cols] => Some((rows, cols)),
            _ => None,
        }
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> Result<()> {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let weight = self.contiguous();
        match weight.dtype() {
            // BF16으로 저장된 가중치는 펼치지 않고 그대로 커널에 넘긴다.
            DType::Bf16 => simd::linear(x, count, weight.as_slice::<Bf16>()?, rows, cols, y),
            _ => simd::linear(x, count, weight.as_slice::<f32>()?, rows, cols, y),
        }
        Ok(())
    }
}

impl ops::LinearWeight for Packed {
    fn dimensions(&self) -> Option<(usize, usize)> {
        Some((self.shape[0], self.shape[1]))
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> Result<()> {
        self.linear(x, count, y);
        Ok(())
    }
}

impl ops::LinearWeight for Matrix {
    fn dimensions(&self) -> Option<(usize, usize)> {
        match self {
            Self::Dense(tensor) => tensor.dimensions(),
            Self::Packed(packed) => packed.dimensions(),
        }
    }

    fn apply(&self, x: &[f32], count: usize, y: &mut [f32]) -> Result<()> {
        match self {
            Self::Dense(tensor) => tensor.apply(x, count, y),
            Self::Packed(packed) => packed.apply(x, count, y),
        }
    }
}
// ANCHOR_END: linear_weight

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Pcg64;

    fn values(count: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64::new(seed);
        (0..count)
            .map(|_| ((rng.next_f64() - 0.5) * (rng.next_f64() * 4.0)) as f32)
            .collect()
    }

    #[test]
    fn formats_have_the_documented_sizes() {
        assert_eq!(Format::Q8_0.block_bytes(), 34);
        assert_eq!(Format::Q4_0.block_bytes(), 18);
        assert_eq!(Format::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(Format::Q4_0.bits_per_weight(), 4.5);
    }

    // ANCHOR: block_tests
    #[test]
    fn q8_0_block_keeps_the_largest_value_and_bounds_the_error() {
        let original = values(BLOCK, 1);
        let mut block = [0_u8; 34];
        quantize_block_q8_0(&original, &mut block);
        let mut restored = [0.0_f32; BLOCK];
        dequantize_block(Format::Q8_0, &block, &mut restored);
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // 가장 큰 절댓값이 127이나 -127이 된다.
        assert!(block[2..].iter().any(|&code| (code as i8).abs() == 127));
        for (a, b) in original.iter().zip(restored) {
            // 눈금 반 칸에, 눈금의 크기를 16비트로 줄인 몫을 조금 더한 것 안이다.
            assert!((a - b).abs() <= scale * 0.5 * 1.01, "{a} {b}");
        }
    }

    #[test]
    fn q4_0_block_uses_minus_eight_for_the_largest_value() {
        for sign in [1.0_f32, -1.0] {
            let mut original = values(BLOCK, 2);
            original[7] = 9.0 * sign;
            let mut block = [0_u8; 18];
            quantize_block_q4_0(&original, &mut block);
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            // 가장 큰 값이 양수이면 눈금의 크기가 음수다.
            assert_eq!(scale, -9.0 * sign / 8.0);
            // 7번째 값은 7번째 바이트의 아래 4비트에 있고, -8(저장된 값 0)이다.
            assert_eq!(block[2 + 7] & 0x0f, 0);
            let mut restored = [0.0_f32; BLOCK];
            dequantize_block(Format::Q4_0, &block, &mut restored);
            assert_eq!(restored[7], 9.0 * sign);
            for (a, b) in original.iter().zip(restored) {
                assert!((a - b).abs() <= scale.abs() * 0.5 * 1.01, "{a} {b}");
            }
        }
    }

    #[test]
    fn a_block_of_zeros_stays_zero() {
        for format in [Format::Q8_0, Format::Q4_0] {
            let packed = Packed::quantize(format, &[0.0; 64], 2, 32).unwrap();
            assert!(packed.dequantize_rows(0..2).iter().all(|&v| v == 0.0));
        }
    }
    // ANCHOR_END: block_tests

    // ANCHOR: same_bits
    /// 벡터 명령으로 짠 내적이 기준 구현과 같은 비트를 내는지 본다.
    #[test]
    fn fast_dots_give_the_same_bits_as_the_portable_ones() {
        for format in [Format::Q8_0, Format::Q4_0] {
            for cols in [32, 64, 576, 2048] {
                for seed in 0..8 {
                    let packed = Packed::quantize(format, &values(cols, seed), 1, cols).unwrap();
                    let input = Activations::quantize(&values(cols, seed + 100));
                    let expected =
                        portable_dot(format)(packed.bytes(), &input.codes, &input.scales);
                    let got = fastest_dot(format)(packed.bytes(), &input.codes, &input.scales);
                    assert_eq!(got.to_bits(), expected.to_bits(), "{format:?}, 열 {cols}개");
                }
            }
        }
    }

    /// 정수가 양쪽 끝값일 때도 같은지 본다. 곱이 가장 커지는 경우다.
    #[test]
    fn extreme_codes_do_not_overflow() {
        for format in [Format::Q8_0, Format::Q4_0] {
            for (w, x) in [(1.0_f32, 1.0_f32), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                let packed = Packed::quantize(format, &[w; 64], 1, 64).unwrap();
                let input = Activations::quantize(&[x; 64]);
                let expected = portable_dot(format)(packed.bytes(), &input.codes, &input.scales);
                let got = fastest_dot(format)(packed.bytes(), &input.codes, &input.scales);
                assert_eq!(got.to_bits(), expected.to_bits());
                assert!(
                    (expected - 64.0 * w * x).abs() < 0.5,
                    "{format:?} {w} {x}: {expected}"
                );
            }
        }
    }
    // ANCHOR_END: same_bits

    #[test]
    fn linear_matches_the_dequantized_weights_closely() {
        let (rows, cols, count) = (40, 64, 3);
        let weights = values(rows * cols, 5);
        let x = values(count * cols, 6);
        for format in [Format::Q8_0, Format::Q4_0] {
            let packed = Packed::quantize(format, &weights, rows, cols).unwrap();
            let mut y = vec![0.0; count * rows];
            packed.linear(&x, count, &mut y);
            // 되돌린 가중치와 f32 입력으로 계산한 값. 입력을 8비트로 바꾼
            // 만큼만 다르다.
            let restored = packed.dequantize_rows(0..rows);
            let mut exact = vec![0.0; count * rows];
            simd::linear(&x, count, &restored, rows, cols, &mut exact);
            let largest = exact.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
            for (a, b) in y.iter().zip(&exact) {
                assert!((a - b).abs() < largest * 0.02, "{format:?}: {a} {b}");
            }
        }
    }

    #[test]
    fn threads_do_not_change_the_bits() {
        // 나눌 만큼 큰 행렬. 벡터가 하나일 때와 여럿일 때.
        let (rows, cols) = (301, 1024);
        let weights = values(rows * cols, 7);
        for format in [Format::Q8_0, Format::Q4_0] {
            let packed = Packed::quantize(format, &weights, rows, cols).unwrap();
            for count in [1, 2, 5] {
                let x = values(count * cols, 8);
                let input = Activations::quantize(&x);
                let mut y = vec![f32::NAN; count * rows];
                packed.linear(&x, count, &mut y);
                // 스레드 하나로, 기준 구현으로 계산한 것과도 같아야 한다.
                let mut alone = vec![f32::NAN; count * rows];
                packed.linear_with(Kernel::Portable, 1, &x, count, &mut alone);
                assert!(
                    y.iter()
                        .zip(&alone)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                for vector in 0..count {
                    for row in 0..rows {
                        let bytes =
                            &packed.bytes()[row * packed.row_bytes()..][..packed.row_bytes()];
                        let codes = &input.codes[vector * cols..][..cols];
                        let scales = &input.scales[vector * cols / BLOCK..][..cols / BLOCK];
                        let expected = portable_dot(format)(bytes, codes, scales);
                        assert_eq!(y[vector * rows + row].to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn q4_k_blocks_decode_by_the_documented_layout() {
        let mut block = [0_u8; Q4_K_BYTES];
        block[..2].copy_from_slice(&f16_from_f32(0.5).to_le_bytes());
        block[2..4].copy_from_slice(&f16_from_f32(0.25).to_le_bytes());
        // 작은 묶음 0: 눈금 3, 최솟값 2. 작은 묶음 1: 눈금 5, 최솟값 7.
        (block[4], block[8]) = (3, 2);
        (block[5], block[9]) = (5, 7);
        // 작은 묶음 5: 눈금 40(= 8 + 2 × 16), 최솟값 21(= 5 + 1 × 16).
        block[4 + 9] = 8 | (5 << 4);
        block[4 + 1] |= 2 << 6;
        block[4 + 5] |= 1 << 6;
        // 값: 묶음 0의 첫 값은 9, 묶음 1의 첫 값은 4, 묶음 5의 둘째 값은 15.
        block[16] = 9 | (4 << 4);
        block[16 + 2 * 32 + 1] = 15 << 4;
        let mut values = [f32::NAN; Q4_K_VALUES];
        dequantize_q4_k(&block, &mut values);
        assert_eq!(values[0], 0.5 * 3.0 * 9.0 - 0.25 * 2.0);
        assert_eq!(values[32], 0.5 * 5.0 * 4.0 - 0.25 * 7.0);
        assert_eq!(values[5 * 32 + 1], 0.5 * 40.0 * 15.0 - 0.25 * 21.0);
        // 정수가 0인 값은 최솟값의 반대 부호다.
        assert_eq!(values[1], -0.5);
    }

    #[test]
    fn q6_k_blocks_decode_by_the_documented_layout() {
        let mut block = [0_u8; Q6_K_BYTES];
        block[208..].copy_from_slice(&f16_from_f32(0.5).to_le_bytes());
        // 눈금: 0번(값 0~15) 3, 2번(값 32~47) -2, 13번(값 208~223) 5.
        block[192] = 3;
        block[192 + 2] = (-2_i8) as u8;
        block[192 + 13] = 5;
        // 값 0: 정수 45 - 32 = 13. 아래 4비트 13, 위 2비트 2.
        block[0] = 13;
        block[128] = 2;
        // 값 33: 정수 7 - 32 = -25. 아래 4비트 7, 위 2비트 0.
        block[33] = 7;
        // 값 208(뒤 절반의 80번째 = 64 + 16): 정수 63 - 32 = 31.
        // 아래 4비트는 뒤 절반의 16번째 바이트의 위쪽, 위 2비트는 4~5번 비트.
        block[64 + 16] = 0x0f << 4;
        block[128 + 32 + 16] = 3 << 4;
        let mut values = [f32::NAN; Q4_K_VALUES];
        dequantize_q6_k(&block, &mut values);
        assert_eq!(values[0], 0.5 * 3.0 * 13.0);
        assert_eq!(values[33], 0.5 * -2.0 * -25.0);
        assert_eq!(values[208], 0.5 * 5.0 * 31.0);
        // 눈금이 0인 곳은 0이다.
        assert_eq!(values[100], 0.0);
    }

    #[test]
    fn a_packed_matrix_can_borrow_its_bytes_from_a_tensor() {
        let (rows, cols) = (6, 64);
        let weights = values(rows * cols, 11);
        let x = values(cols, 12);
        for format in [Format::Q8_0, Format::Q4_0] {
            let owned = Packed::quantize(format, &weights, rows, cols).unwrap();
            // 바이트를 `u8` 텐서로 넘겼다가 다시 행렬로 받는다. 파일에 적었다가
            // 비춰 읽을 때 지나는 길이다.
            let borrowed = Packed::from_tensor(format, owned.storage().clone()).unwrap();
            assert_eq!(borrowed.shape(), [rows, cols]);
            assert_eq!(borrowed.bytes(), owned.bytes());
            let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
            owned.linear(&x, 1, &mut a);
            borrowed.linear(&x, 1, &mut b);
            assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // 행 하나의 바이트 수가 묶음의 배수가 아니면 받지 않는다.
        let odd = Tensor::from_vec(vec![0_u8; 70], &[2, 35]).unwrap();
        assert!(Packed::from_tensor(Format::Q8_0, odd).is_err());
        let flat = Tensor::from_vec(vec![0_u8; 68], &[68]).unwrap();
        assert!(Packed::from_tensor(Format::Q8_0, flat).is_err());
    }

    #[test]
    fn matrix_looks_up_rows_and_converts() {
        let weights = values(4 * 32, 9);
        let ids = Tensor::from_vec(vec![3_i32, 0], &[2]).unwrap();
        let dense = Matrix::from(Tensor::from_vec(weights.clone(), &[4, 32]).unwrap());
        assert_eq!(dense.shape(), [4, 32]);
        assert_eq!(dense.lookup(&ids).unwrap().shape(), [2, 32]);
        let packed = Matrix::from(Packed::quantize(Format::Q8_0, &weights, 4, 32).unwrap());
        assert_eq!((packed.len(), packed.size_in_bytes()), (128, 4 * 34));
        let rows = packed.lookup(&ids).unwrap().to_vec::<f32>().unwrap();
        let all = packed.to_f32().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(rows[..32], all[96..]);
        assert_eq!(rows[32..], all[..32]);
        assert!(Packed::quantize(Format::Q8_0, &weights, 4, 31).is_err());
        assert!(Packed::from_bytes(Format::Q4_0, 4, 32, vec![0; 10]).is_err());
    }
}
