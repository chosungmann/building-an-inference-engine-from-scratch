//! 정수 내적을 x86의 AVX2로 짠 것 (20장).
//!
//! AVX2에는 부호 있는 8비트끼리 곱하는 명령이 없다. "부호 없는 8비트 ×
//! 부호 있는 8비트"를 곱해 이웃한 둘을 더하는 `_mm256_maddubs_epi16`을
//! 쓴다. 한쪽의 부호를 떼어 다른 쪽에 옮겨 두면 곱은 같다. 정수 계산이라
//! 결과가 [`super::dot_q8_0`], [`super::dot_q4_0`]과 정확히 같다.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32, _mm_loadu_si128,
    _mm_set1_epi8, _mm_shuffle_epi32, _mm_srli_epi16, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_sign_epi8, _mm256_sub_epi8,
};

use super::{BLOCK, Format};
use crate::quant::f16_to_f32;

/// 8비트 정수 32쌍의 곱의 합.
///
/// 곱 둘의 합이 16비트를 넘지 않아야 한다. 값이 -127부터 127까지이면
/// `2 × 127 × 127 = 32258`이라 넘지 않는다.
#[inline]
#[target_feature(enable = "avx2")]
fn products(w: __m256i, x: __m256i) -> i32 {
    // w의 부호를 떼고(|w|), 그 부호를 x에 옮긴다. |w| × (부호 옮긴 x) = w × x.
    let magnitude = _mm256_sign_epi8(w, w);
    let signed = _mm256_sign_epi8(x, w);
    // 8비트 곱을 이웃한 둘씩 더해 16비트 16개로, 다시 둘씩 더해 32비트 8개로.
    let pairs = _mm256_maddubs_epi16(magnitude, signed);
    let fours = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    // 32비트 여덟을 하나로 더한다.
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(fours),
        _mm256_extracti128_si256::<1>(fours),
    );
    let twos = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b00_00_11_10>(halves));
    let one = _mm_add_epi32(twos, _mm_shuffle_epi32::<0b00_00_00_01>(twos));
    _mm_cvtsi128_si32(one)
}

/// [`super::dot_q8_0`]과 같은 계산.
///
/// # Safety
///
/// 프로세서가 AVX2를 지원해야 한다.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q8_0(row: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let blocks = row.as_chunks::<{ Format::Q8_0.block_bytes() }>().0;
    let codes = codes.as_chunks::<BLOCK>().0;
    for ((block, codes), &x_scale) in blocks.iter().zip(codes).zip(scales) {
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: `block`은 34바이트이고 값은 2번째 바이트부터 32개다.
        // `codes`는 32개다.
        let (w, x) = unsafe {
            (
                _mm256_loadu_si256(block.as_ptr().add(2).cast::<__m256i>()),
                _mm256_loadu_si256(codes.as_ptr().cast::<__m256i>()),
            )
        };
        sum += products(w, x) as f32 * (w_scale * x_scale);
    }
    sum
}

/// [`super::dot_q4_0`]과 같은 계산.
///
/// # Safety
///
/// 프로세서가 AVX2를 지원해야 한다.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q4_0(row: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let blocks = row.as_chunks::<{ Format::Q4_0.block_bytes() }>().0;
    let codes = codes.as_chunks::<BLOCK>().0;
    for ((block, codes), &x_scale) in blocks.iter().zip(codes).zip(scales) {
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: `block`은 18바이트이고 값은 2번째 바이트부터 16바이트다.
        // `codes`는 32개다.
        let (bytes, x) = unsafe {
            (
                _mm_loadu_si128(block.as_ptr().add(2).cast::<__m128i>()),
                _mm256_loadu_si256(codes.as_ptr().cast::<__m256i>()),
            )
        };
        // 아래 4비트 16개를 앞 절반에, 위 4비트 16개를 뒤 절반에 놓고 8을 뺀다.
        let mask = _mm_set1_epi8(0x0f);
        let low = _mm_and_si128(bytes, mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask);
        let w = _mm256_sub_epi8(_mm256_set_m128i(high, low), _mm256_set1_epi8(8));
        sum += products(w, x) as f32 * (w_scale * x_scale);
    }
    sum
}
