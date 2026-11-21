//! 정수 내적을 ARM의 벡터 명령으로 짠 것 (20장).
//!
//! `vdotq_s32`는 8비트 정수 16쌍을 곱해, 네 쌍씩 더한 32비트 합 넷을 내는
//! 명령이다. 명령 하나가 곱셈 16번이다. 정수 계산이라 어떻게 묶어 더하든
//! [`super::dot_q8_0`], [`super::dot_q4_0`]과 같은 값이 나온다.

use core::arch::aarch64::{
    int8x16_t, int32x4_t, vaddq_f32, vaddvq_s32, vandq_u8, vcvt_f32_f16, vcvtq_f32_s32, vdotq_s32,
    vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vld1_dup_u16, vld1_lane_u16, vld1q_dup_f32,
    vld1q_lane_f32, vld1q_s8, vld1q_u8, vmulq_f32, vpaddq_s32, vreinterpret_f16_u16,
    vreinterpretq_s8_u8, vshrq_n_u8, vst1q_f32, vsubq_s8,
};

use super::{BLOCK, Format};
use crate::quant::f16_to_f32;

/// 가중치 묶음 하나의 정수 32개를 16개씩 둘로 읽는다.
///
/// # Safety
///
/// `values`부터 `Q8_0`이면 32바이트, `Q4_0`이면 16바이트를 읽을 수 있어야 한다.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack<const Q4: bool>(values: *const u8) -> (int8x16_t, int8x16_t) {
    // SAFETY: 부르는 쪽이 읽을 수 있음을 보장한다.
    unsafe {
        if Q4 {
            // 바이트 16개에서 아래 4비트 16개와 위 4비트 16개를 꺼내고, 8을
            // 빼서 -8부터 7까지의 정수로 만든다.
            let bytes = vld1q_u8(values);
            let eight = vdupq_n_s8(8);
            let low = vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0f)));
            let high = vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes));
            (vsubq_s8(low, eight), vsubq_s8(high, eight))
        } else {
            let values = values.cast::<i8>();
            (vld1q_s8(values), vld1q_s8(values.add(16)))
        }
    }
}

/// 묶음 하나에서 정수 32쌍의 곱을 넷씩 더한 합 넷.
///
/// # Safety
///
/// `x`부터 정수 32개를 읽을 수 있어야 한다.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn products(weights: (int8x16_t, int8x16_t), x: *const i8) -> int32x4_t {
    // SAFETY: 부르는 쪽이 읽을 수 있음을 보장한다.
    let (front, back) = unsafe { (vld1q_s8(x), vld1q_s8(x.add(16))) };
    vdotq_s32(vdotq_s32(vdupq_n_s32(0), weights.0, front), weights.1, back)
}

// ANCHOR: four
/// 내적 넷을 한꺼번에 계산한다: 입력 벡터 `M`개 각각과 가중치 `N`행 각각.
/// `M × N`은 4여야 한다.
///
/// 벡터 레지스터의 네 칸에 **내적 넷의 합**을 하나씩 담는다. 묶음마다 정수
/// 합 넷을 한 번에 `f32`로 바꾸고, 한 번에 곱하고, 한 번에 더한다. 칸마다의
/// 계산은 [`super::dot_q8_0`], [`super::dot_q4_0`]과 같은 순서이므로 같은
/// 비트가 나온다. 읽어 온 가중치와 입력은 여러 내적이 함께 쓴다(16.2절).
///
/// `Q4`가 참이면 가중치가 `Q4_0`, 거짓이면 `Q8_0`이다.
///
/// # Safety
///
/// 프로세서가 `dotprod`와 `fp16` 기능을 지원해야 한다.
#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
pub unsafe fn four<const Q4: bool, const M: usize, const N: usize>(
    rows: [&[u8]; N],
    codes: [&[i8]; M],
    scales: [&[f32]; M],
) -> [[f32; N]; M] {
    let format = if Q4 { Format::Q4_0 } else { Format::Q8_0 };
    let size = format.block_bytes();
    let blocks = scales[0].len();
    assert!(M * N == 4, "내적은 넷이어야 합니다");
    assert!(
        rows.iter().all(|row| row.len() == blocks * size)
            && codes.iter().all(|codes| codes.len() == blocks * BLOCK)
            && scales.iter().all(|scales| scales.len() == blocks),
        "묶음의 수가 다릅니다"
    );

    // `k`번째 칸은 `k / N`번째 벡터와 `k % N`번째 행의 내적이다.
    let mut sums = vdupq_n_f32(0.0);
    for block in 0..blocks {
        // SAFETY: 위에서 행마다 묶음 `blocks`개, 벡터마다 정수 `blocks × 32`개와
        // 스케일 `blocks`개가 있음을 확인했다. 아래의 읽기는 모두 그 안이다.
        let (totals, scale) = unsafe {
            let weights: [(int8x16_t, int8x16_t); N] =
                std::array::from_fn(|n| unpack::<Q4>(rows[n].as_ptr().add(block * size + 2)));
            let partial: [int32x4_t; 4] = std::array::from_fn(|k| {
                products(weights[k % N], codes[k / N].as_ptr().add(block * BLOCK))
            });
            // 내적마다 넷으로 나뉘어 있는 정수 합을 이웃끼리 두 번 더해, 네
            // 칸에 내적 넷의 합이 하나씩 놓이게 한다.
            let totals = vpaddq_s32(
                vpaddq_s32(partial[0], partial[1]),
                vpaddq_s32(partial[2], partial[3]),
            );
            // 두 스케일의 곱 넷. 스케일들을 레지스터의 네 칸에 하나씩 읽어
            // 들인다. 가중치 쪽은 16비트 실수 넷을 한 명령으로 `f32`로 바꾼다.
            let w = |k: usize| rows[k % N].as_ptr().add(block * size).cast::<u16>();
            let bits = vld1_dup_u16(w(0));
            let bits = vld1_lane_u16::<1>(w(1), bits);
            let bits = vld1_lane_u16::<2>(w(2), bits);
            let bits = vld1_lane_u16::<3>(w(3), bits);
            let x = |k: usize| scales[k / N].as_ptr().add(block);
            let factors = vld1q_dup_f32(x(0));
            let factors = vld1q_lane_f32::<1>(x(1), factors);
            let factors = vld1q_lane_f32::<2>(x(2), factors);
            let factors = vld1q_lane_f32::<3>(x(3), factors);
            let scale = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(bits)), factors);
            (totals, scale)
        };
        // 곱하기와 더하기를 따로 한다. 기준 구현과 같은 반올림이다.
        sums = vaddq_f32(sums, vmulq_f32(vcvtq_f32_s32(totals), scale));
    }
    let mut values = [0.0_f32; 4];
    // SAFETY: `values`는 `f32` 넷을 쓸 수 있는 배열이다.
    unsafe { vst1q_f32(values.as_mut_ptr(), sums) };
    std::array::from_fn(|m| std::array::from_fn(|n| values[m * N + n]))
}
// ANCHOR_END: four

// ANCHOR: one
/// 내적 하나. [`super::dot_q8_0`], [`super::dot_q4_0`]과 같은 계산이다.
///
/// # Safety
///
/// 프로세서가 `dotprod` 기능을 지원해야 한다.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn one<const Q4: bool>(row: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let format = if Q4 { Format::Q4_0 } else { Format::Q8_0 };
    let size = format.block_bytes();
    let blocks = scales.len();
    assert!(
        row.len() == blocks * size && codes.len() == blocks * BLOCK,
        "묶음의 수가 다릅니다"
    );
    let mut sum = 0.0_f32;
    for (block, &x_scale) in scales.iter().enumerate() {
        let w_scale = f16_to_f32(u16::from_le_bytes([
            row[block * size],
            row[block * size + 1],
        ]));
        // SAFETY: 위에서 묶음 `blocks`개가 있음을 확인했다.
        let total = unsafe {
            let weights = unpack::<Q4>(row.as_ptr().add(block * size + 2));
            // 32비트 합 넷을 하나로 더한다.
            vaddvq_s32(products(weights, codes.as_ptr().add(block * BLOCK)))
        };
        sum += total as f32 * (w_scale * x_scale);
    }
    sum
}
// ANCHOR_END: one

// ANCHOR: piece
/// 가중치의 행 `width`개를 입력 벡터 `count`개와 곱해 `(count, width)`
/// 모양의 결과를 채운다. 내적을 넷씩 묶어 계산한다.
///
/// 벡터가 하나이면 가중치를 네 부분으로 나누어 부분마다 한 행씩, 네 행을
/// 함께 읽는다(17.5절의 네 줄기). 여럿이면 이웃한 두 행과 벡터 둘을 묶는다.
///
/// # Safety
///
/// 프로세서가 `dotprod`와 `fp16` 기능을 지원해야 한다.
#[target_feature(enable = "neon,dotprod,fp16")]
pub unsafe fn piece<const Q4: bool>(
    bytes: &[u8],
    width: usize,
    codes: &[i8],
    scales: &[f32],
    count: usize,
    output: &mut [f32],
) {
    let row_bytes = bytes.len() / width.max(1);
    let (cols, blocks) = (codes.len() / count.max(1), scales.len() / count.max(1));
    let row = |index: usize| &bytes[index * row_bytes..(index + 1) * row_bytes];
    let code = |vector: usize| &codes[vector * cols..(vector + 1) * cols];
    let scale = |vector: usize| &scales[vector * blocks..(vector + 1) * blocks];

    let mut done = 0;
    if count == 1 {
        let part = width / 4;
        for step in 0..part {
            let rows: [&[u8]; 4] = std::array::from_fn(|stream| row(stream * part + step));
            // SAFETY: 부르는 쪽이 프로세서의 기능을 확인했다.
            let [values] = unsafe { four::<Q4, 1, 4>(rows, [code(0)], [scale(0)]) };
            for (stream, value) in values.into_iter().enumerate() {
                output[stream * part + step] = value;
            }
        }
        done = 4 * part;
    }
    let paired = count - count % 2;
    while done + 2 <= width && paired > 0 {
        let rows = [row(done), row(done + 1)];
        for vector in (0..paired).step_by(2) {
            let codes = [code(vector), code(vector + 1)];
            let scales = [scale(vector), scale(vector + 1)];
            // SAFETY: 위와 같다.
            let [first, second] = unsafe { four::<Q4, 2, 2>(rows, codes, scales) };
            output[vector * width + done..][..2].copy_from_slice(&first);
            output[(vector + 1) * width + done..][..2].copy_from_slice(&second);
        }
        for vector in paired..count {
            for offset in 0..2 {
                // SAFETY: 위와 같다.
                let value = unsafe { one::<Q4>(row(done + offset), code(vector), scale(vector)) };
                output[vector * width + done + offset] = value;
            }
        }
        done += 2;
    }
    // 남은 행은 내적을 하나씩 계산한다.
    for index in done..width {
        for vector in 0..count {
            // SAFETY: 위와 같다.
            let value = unsafe { one::<Q4>(row(index), code(vector), scale(vector)) };
            output[vector * width + index] = value;
        }
    }
}
// ANCHOR_END: piece
