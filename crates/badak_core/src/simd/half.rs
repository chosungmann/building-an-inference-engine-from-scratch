//! 반정밀도(16비트) 부동소수점으로 계산하는 선형 층의 커널 (28장).
//!
//! 입력도, 가중치도, 누산도 16비트다. 레지스터 하나에 값이 여덟 개 들어가므로
//! 명령 하나가 `f32` 커널의 두 배를 계산한다. 대가는 정밀도다. 16비트
//! 부동소수점의 유효 숫자는 세 자리 남짓이다(19.2절).
//!
//! ARM의 선택 기능(FEAT_FP16)이 있어야 쓸 수 있다. 실험용이고, 엔진의 다른
//! 곳에서는 쓰지 않는다.

use std::arch::aarch64::*;

use crate::quant::F16;

/// 이 프로세서가 16비트 부동소수점 연산을 지원하는지.
pub fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("fp16")
}

// ANCHOR: dots
/// 입력 벡터 둘과 가중치 행 넷의 내적 여덟 개. 내적마다 레지스터 둘(값
/// 열여섯 개)에 누산한다.
///
/// # Safety
///
/// 프로세서가 FEAT_FP16을 지원해야 한다. 모든 슬라이스의 길이가 같아야 한다.
#[target_feature(enable = "neon,fp16")]
unsafe fn dots(xs: [&[F16]; 2], ws: [&[F16]; 4]) -> [[F16; 4]; 2] {
    let len = xs[0].len();
    let full = len - len % 16;
    // SAFETY: `F16`은 `u16` 하나를 감싼 것이고, 읽는 자리는 모두 `full`
    // 안이다. 16비트 값 여덟 개를 읽어 부동소수점 레지스터로 본다.
    let load = |values: &[F16], offset: usize| unsafe {
        vreinterpretq_f16_u16(vld1q_u16(values.as_ptr().add(offset).cast()))
    };
    let zero = vreinterpretq_f16_u16(vdupq_n_u16(0));
    let mut sums = [[[zero; 2]; 4]; 2];
    for start in (0..full).step_by(16) {
        #[allow(clippy::needless_range_loop)]
        for half in 0..2 {
            let offset = start + 8 * half;
            let x = [load(xs[0], offset), load(xs[1], offset)];
            for c in 0..4 {
                let w = load(ws[c], offset);
                for r in 0..2 {
                    sums[r][c][half] = vfmaq_f16(sums[r][c][half], x[r], w);
                }
            }
        }
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            // 열여섯 개를 여덟, 넷으로 줄이고, f32로 넓혀서 마저 더한다.
            let eight = vaddq_f16(sums[r][c][0], sums[r][c][1]);
            let four = vaddq_f32(
                vcvt_f32_f16(vget_low_f16(eight)),
                vcvt_f32_f16(vget_high_f16(eight)),
            );
            let mut sum = vaddvq_f32(four);
            for (x, w) in xs[r][full..].iter().zip(&ws[c][full..]) {
                sum += x.to_f32() * w.to_f32();
            }
            F16::from_f32(sum)
        })
    })
}
// ANCHOR_END: dots

/// 16비트로 계산하는 선형 층. `x`는 `(count, cols)`, `w`는 `(rows, cols)`,
/// `y`는 `(count, rows)`다. 스레드 하나로 계산한다. `rows`는 4의 배수,
/// `count`는 짝수여야 한다.
///
/// # Panics
///
/// 프로세서가 16비트 부동소수점 연산을 지원하지 않거나 크기가 맞지 않으면
/// 패닉한다.
pub fn linear(x: &[F16], count: usize, w: &[F16], rows: usize, cols: usize, y: &mut [F16]) {
    assert!(
        available(),
        "이 프로세서는 16비트 부동소수점 연산을 지원하지 않습니다"
    );
    assert!(rows.is_multiple_of(4) && count.is_multiple_of(2));
    assert!(x.len() == count * cols && w.len() == rows * cols && y.len() == count * rows);
    for row in (0..rows).step_by(4) {
        let ws: [&[F16]; 4] = std::array::from_fn(|c| &w[(row + c) * cols..][..cols]);
        for index in (0..count).step_by(2) {
            let xs = [&x[index * cols..][..cols], &x[(index + 1) * cols..][..cols]];
            // SAFETY: 위에서 FEAT_FP16을 확인했고, 슬라이스의 길이는 모두
            // `cols`다.
            let [first, second] = unsafe { dots(xs, ws) };
            y[index * rows + row..][..4].copy_from_slice(&first);
            y[(index + 1) * rows + row..][..4].copy_from_slice(&second);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_f32_within_half_precision() {
        if !available() {
            return;
        }
        let (count, rows, cols) = (4, 8, 37);
        let x: Vec<f32> = (0..count * cols)
            .map(|i| ((i * 7) % 13) as f32 * 0.125 - 0.75)
            .collect();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 5) % 11) as f32 * 0.0625 - 0.25)
            .collect();
        let half = |values: &[f32]| values.iter().map(|&v| F16::from_f32(v)).collect::<Vec<_>>();
        let mut y = vec![F16::from_f32(0.0); count * rows];
        linear(&half(&x), count, &half(&w), rows, cols, &mut y);
        for index in 0..count {
            for row in 0..rows {
                let exact: f32 = (0..cols)
                    .map(|c| x[index * cols + c] * w[row * cols + c])
                    .sum();
                let got = y[index * rows + row].to_f32();
                assert!(
                    (got - exact).abs() <= 0.01 * exact.abs().max(1.0),
                    "{got} != {exact}"
                );
            }
        }
    }
}
