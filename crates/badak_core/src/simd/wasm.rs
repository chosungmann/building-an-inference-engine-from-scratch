//! WebAssembly의 128비트 벡터 명령으로 짠 커널 (48장).
//!
//! 레지스터 하나에 `f32` 넷이 들어간다. NEON과 같은 폭이고, 커널의 생김새도
//! NEON의 것과 같다. 다른 점은 하나다. **곱하고 더하기를 한 번에 하는 명령이
//! 없다.** 그래서 이 커널은 곱한 결과를 한 번 반올림하고 더한 결과를 또
//! 반올림한다. 기준 구현과 끝자리가 다를 수 있다.
//!
//! `relaxed-simd`를 켜고 빌드하면 "프로세서에 그런 명령이 있으면 쓰라"는
//! 명령(`f32x4_relaxed_madd`)을 쓴다. 그 명령이 어느 쪽으로 계산할지는
//! 브라우저와 프로세서가 정한다. 그래서 역시 비트를 약속하지 못한다.

use core::arch::wasm32::{f32x4_add, f32x4_splat, v128, v128_store};

use super::{LANES, Walk, Weight};

// ANCHOR: madd
/// `sum + x × w`를 값 넷에 대해 계산한다.
#[inline(always)]
fn multiply_add(sum: v128, x: v128, w: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        core::arch::wasm32::f32x4_relaxed_madd(x, w, sum)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(sum, core::arch::wasm32::f32x4_mul(x, w))
    }
}
// ANCHOR_END: madd

// ANCHOR: dots
/// 내적 `M × N`개. 생김새는 [`super::neon`]의 것과 같다.
///
/// # Safety
///
/// `simd128`을 켜고 빌드한 모듈에서만 부른다. 이 파일이 그런 빌드에서만
/// 컴파일된다.
#[inline]
#[target_feature(enable = "simd128")]
pub fn dots<W: Weight, const M: usize, const N: usize>(
    xs: [&[f32]; M],
    ws: [&[W]; N],
) -> [[f32; N]; M] {
    let len = ws[0].len();
    assert!(
        xs.iter().all(|x| x.len() == len) && ws.iter().all(|w| w.len() == len),
        "벡터의 길이가 다릅니다"
    );
    let full = len - len % LANES;

    // 내적마다 레지스터 둘: 앞의 것이 합 0~3, 뒤의 것이 합 4~7.
    let mut sums = [[[f32x4_splat(0.0); 2]; N]; M];
    for start in (0..full).step_by(LANES) {
        #[allow(clippy::needless_range_loop)]
        for half in 0..2 {
            let offset = start + 4 * half;
            // SAFETY: `offset + 4 <= full <= len`이고 모든 벡터의 길이가
            // `len`임을 위에서 확인했다.
            let x: [v128; M] =
                std::array::from_fn(|r| unsafe { f32::load_wasm(xs[r].as_ptr().add(offset)) });
            for c in 0..N {
                // SAFETY: 위와 같다.
                let w = unsafe { W::load_wasm(ws[c].as_ptr().add(offset)) };
                for r in 0..M {
                    sums[r][c][half] = multiply_add(sums[r][c][half], x[r], w);
                }
            }
        }
    }

    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let four = f32x4_add(sums[r][c][0], sums[r][c][1]);
            let mut lanes = [0.0_f32; 4];
            // SAFETY: `lanes`는 `f32` 넷(16바이트)을 쓸 수 있는 배열이다.
            unsafe { v128_store(lanes.as_mut_ptr().cast(), four) };
            let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
            // 끝부분도 몸통과 같은 방식으로: 곱하고, 더한다.
            for (x, w) in xs[r][full..].iter().zip(&ws[c][full..]) {
                sum += x * w.widen();
            }
            sum
        })
    })
}
// ANCHOR_END: dots

/// 선형 층의 몸통. [`super::linear_with`]가 부른다.
///
/// # Safety
///
/// `simd128`을 켜고 빌드한 모듈에서만 부른다.
#[target_feature(enable = "simd128")]
pub unsafe fn linear<W: Weight>(
    x: &[f32],
    count: usize,
    w: &[W],
    rows: usize,
    cols: usize,
    y: &mut [f32],
    walk: Walk,
) {
    drive!(dots, x, count, w, rows, cols, y, walk);
}
