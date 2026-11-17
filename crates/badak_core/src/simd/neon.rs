//! ARM의 NEON으로 짠 커널.
//!
//! NEON의 레지스터 하나에는 `f32` 넷이 들어간다. 합 여덟 개는 레지스터 둘에
//! 담는다.

use core::arch::aarch64::{float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

use super::{LANES, Walk, Weight};

// ANCHOR: dots
/// 내적 `M × N`개. [`super::dots_portable`]과 같은 비트를 낸다.
///
/// `#[target_feature]`는 컴파일러에게 "이 함수 안에서는 NEON 명령을 써도
/// 된다"고 알린다. 같은 표시가 있는 함수에서만 안전하게 부를 수 있다.
///
/// # Safety
///
/// 같은 표시가 없는 함수에서 부를 때는 이 프로세서가 NEON을 지원함을
/// 부르는 쪽이 확인해야 한다.
#[inline]
#[target_feature(enable = "neon")]
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
    let mut sums = [[[vdupq_n_f32(0.0); 2]; N]; M];
    for start in (0..full).step_by(LANES) {
        // `half`는 레지스터를 고르는 번호이면서 읽을 자리를 정하는 데도 쓴다.
        #[allow(clippy::needless_range_loop)]
        for half in 0..2 {
            let offset = start + 4 * half;
            // SAFETY: `offset + 4 <= full <= len`이고 모든 벡터의 길이가
            // `len`임을 위에서 확인했다. 읽는 네 값은 슬라이스 안에 있다.
            let x: [float32x4_t; M] =
                std::array::from_fn(|r| unsafe { vld1q_f32(xs[r].as_ptr().add(offset)) });
            for c in 0..N {
                // SAFETY: 위와 같다. 가중치는 저장된 형식에서 `f32` 넷으로
                // 바꿔 읽는다(20장). `f32`이면 그대로 읽는다.
                let w = unsafe { W::load_neon(ws[c].as_ptr().add(offset)) };
                for r in 0..M {
                    // 합 += x × w. 값 넷을 한꺼번에, 곱하고 더하기를 한 번의 연산으로.
                    sums[r][c][half] = vfmaq_f32(sums[r][c][half], x[r], w);
                }
            }
        }
    }

    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            // 합 여덟을 넷으로: i번째 + (i + 4)번째.
            let four = vaddq_f32(sums[r][c][0], sums[r][c][1]);
            let mut lanes = [0.0_f32; 4];
            // SAFETY: `lanes`는 `f32` 넷을 쓸 수 있는 배열이다.
            unsafe { vst1q_f32(lanes.as_mut_ptr(), four) };
            let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
            for (x, w) in xs[r][full..].iter().zip(&ws[c][full..]) {
                sum = x.mul_add(w.widen(), sum);
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
/// 이 프로세서가 NEON을 지원해야 한다. 64비트 ARM에서는 늘 그렇다.
#[target_feature(enable = "neon")]
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
