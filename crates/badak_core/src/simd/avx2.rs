//! x86의 AVX2와 FMA로 짠 커널.
//!
//! AVX2의 레지스터 하나에는 `f32` 여덟이 들어간다. 합 여덟 개가 레지스터
//! 하나다. 이 명령들은 x86 프로세서 가운데 일부에만 있으므로, 이 모듈의
//! 함수는 있다는 것을 확인한 뒤에만 불러야 한다.

use core::arch::x86_64::{
    __m256, _mm_add_ps, _mm_storeu_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
};

use super::{LANES, Walk};

// ANCHOR: dots
/// 내적 `M × N`개. [`super::dots_portable`]과 같은 비트를 낸다.
///
/// `#[target_feature]`는 컴파일러에게 "이 함수 안에서는 AVX2와 FMA 명령을
/// 써도 된다"고 알린다. 같은 표시가 있는 함수에서만 안전하게 부를 수 있다.
///
/// # Safety
///
/// 같은 표시가 없는 함수에서 부를 때는 이 프로세서가 AVX2와 FMA를 지원함을
/// 부르는 쪽이 확인해야 한다.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub fn dots<const M: usize, const N: usize>(xs: [&[f32]; M], ws: [&[f32]; N]) -> [[f32; N]; M] {
    let len = ws[0].len();
    assert!(
        xs.iter().chain(&ws).all(|vector| vector.len() == len),
        "벡터의 길이가 다릅니다"
    );
    let full = len - len % LANES;

    // 내적마다 레지스터 하나: 합 0~7.
    let mut sums = [[_mm256_setzero_ps(); N]; M];
    for start in (0..full).step_by(LANES) {
        // SAFETY: `start + 8 <= full <= len`이고 모든 벡터의 길이가 `len`임을
        // 위에서 확인했다. 읽는 여덟 값은 슬라이스 안에 있다.
        let x: [__m256; M] =
            std::array::from_fn(|r| unsafe { _mm256_loadu_ps(xs[r].as_ptr().add(start)) });
        for c in 0..N {
            // SAFETY: 위와 같다.
            let w = unsafe { _mm256_loadu_ps(ws[c].as_ptr().add(start)) };
            for r in 0..M {
                // 합 += x × w. 값 여덟을 한꺼번에, 곱하고 더하기를 한 번의 연산으로.
                sums[r][c] = _mm256_fmadd_ps(x[r], w, sums[r][c]);
            }
        }
    }

    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            // 합 여덟을 넷으로: 아래 절반(0~3) + 위 절반(4~7).
            let low = _mm256_castps256_ps128(sums[r][c]);
            let high = _mm256_extractf128_ps::<1>(sums[r][c]);
            let four = _mm_add_ps(low, high);
            let mut lanes = [0.0_f32; 4];
            // SAFETY: `lanes`는 `f32` 넷을 쓸 수 있는 배열이다.
            unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), four) };
            let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
            for (x, w) in xs[r][full..].iter().zip(&ws[c][full..]) {
                sum = x.mul_add(*w, sum);
            }
            sum
        })
    })
}
// ANCHOR_END: dots

// ANCHOR: linear
/// 선형 층의 몸통. [`super::linear_with`]가 부른다.
///
/// # Safety
///
/// 이 프로세서가 AVX2와 FMA를 지원해야 한다.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn linear(
    x: &[f32],
    count: usize,
    w: &[f32],
    rows: usize,
    cols: usize,
    y: &mut [f32],
    walk: Walk,
) {
    drive!(dots, x, count, w, rows, cols, y, walk);
}
// ANCHOR_END: linear
