//! 벡터와 행렬의 기본 연산 (2장).
//!
//! 벡터는 `&[f32]`로, 행렬은 행 우선(row-major) 순서로 펼친 `&[f32]`와
//! 행·열 개수로 나타낸다. 여기 있는 구현은 가장 단순한 형태이며, 뒤의
//! 장에서 만드는 빠른 구현의 정답 기준으로 계속 쓰인다.

// ANCHOR: dot
/// 두 벡터의 내적. 같은 자리의 원소끼리 곱해 모두 더한다.
///
/// # Panics
///
/// 두 벡터의 길이가 다르면 패닉한다.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "내적: 두 벡터의 길이가 다릅니다");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
// ANCHOR_END: dot

// ANCHOR: add_scale
/// 두 벡터의 합. 같은 자리의 원소끼리 더한다.
///
/// # Panics
///
/// 두 벡터의 길이가 다르면 패닉한다.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "덧셈: 두 벡터의 길이가 다릅니다");
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// 벡터의 스칼라배. 모든 원소에 같은 수를 곱한다.
pub fn scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}
// ANCHOR_END: add_scale

// ANCHOR: norm_cosine
/// 벡터의 길이(유클리드 노름).
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// 두 벡터의 코사인 유사도. 방향이 같으면 1, 수직이면 0, 반대면 -1이다.
///
/// 어느 한쪽이 영벡터이면 방향이 정의되지 않으므로 0을 돌려준다.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let denominator = norm(a) * norm(b);
    if denominator == 0.0 {
        0.0
    } else {
        dot(a, b) / denominator
    }
}
// ANCHOR_END: norm_cosine

// ANCHOR: matvec
/// 행렬과 벡터의 곱 `W x`.
///
/// `w`는 `rows × cols` 행렬을 행 우선으로 펼친 것이다. 결과의 `i`번째
/// 원소는 `w`의 `i`번째 행과 `x`의 내적이다.
///
/// # Panics
///
/// `w`의 길이가 `rows * cols`가 아니거나 `x`의 길이가 `cols`가 아니면
/// 패닉한다.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "행렬의 크기가 맞지 않습니다");
    assert_eq!(x.len(), cols, "행렬의 열 수와 벡터의 길이가 다릅니다");
    w.chunks_exact(cols).map(|row| dot(row, x)).collect()
}
// ANCHOR_END: matvec

// ANCHOR: matmul
/// 행렬 곱 `C = A B`. `A`는 `m × k`, `B`는 `k × n`, 결과 `C`는 `m × n`이다.
///
/// 정의를 그대로 옮긴 삼중 반복문이다. `C`의 `(i, j)` 원소는 `A`의
/// `i`번째 행과 `B`의 `j`번째 열의 내적이다.
///
/// # Panics
///
/// 두 행렬의 길이가 주어진 크기와 맞지 않으면 패닉한다.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A의 크기가 맞지 않습니다");
    assert_eq!(b.len(), k * n, "B의 크기가 맞지 않습니다");
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}
// ANCHOR_END: matmul

// ANCHOR: transpose
/// 전치 행렬. `rows × cols` 행렬의 행과 열을 맞바꾼 `cols × rows` 행렬을
/// 돌려준다.
///
/// # Panics
///
/// `a`의 길이가 `rows * cols`가 아니면 패닉한다.
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols, "행렬의 크기가 맞지 않습니다");
    let mut t = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}
// ANCHOR_END: transpose

// ANCHOR: sum_naive
/// 앞에서부터 차례로 더한다. 원소가 많으면 오차가 원소 수에 비례해 쌓인다.
pub fn sum_naive(values: &[f32]) -> f32 {
    values.iter().sum()
}
// ANCHOR_END: sum_naive

// ANCHOR: sum_pairwise
/// 반으로 나눠 각각 더한 뒤 두 합을 더한다(쌍별 합산).
///
/// 비슷한 크기의 수끼리 더하게 되어, 오차가 원소 수의 로그에 비례해
/// 느리게 쌓인다. 조각이 충분히 작아지면 단순한 반복문으로 더한다.
pub fn sum_pairwise(values: &[f32]) -> f32 {
    /// 이보다 짧은 조각은 나누지 않고 그냥 더한다.
    const LEAF: usize = 128;
    if values.len() <= LEAF {
        return sum_naive(values);
    }
    let (left, right) = values.split_at(values.len() / 2);
    sum_pairwise(left) + sum_pairwise(right)
}
// ANCHOR_END: sum_pairwise

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_hand_calculation() {
        // 1×4 + 2×5 + 3×6 = 32
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    #[should_panic(expected = "길이가 다릅니다")]
    fn dot_rejects_mismatched_lengths() {
        dot(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn add_and_scale() {
        assert_eq!(add(&[1.0, 2.0], &[10.0, 20.0]), [11.0, 22.0]);
        assert_eq!(scale(&[1.0, -2.0], 3.0), [3.0, -6.0]);
    }

    #[test]
    fn norm_of_3_4_is_5() {
        assert_eq!(norm(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn cosine_similarity_of_known_directions() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 3.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-5.0, 0.0]), -1.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn matvec_is_one_dot_product_per_row() {
        // [1 2 3]   [1]   [1 + 0 - 3]   [-2]
        // [4 5 6] · [0] = [4 + 0 - 6] = [-2]
        //           [-1]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(matvec(&w, 2, 3, &[1.0, 0.0, -1.0]), [-2.0, -2.0]);
    }

    #[test]
    fn matmul_matches_hand_calculation() {
        // [1 2]   [5 6]   [1×5+2×7  1×6+2×8]   [19 22]
        // [3 4] · [7 8] = [3×5+4×7  3×6+4×8] = [43 50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(matmul(&a, &b, 2, 2, 2), [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_handles_non_square_shapes() {
        // (2×3) · (3×1) 은 matvec과 같은 결과여야 한다.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, -1.0];
        assert_eq!(matmul(&a, &x, 2, 3, 1), matvec(&a, 2, 3, &x));
    }

    #[test]
    fn matmul_by_identity_changes_nothing() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul(&a, &identity, 2, 3, 3), a);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        // [1 2 3]ᵀ   [1 4]
        // [4 5 6]  = [2 5]
        //            [3 6]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose(&a, 2, 3), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&transpose(&a, 2, 3), 3, 2), a);
    }

    #[test]
    fn both_sums_are_exact_for_small_integers() {
        let values: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        assert_eq!(sum_naive(&values), 500_500.0);
        assert_eq!(sum_pairwise(&values), 500_500.0);
        assert_eq!(sum_pairwise(&[]), 0.0);
    }

    #[test]
    fn pairwise_sum_is_more_accurate_than_naive_sum() {
        // 0.1을 100만 번 더한다. 정답은 f64로 계산한다.
        let values = vec![0.1_f32; 1_000_000];
        let exact: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let naive_error = (f64::from(sum_naive(&values)) - exact).abs();
        let pairwise_error = (f64::from(sum_pairwise(&values)) - exact).abs();
        assert!(pairwise_error < naive_error / 100.0);
    }
}
