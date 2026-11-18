//! 점수를 확률로 바꾸는 함수들 (2장).

use crate::linalg::sum_pairwise;

// ANCHOR: softmax_naive
/// 정의를 그대로 옮긴 소프트맥스. 점수가 크면 `exp`가 넘쳐 `NaN`을 낸다.
///
/// 수치 문제를 보여 주기 위한 구현이다. 실제로는 [`softmax`]를 쓴다.
pub fn softmax_naive(scores: &[f32]) -> Vec<f32> {
    let exps: Vec<f32> = scores.iter().map(|s| s.exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}
// ANCHOR_END: softmax_naive

// ANCHOR: softmax
/// 수치적으로 안정한 소프트맥스.
///
/// 모든 점수에서 최댓값을 뺀 뒤 계산한다. 결과는 수학적으로 같지만
/// `exp`의 인자가 0 이하가 되어 넘치지 않는다. 빈 입력에는 빈 벡터를
/// 돌려준다.
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    // 값이 많을 때 오차가 쌓이지 않도록 쌍별로 더한다 (5.6절, 12.6절).
    let total = sum_pairwise(&exps);
    exps.iter().map(|e| e / total).collect()
}
// ANCHOR_END: softmax

// ANCHOR: softmax_in_place
/// [`softmax`]와 같은 계산을, 새 벡터를 만들지 않고 받은 자리에서 한다 (22장).
///
/// 연산과 순서가 같으므로 결과의 비트도 같다.
pub fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
    }
    let total = sum_pairwise(scores);
    for score in scores.iter_mut() {
        *score /= total;
    }
}
// ANCHOR_END: softmax_in_place

// ANCHOR: log_softmax
/// 소프트맥스의 로그. `softmax` 결과에 `ln`을 취한 것과 같지만, 확률이
/// 0에 가까울 때도 `-inf`가 되지 않도록 직접 계산한다.
pub fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let log_total = sum_pairwise(&exps).ln();
    scores.iter().map(|s| s - max - log_total).collect()
}
// ANCHOR_END: log_softmax

// ANCHOR: argmax
/// 가장 큰 값의 위치. 같은 값이 여럿이면 가장 앞의 것을 돌려준다.
///
/// `NaN`은 건너뛴다. 입력이 비었거나 모두 `NaN`이면 `None`을 돌려준다.
pub fn argmax(values: &[f32]) -> Option<usize> {
    // 두 번에 나누어 한다. 먼저 가장 큰 값만 찾고(`f32::max`는 `NaN`을
    // 무시한다), 그다음 그 값이 처음 나오는 자리를 찾는다. 한 번에 하는
    // 것보다 훨씬 빠르다. 이유는 14.10절에서 잰다.
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    values.iter().position(|&value| value == max)
}
// ANCHOR_END: argmax

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_in_place_gives_the_same_bits() {
        for len in [0, 1, 2, 7, 64, 513] {
            let scores: Vec<f32> = (0..len)
                .map(|i| ((i * 37) % 23) as f32 * 0.31 - 3.0)
                .collect();
            let expected = softmax(&scores);
            let mut in_place = scores.clone();
            softmax_in_place(&mut in_place);
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&in_place), bits(&expected), "길이 {len}");
        }
    }
    use crate::approx::allclose;

    #[test]
    fn softmax_matches_hand_calculation() {
        // e⁰ = 1, e^(ln 3) = 3 이므로 확률은 1/4, 3/4.
        let p = softmax(&[0.0, 3.0_f32.ln()]);
        assert!(allclose(&p, &[0.25, 0.75], 1e-6, 0.0));
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let p = softmax(&[2.0, 1.0, 0.1, -3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p.is_sorted_by(|a, b| a >= b));
    }

    #[test]
    fn softmax_is_unchanged_by_adding_a_constant() {
        let a = softmax(&[1.0, 2.0, 3.0]);
        let b = softmax(&[101.0, 102.0, 103.0]);
        assert!(allclose(&a, &b, 1e-6, 0.0));
    }

    #[test]
    fn naive_softmax_overflows_but_stable_one_does_not() {
        let scores = [1000.0, 1001.0, 1002.0];
        assert!(softmax_naive(&scores).iter().all(|p| p.is_nan()));
        let p = softmax(&scores);
        assert!(allclose(&p, &softmax(&[0.0, 1.0, 2.0]), 1e-6, 0.0));
    }

    #[test]
    fn log_softmax_agrees_with_log_of_softmax() {
        let scores = [2.0, 1.0, 0.1];
        let expected: Vec<f32> = softmax(&scores).iter().map(|p| p.ln()).collect();
        assert!(allclose(&log_softmax(&scores), &expected, 1e-6, 1e-6));
    }

    #[test]
    fn log_softmax_stays_finite_for_tiny_probabilities() {
        let scores = [0.0, 200.0];
        assert_eq!(softmax(&scores)[0].ln(), f32::NEG_INFINITY);
        assert_eq!(log_softmax(&scores)[0], -200.0);
    }

    #[test]
    fn argmax_finds_first_largest() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), Some(1));
        assert_eq!(argmax(&[5.0, 5.0]), Some(0));
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[f32::NAN, 1.0]), Some(1));
        assert_eq!(argmax(&[f32::NAN]), None);
        // 가장 큰 값이 음의 무한대여도 자리를 찾는다.
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), Some(0));
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY, 2.0, 2.0]), Some(2));
    }
}
