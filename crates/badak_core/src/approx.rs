//! 부동소수점 값을 허용 오차 안에서 비교하는 도구 (2장).
//!
//! 부동소수점 계산은 연산 순서만 달라져도 마지막 자리가 달라진다. 그래서
//! 두 구현의 결과를 `==`로 비교할 수 없고, "충분히 가까운가"를 물어야
//! 한다.

// ANCHOR: max_abs_diff
/// 같은 자리 원소끼리의 차이 가운데 가장 큰 절댓값.
///
/// # Panics
///
/// 두 슬라이스의 길이가 다르면 패닉한다.
pub fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "비교할 길이가 다릅니다");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}
// ANCHOR_END: max_abs_diff

// ANCHOR: allclose
/// 모든 원소가 허용 오차 안에서 같은지 검사한다.
///
/// 원소마다 `|actual - expected| <= atol + rtol * |expected|`를 확인한다.
/// `rtol`은 값의 크기에 비례하는 상대 오차, `atol`은 0 근처의 값을 위한
/// 절대 오차다. NumPy의 `allclose`와 같은 규칙이다.
///
/// 길이가 다르거나 `NaN`이 있으면 `false`를 돌려준다.
pub fn allclose(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= atol + rtol * e.abs())
}
// ANCHOR_END: allclose

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_abs_diff_reports_worst_element() {
        assert_eq!(max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.5, 2.75]), 0.5);
        assert_eq!(max_abs_diff(&[], &[]), 0.0);
    }

    #[test]
    fn relative_tolerance_scales_with_magnitude() {
        // 100만 근처에서 1의 차이는 상대 오차 1e-6 수준이다.
        assert!(allclose(&[1_000_001.0], &[1_000_000.0], 1e-5, 0.0));
        assert!(!allclose(&[2.0], &[1.0], 1e-5, 0.0));
    }

    #[test]
    fn absolute_tolerance_handles_values_near_zero() {
        // 기준값이 0이면 상대 오차만으로는 어떤 차이도 통과하지 못한다.
        assert!(!allclose(&[1e-9], &[0.0], 1e-5, 0.0));
        assert!(allclose(&[1e-9], &[0.0], 1e-5, 1e-8));
    }

    #[test]
    fn nan_and_length_mismatch_are_never_close() {
        assert!(!allclose(&[f32::NAN], &[f32::NAN], 1e-5, 1e-8));
        assert!(!allclose(&[1.0], &[1.0, 1.0], 1e-5, 1e-8));
    }
}
