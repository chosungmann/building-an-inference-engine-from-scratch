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

// ANCHOR: compare
/// 두 결과를 비교한 보고서. 검증이 실패했을 때 어디가 얼마나 틀렸는지
/// 알려 준다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// 비교한 원소 수.
    pub len: usize,
    /// 허용 오차를 벗어난 원소 수.
    pub mismatches: usize,
    /// 가장 큰 절대 오차.
    pub max_abs_diff: f32,
    /// 가장 큰 절대 오차가 난 위치.
    pub worst_index: usize,
    /// 그 위치의 실제값.
    pub worst_actual: f32,
    /// 그 위치의 기준값.
    pub worst_expected: f32,
}

impl Comparison {
    /// 모든 원소가 허용 오차 안에 있는지.
    pub fn is_close(&self) -> bool {
        self.mismatches == 0
    }
}

impl std::fmt::Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "원소 {}개 중 {}개가 허용 오차를 벗어남. 가장 큰 차이 {:e} (위치 {}: 실제 {}, 기준 {})",
            self.len,
            self.mismatches,
            self.max_abs_diff,
            self.worst_index,
            self.worst_actual,
            self.worst_expected
        )
    }
}

/// [`allclose`]와 같은 규칙으로 비교하되, 결과를 보고서로 돌려준다.
///
/// `NaN`이 있는 자리는 허용 오차를 벗어난 것으로 센다.
///
/// # Panics
///
/// 두 슬라이스의 길이가 다르면 패닉한다.
pub fn compare(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> Comparison {
    assert_eq!(actual.len(), expected.len(), "비교할 길이가 다릅니다");
    let mut report = Comparison {
        len: actual.len(),
        mismatches: 0,
        max_abs_diff: 0.0,
        worst_index: 0,
        worst_actual: 0.0,
        worst_expected: 0.0,
    };
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        // `diff <= ...`의 부정으로 쓰면 NaN도 불일치로 잡힌다.
        let close = diff <= atol + rtol * e.abs();
        if !close {
            report.mismatches += 1;
        }
        if diff > report.max_abs_diff || (diff.is_nan() && !report.max_abs_diff.is_nan()) {
            report.max_abs_diff = diff;
            report.worst_index = index;
            report.worst_actual = a;
            report.worst_expected = e;
        }
    }
    report
}
// ANCHOR_END: compare

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

    #[test]
    fn compare_reports_where_and_how_much() {
        let report = compare(&[1.0, 2.5, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.1], 0.0, 0.2);
        assert!(!report.is_close());
        assert_eq!(
            (report.len, report.mismatches, report.worst_index),
            (4, 1, 1)
        );
        assert_eq!((report.worst_actual, report.worst_expected), (2.5, 2.0));
        assert!(report.to_string().contains("4개 중 1개"));

        assert!(compare(&[1.0, 2.0], &[1.0, 2.0], 0.0, 0.0).is_close());
        assert!(!compare(&[f32::NAN], &[1.0], 1.0, 1.0).is_close());
    }
}
