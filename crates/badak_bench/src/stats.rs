//! 측정값 여러 개를 요약한다.

// ANCHOR: percentile
/// 작은 순서로 정렬된 값들에서 `q` 분위의 값을 고른다. `q`는 0과 1 사이다.
///
/// "가장 가까운 순위" 방법이다. 값들 가운데 `q`의 비율이 그 값 이하가 되는
/// 가장 작은 값을 돌려준다. 실제로 측정된 값만 나온다.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "값이 하나는 있어야 합니다");
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}
// ANCHOR_END: percentile

// ANCHOR: summary
/// 측정값들의 요약.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// 값의 수.
    pub count: usize,
    /// 가장 작은 값.
    pub min: f64,
    /// 가장 큰 값.
    pub max: f64,
    /// 평균.
    pub mean: f64,
    /// 표본 표준편차. 값이 하나뿐이면 0이다.
    pub deviation: f64,
    /// 중앙값(50 분위).
    pub median: f64,
    /// 90 분위. 열 번 가운데 아홉 번은 이 값 이하다.
    pub p90: f64,
    /// 99 분위.
    pub p99: f64,
}

impl Summary {
    /// 값들을 요약한다. 값이 하나는 있어야 한다.
    pub fn new(samples: &[f64]) -> Self {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let count = sorted.len();
        let mean = sorted.iter().sum::<f64>() / count as f64;
        let deviation = if count < 2 {
            0.0
        } else {
            let squares: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum();
            (squares / (count - 1) as f64).sqrt()
        };
        Self {
            count,
            min: sorted[0],
            max: sorted[count - 1],
            mean,
            deviation,
            median: percentile(&sorted, 0.5),
            p90: percentile(&sorted, 0.9),
            p99: percentile(&sorted, 0.99),
        }
    }

    /// 흩어진 정도를 평균에 대한 비율로 나타낸 것(변동 계수).
    pub fn relative_deviation(&self) -> f64 {
        self.deviation / self.mean
    }
}
// ANCHOR_END: summary

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_the_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&sorted, 0.5), 5.0);
        assert_eq!(percentile(&sorted, 0.9), 9.0);
        assert_eq!(percentile(&sorted, 0.99), 10.0);
        assert_eq!(percentile(&sorted, 0.0), 1.0);
        assert_eq!(percentile(&sorted, 1.0), 10.0);
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }

    #[test]
    fn summary_of_known_values() {
        let summary = Summary::new(&[4.0, 2.0, 8.0, 6.0]);
        assert_eq!((summary.count, summary.min, summary.max), (4, 2.0, 8.0));
        assert_eq!(summary.mean, 5.0);
        // 분산 = (9 + 1 + 1 + 9) / 3
        assert!((summary.deviation - (20.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(summary.median, 4.0);
        assert_eq!(Summary::new(&[3.0]).deviation, 0.0);
    }

    #[test]
    fn one_outlier_moves_the_mean_but_not_the_median() {
        let mut samples = vec![10.0; 99];
        samples.push(1000.0);
        let summary = Summary::new(&samples);
        assert_eq!(summary.median, 10.0);
        assert_eq!(summary.p99, 10.0);
        assert!(summary.mean > 19.0);
        assert_eq!(summary.max, 1000.0);
    }
}
