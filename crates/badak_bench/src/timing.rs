//! 무엇인가를 여러 번 실행해 시간을 잰다.

use std::time::Instant;

// ANCHOR: measure
/// 측정의 계획.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// 재지 않고 먼저 실행해 보는 횟수.
    pub warmup: usize,
    /// 재는 횟수.
    pub repetitions: usize,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            warmup: 1,
            repetitions: 5,
        }
    }
}

/// `work`를 계획대로 실행하고, 한 번 한 번에 걸린 시간을 초로 돌려준다.
///
/// `work`의 결과는 [`std::hint::black_box`]에 넘겨, 컴파일러가 "쓰이지 않는
/// 계산"이라며 지워 버리지 못하게 한다.
pub fn measure<R>(plan: Plan, mut work: impl FnMut() -> R) -> Vec<f64> {
    for _ in 0..plan.warmup {
        std::hint::black_box(work());
    }
    (0..plan.repetitions)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(work());
            start.elapsed().as_secs_f64()
        })
        .collect()
}
// ANCHOR_END: measure

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_warmup_and_repetitions() {
        let mut calls = 0;
        let plan = Plan {
            warmup: 2,
            repetitions: 3,
        };
        let samples = measure(plan, || calls += 1);
        assert_eq!(calls, 5);
        assert_eq!(samples.len(), 3);
        assert!(samples.iter().all(|&seconds| seconds >= 0.0));
    }
}
