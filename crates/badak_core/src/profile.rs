//! 계측 프로파일러: 연산마다 걸린 시간을 엔진이 스스로 적는다 (14장).
//!
//! 재고 싶은 함수의 첫 줄에 `let _scope = profile::scope("이름");`을 둔다.
//! 프로파일러가 꺼져 있으면 이 줄은 플래그 하나를 읽고 끝난다.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// 지금 실행 중인 구간들과 지금까지의 기록. 스레드마다 따로 둔다.
#[derive(Default)]
struct State {
    /// 바깥 구간부터 안쪽 구간까지의 이름과, 그 구간 안에서 더 안쪽 구간들이
    /// 쓴 시간.
    stack: Vec<(&'static str, Duration)>,
    /// 경로("바깥/안")에서 기록으로.
    records: BTreeMap<String, Record>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
}

// ANCHOR: record
/// 한 경로의 기록.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Record {
    /// 불린 횟수.
    pub calls: u64,
    /// 들어가서 나올 때까지의 시간. 안에서 부른 구간의 시간을 포함한다.
    pub total: Duration,
    /// 안에서 부른 구간의 시간을 뺀, 이 구간이 직접 쓴 시간.
    pub own: Duration,
}
// ANCHOR_END: record

/// 프로파일러를 켜거나 끈다.
pub fn enable(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// 이 스레드의 기록을 지운다.
pub fn reset() {
    STATE.with_borrow_mut(|state| state.records.clear());
}

/// 이 스레드의 기록을 (경로, 기록)의 목록으로 돌려준다. 경로의 사전 순서다.
pub fn report() -> Vec<(String, Record)> {
    STATE.with_borrow(|state| {
        state
            .records
            .iter()
            .map(|(path, record)| (path.clone(), *record))
            .collect()
    })
}

// ANCHOR: scope
/// 시간을 재는 구간. 만들어질 때 시작하고 사라질 때 끝난다.
pub struct Scope {
    /// 프로파일러가 꺼져 있었으면 `None`이고, 아무 일도 하지 않는다.
    start: Option<Instant>,
}

/// 구간을 시작한다. 돌려받은 값을 변수에 묶어 두면, 그 변수가 사라질 때
/// (보통 함수가 끝날 때) 구간이 끝난다.
pub fn scope(name: &'static str) -> Scope {
    if !ENABLED.load(Ordering::Relaxed) {
        return Scope { start: None };
    }
    STATE.with_borrow_mut(|state| state.stack.push((name, Duration::ZERO)));
    Scope {
        start: Some(Instant::now()),
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        let total = start.elapsed();
        STATE.with_borrow_mut(|state| {
            // 경로는 바깥 구간들의 이름을 이은 것이다.
            let names: Vec<&str> = state.stack.iter().map(|&(name, _)| name).collect();
            let path = names.join("/");
            let (_, children) = state.stack.pop().expect("시작할 때 넣어 두었다");

            let record = state.records.entry(path).or_default();
            record.calls += 1;
            record.total += total;
            record.own += total.saturating_sub(children);

            // 바깥 구간에게 "이만큼은 안쪽이 썼다"고 알린다.
            if let Some((_, outer_children)) = state.stack.last_mut() {
                *outer_children += total;
            }
        });
    }
}
// ANCHOR_END: scope

#[cfg(test)]
mod tests {
    use super::*;

    fn busy(duration: Duration) {
        let start = Instant::now();
        while start.elapsed() < duration {
            std::hint::spin_loop();
        }
    }

    fn inner() {
        let _scope = scope("inner");
        busy(Duration::from_millis(4));
    }

    fn outer() {
        let _scope = scope("outer");
        busy(Duration::from_millis(2));
        inner();
        inner();
    }

    // 켜고 끄는 플래그가 프로세스 전체의 것이어서 테스트 하나로 묶었다.
    #[test]
    fn records_paths_calls_and_own_time() {
        reset();
        outer();
        assert!(report().is_empty(), "꺼져 있으면 아무것도 적지 않는다");

        enable(true);
        outer();
        enable(false);
        let records: BTreeMap<String, Record> = report().into_iter().collect();
        assert_eq!(records.len(), 2);
        let (outer, inner) = (records["outer"], records["outer/inner"]);
        assert_eq!((outer.calls, inner.calls), (1, 2));
        assert_eq!(inner.own, inner.total);
        assert!(inner.total >= Duration::from_millis(8));
        assert!(outer.total >= Duration::from_millis(10));
        // 바깥 구간이 직접 쓴 시간에는 안쪽의 시간이 들어 있지 않다.
        assert!(outer.own >= Duration::from_millis(2));
        assert_eq!(outer.own + inner.total, outer.total);

        reset();
        assert!(report().is_empty());
    }
}
