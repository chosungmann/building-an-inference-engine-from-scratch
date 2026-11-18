//! 메모리 계획: 값마다 큰 자리 하나 안의 어디를 쓸지 정한다.
//!
//! 그래프를 보면 값이 언제 만들어지고 언제 마지막으로 읽히는지 알 수 있다.
//! 쓰이는 동안이 겹치지 않는 두 값은 같은 자리를 써도 된다. 이 모듈은 그
//! 사실만으로 자리를 나눈다. 연산이 무엇인지는 모른다.

use crate::error::{Error, Result};
use crate::graph::Graph;

// ANCHOR: lifetime
/// 값 하나의 크기와, 그 값이 살아 있는 동안.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    /// 원소의 수.
    pub size: usize,
    /// 이 값을 만드는 단계.
    pub first: usize,
    /// 이 값을 마지막으로 읽는 단계. 그 단계가 끝나야 자리를 내줄 수 있다.
    pub last: usize,
}

impl Lifetime {
    /// 두 값이 함께 살아 있는 단계가 있는지.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// 값마다 정해진 자리.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// 값마다 자리가 시작하는 위치(원소 단위).
    pub offsets: Vec<usize>,
    /// 필요한 자리 전체의 크기.
    pub total: usize,
}

/// 자리를 나누는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// 값마다 자기 자리를 준다. 함께 쓰지 않는다.
    Separate,
    /// 만들어지는 순서대로, 그때 비어 있는 틈에 넣는다.
    InOrder,
    /// 큰 값부터 넣는다. LiteRT가 쓰는 방법이다.
    BySize,
}
// ANCHOR_END: lifetime

// ANCHOR: lifetimes
/// 그래프에서 계산되는 값마다의 수명을 읽는다. 값의 번호와 수명의 쌍을
/// 돌려준다.
///
/// `order`는 마디를 실행할 순서이고([`Graph::order`]), 단계의 번호는 그
/// 순서에서의 위치다. 값은 자기를 만드는 단계에서 태어나, 자기를 읽는 마지막
/// 단계가 끝날 때까지 산다. 가중치처럼 값이 정해진 것은 자리가 필요 없으므로
/// 뺀다.
pub fn lifetimes(
    graph: &Graph,
    order: &[usize],
    shapes: &[Vec<usize>],
) -> Result<Vec<(usize, Lifetime)>> {
    let mut spans: Vec<Option<(usize, usize)>> = vec![None; graph.values.len()];
    // 입력은 실행을 시작할 때 이미 있어야 한다.
    for &input in &graph.inputs {
        spans[input] = Some((0, 0));
    }
    for (step, &index) in order.iter().enumerate() {
        let node = &graph.nodes[index];
        for &output in &node.outputs {
            spans[output] = Some((step, step));
        }
        for &input in &node.inputs {
            if let Some((_, last)) = &mut spans[input] {
                *last = step;
            }
        }
    }
    // 그래프의 출력은 실행이 끝난 뒤에도 읽을 수 있어야 한다. 마지막 단계
    // 다음에 단계가 하나 더 있는 것으로 친다.
    for &output in &graph.outputs {
        match &mut spans[output] {
            Some((_, last)) => *last = order.len(),
            None if graph.values[output].constant.is_some() => {}
            None => {
                return Err(Error::Graph(
                    "그래프의 출력을 만드는 마디가 없습니다".into(),
                ));
            }
        }
    }
    Ok(spans
        .iter()
        .enumerate()
        .filter(|&(value, _)| graph.values[value].constant.is_none())
        .filter_map(|(value, span)| {
            let (first, last) = (*span)?;
            let size = shapes[value].iter().product();
            Some((value, Lifetime { size, first, last }))
        })
        .collect())
}
// ANCHOR_END: lifetimes

// ANCHOR: plan
/// 값들의 자리를 정한다.
pub fn plan(lifetimes: &[Lifetime], strategy: Strategy) -> Layout {
    let mut order: Vec<usize> = (0..lifetimes.len()).collect();
    match strategy {
        Strategy::Separate => {
            let mut total = 0;
            let offsets = lifetimes
                .iter()
                .map(|lifetime| {
                    let offset = total;
                    total += lifetime.size;
                    offset
                })
                .collect();
            return Layout { offsets, total };
        }
        Strategy::InOrder => order.sort_by_key(|&index| lifetimes[index].first),
        // 크기가 같으면 먼저 만들어지는 것부터.
        Strategy::BySize => order.sort_by_key(|&index| {
            (
                std::cmp::Reverse(lifetimes[index].size),
                lifetimes[index].first,
            )
        }),
    }
    place(lifetimes, &order)
}

/// `order`의 순서로 값을 하나씩 놓는다.
///
/// 놓으려는 값과 함께 살아 있는 값들만이 걸림돌이다. 그 값들 사이의 틈
/// 가운데 들어갈 수 있는 가장 작은 틈에 넣고, 없으면 맨 뒤에 붙인다.
fn place(lifetimes: &[Lifetime], order: &[usize]) -> Layout {
    let mut offsets = vec![0; lifetimes.len()];
    let mut total = 0;
    // 이미 놓은 값들. 자리가 앞인 것부터 늘어놓는다.
    let mut placed: Vec<usize> = Vec::with_capacity(order.len());
    for &index in order {
        let lifetime = &lifetimes[index];
        if lifetime.size == 0 {
            continue;
        }
        let mut best: Option<(usize, usize)> = None;
        // 지금까지 본 걸림돌들이 끝나는 곳.
        let mut end = 0;
        for &other in &placed {
            if !lifetime.overlaps(&lifetimes[other]) {
                continue;
            }
            let start = offsets[other];
            if end + lifetime.size <= start {
                let gap = start - end;
                if best.is_none_or(|(_, smallest)| gap < smallest) {
                    best = Some((end, gap));
                }
            }
            end = end.max(start + lifetimes[other].size);
        }
        let offset = best.map_or(end, |(offset, _)| offset);
        offsets[index] = offset;
        total = total.max(offset + lifetime.size);
        let position = placed.partition_point(|&other| offsets[other] <= offset);
        placed.insert(position, index);
    }
    Layout { offsets, total }
}
// ANCHOR_END: plan

// ANCHOR: bound
/// 어떤 계획도 이보다 작을 수 없는 크기. 한 단계에서 함께 살아 있는 값들의
/// 크기를 더한 것 가운데 가장 큰 값이다.
pub fn lower_bound(lifetimes: &[Lifetime]) -> usize {
    let steps = lifetimes.iter().map(|lifetime| lifetime.last + 1).max();
    (0..steps.unwrap_or(0))
        .map(|step| {
            lifetimes
                .iter()
                .filter(|lifetime| lifetime.first <= step && step <= lifetime.last)
                .map(|lifetime| lifetime.size)
                .sum()
        })
        .max()
        .unwrap_or(0)
}
// ANCHOR_END: bound

// ANCHOR: check
/// 계획이 올바른지 확인한다. 함께 살아 있는 두 값의 자리가 겹치면 안 되고,
/// 모든 자리가 전체 크기 안에 있어야 한다.
///
/// 해석기는 이 계획을 믿고 한 메모리의 여러 곳을 동시에 읽고 쓴다. 계획이
/// 틀리면 한 값이 다른 값을 덮어쓴다. 그래서 계획을 쓰기 전에 반드시
/// 확인한다.
pub fn check(lifetimes: &[Lifetime], layout: &Layout) -> Result<()> {
    if layout.offsets.len() != lifetimes.len() {
        return Err(Error::Graph("계획의 값 수가 맞지 않습니다".into()));
    }
    let range = |index: usize| {
        let start = layout.offsets[index];
        start..start + lifetimes[index].size
    };
    for (index, lifetime) in lifetimes.iter().enumerate() {
        if lifetime.size == 0 {
            continue;
        }
        if range(index).end > layout.total {
            return Err(Error::Graph(format!(
                "값 {index}의 자리가 전체 크기를 벗어납니다"
            )));
        }
        for (other, earlier) in lifetimes.iter().enumerate().take(index) {
            let (a, b) = (range(index), range(other));
            let shared = a.start < b.end && b.start < a.end;
            if shared && earlier.size != 0 && lifetime.overlaps(earlier) {
                return Err(Error::Graph(format!(
                    "함께 살아 있는 값 {other}와 {index}의 자리가 겹칩니다"
                )));
            }
        }
    }
    Ok(())
}
// ANCHOR_END: check

#[cfg(test)]
mod tests {
    use super::*;
    use badak_core::random::Pcg64;

    fn lifetime(size: usize, first: usize, last: usize) -> Lifetime {
        Lifetime { size, first, last }
    }

    #[test]
    fn a_chain_needs_only_two_neighbours() {
        // x → a → b → c. 이웃한 둘만 함께 살아 있다.
        let chain = [
            lifetime(100, 0, 0),
            lifetime(100, 0, 1),
            lifetime(100, 1, 2),
            lifetime(100, 2, 3),
        ];
        assert_eq!(plan(&chain, Strategy::Separate).total, 400);
        assert_eq!(lower_bound(&chain), 200);
        for strategy in [Strategy::InOrder, Strategy::BySize] {
            let layout = plan(&chain, strategy);
            check(&chain, &layout).unwrap();
            assert_eq!(layout.total, 200);
        }
    }

    #[test]
    fn small_values_fill_the_smallest_gap() {
        // 큰 값 둘 사이에 틈 30이, 그 뒤에 틈 10이 있다. 크기 10인 값은
        // 딱 맞는 뒤의 틈으로 간다.
        let lifetimes = [
            lifetime(50, 0, 9),
            lifetime(30, 0, 0),
            lifetime(40, 0, 9),
            lifetime(10, 0, 0),
            lifetime(20, 0, 9),
            lifetime(10, 5, 5),
        ];
        let layout = plan(&lifetimes, Strategy::InOrder);
        check(&lifetimes, &layout).unwrap();
        assert_eq!(layout.offsets, [0, 50, 80, 120, 130, 120]);
        assert_eq!(layout.total, 150);
    }

    #[test]
    fn order_of_placement_changes_the_total() {
        // 작은 값이 먼저 만들어지고 큰 값들이 나중에 온다. 순서대로 놓으면
        // 작은 값이 앞을 차지해서 큰 값이 들어갈 틈이 쪼개진다.
        let lifetimes = [
            lifetime(50, 1, 2),
            lifetime(100, 2, 4),
            lifetime(10, 3, 4),
            lifetime(100, 4, 6),
        ];
        let in_order = plan(&lifetimes, Strategy::InOrder);
        let by_size = plan(&lifetimes, Strategy::BySize);
        check(&lifetimes, &in_order).unwrap();
        check(&lifetimes, &by_size).unwrap();
        assert_eq!(lower_bound(&lifetimes), 210);
        assert_eq!(by_size.total, 210);
        assert_eq!(in_order.total, 250);
    }

    #[test]
    fn largest_first_is_not_always_best() {
        // 큰 값부터 놓는 쪽이 지는 경우도 있다. 어느 쪽도 가장 좋은 답을
        // 보장하지 않는다.
        let lifetimes = [
            lifetime(50, 2, 2),
            lifetime(10, 2, 3),
            lifetime(10, 3, 5),
            lifetime(50, 4, 4),
        ];
        let in_order = plan(&lifetimes, Strategy::InOrder);
        let by_size = plan(&lifetimes, Strategy::BySize);
        check(&lifetimes, &in_order).unwrap();
        check(&lifetimes, &by_size).unwrap();
        assert_eq!(lower_bound(&lifetimes), 60);
        assert_eq!(in_order.total, 60);
        assert_eq!(by_size.total, 70);
    }

    #[test]
    fn check_catches_wrong_plans() {
        let lifetimes = [lifetime(10, 0, 1), lifetime(10, 1, 2), lifetime(10, 2, 3)];
        // 첫째와 셋째는 함께 살아 있지 않으므로 겹쳐도 된다.
        let good = Layout {
            offsets: vec![0, 10, 0],
            total: 20,
        };
        check(&lifetimes, &good).unwrap();
        let overlapping = Layout {
            offsets: vec![0, 5, 20],
            total: 30,
        };
        assert!(check(&lifetimes, &overlapping).is_err());
        let outside = Layout {
            offsets: vec![0, 10, 0],
            total: 15,
        };
        assert!(check(&lifetimes, &outside).is_err());
        let short = Layout {
            offsets: vec![0, 10],
            total: 20,
        };
        assert!(check(&lifetimes, &short).is_err());
    }

    #[test]
    fn random_plans_are_valid_and_bounded() {
        let mut rng = Pcg64::new(26);
        for _ in 0..300 {
            let count = 1 + (rng.next_u64() % 24) as usize;
            let lifetimes: Vec<Lifetime> = (0..count)
                .map(|_| {
                    let first = (rng.next_u64() % 16) as usize;
                    let last = first + (rng.next_u64() % 6) as usize;
                    lifetime((rng.next_u64() % 9) as usize * 16, first, last)
                })
                .collect();
            let sum: usize = lifetimes.iter().map(|lifetime| lifetime.size).sum();
            let bound = lower_bound(&lifetimes);
            for strategy in [Strategy::Separate, Strategy::InOrder, Strategy::BySize] {
                let layout = plan(&lifetimes, strategy);
                check(&lifetimes, &layout).unwrap();
                assert!(bound <= layout.total && layout.total <= sum);
            }
            assert_eq!(plan(&lifetimes, Strategy::Separate).total, sum);
        }
    }
}
