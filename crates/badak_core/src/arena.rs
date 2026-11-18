//! 아레나: 메모리를 한 덩어리로 얻어 차례로 떼어 쓰고 한꺼번에 돌려준다 (22장).
//!
//! 순전파는 블록 하나를 계산하는 동안 텐서를 수십 개 만들고, 블록이 끝나면
//! 결과 하나만 남기고 모두 버린다. 텐서마다 힙에서 자리를 얻고 돌려주는
//! 대신, 큰 덩어리 하나에서 앞에서부터 차례로 떼어 주고 블록이 끝나면
//! "처음부터 다시 쓴다"고만 적는다.
//!
//! 아레나는 스레드마다 하나다. [`scope`] 안에서 만드는 `f32` 텐서
//! ([`Tensor::build`](crate::Tensor::build))가 이 아레나에서 자리를 얻는다.

use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// 아레나의 덩어리가 처음에 갖는 크기(`f32`의 수). 256KB다.
///
/// Miri로 검사할 때는 작게 잡는다. Miri는 프로그램을 한 걸음씩 흉내 내어
/// 실행하므로 값 백만 개를 채우는 테스트가 끝나지 않는다.
const INITIAL_CAPACITY: usize = if cfg!(miri) { 1 << 8 } else { 1 << 16 };

// ANCHOR: block
/// 아레나가 쥔 메모리 한 덩어리. `f32`를 `capacity`개 담는다.
///
/// 텐서들이 이 덩어리의 구간을 가리킨다. 구간은 서로 겹치지 않게 앞에서부터
/// 차례로 내준다.
pub struct Block {
    /// 덩어리의 첫 `f32`. `Box<[f32]>`에서 꺼낸 포인터다.
    start: NonNull<f32>,
    capacity: usize,
    /// 지금까지 내준 양. 다음에 내줄 구간이 여기서 시작한다.
    used: AtomicUsize,
}

// SAFETY: 덩어리의 메모리는 구간으로만 건드린다. 구간 하나는 `take`가 한
// 번만 내주고, 받은 쪽이 다 채운 뒤에야 다른 스레드가 읽을 수 있는 텐서가
// 된다. 그 뒤로는 읽기만 한다. 덩어리를 처음부터 다시 쓰는 것은 구간을
// 가리키는 텐서가 하나도 없을 때만 한다([`scope`]).
unsafe impl Sync for Block {}
// SAFETY: 위와 같다. `f32`는 어느 스레드에서 읽고 써도 된다.
unsafe impl Send for Block {}

impl Block {
    fn new(capacity: usize) -> Self {
        // 0으로 채운 메모리를 운영체제에서 얻는다. 실제로 쓰기 전까지는
        // 물리 메모리를 차지하지 않는다(22.3절).
        let memory = vec![0.0_f32; capacity].into_boxed_slice();
        let start = Box::into_raw(memory) as *mut f32;
        Self {
            start: NonNull::new(start).expect("Box의 포인터는 널이 아니다"),
            capacity,
            used: AtomicUsize::new(0),
        }
    }

    /// 다음 구간 `len`개를 내준다. 자리가 모자라면 `None`.
    fn take(&self, len: usize) -> Option<Range<usize>> {
        let start = self.used.load(Ordering::Relaxed);
        let end = start.checked_add(len).filter(|&end| end <= self.capacity)?;
        self.used.store(end, Ordering::Relaxed);
        Some(start..end)
    }

    /// 구간의 값을 읽는다.
    ///
    /// # Safety
    ///
    /// `range`는 [`take`](Self::take)가 내준 구간(또는 그 일부)이어야 하고,
    /// 그 구간을 쓰는 중인 쪽이 없어야 한다.
    pub(crate) unsafe fn slice(&self, range: Range<usize>) -> &[f32] {
        debug_assert!(range.start <= range.end && range.end <= self.capacity);
        // SAFETY: 부르는 쪽이 구간이 덩어리 안이고 쓰는 쪽이 없음을 보장한다.
        unsafe { std::slice::from_raw_parts(self.start.as_ptr().add(range.start), range.len()) }
    }

    /// 방금 [`take`](Self::take)로 얻은 구간을 0으로 채우고, 값을 쓸 수 있게
    /// 빌려준다.
    ///
    /// # Safety
    ///
    /// `range`는 방금 `take`가 내준 구간이어야 하고, 이 빌림이 끝나기 전에는
    /// 누구도 그 구간을 읽거나 쓰지 않아야 한다. 곧 그 구간을 가리키는
    /// 텐서를 아직 만들지 않았어야 한다.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn fresh(&self, range: Range<usize>) -> &mut [f32] {
        debug_assert!(range.start <= range.end && range.end <= self.capacity);
        // SAFETY: 부르는 쪽이 이 구간을 혼자 쓴다는 것을 보장한다.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(self.start.as_ptr().add(range.start), range.len())
        };
        slice.fill(0.0);
        slice
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        let memory = std::ptr::slice_from_raw_parts_mut(self.start.as_ptr(), self.capacity);
        // SAFETY: `new`에서 `Box<[f32]>`를 풀어 얻은 포인터와 길이 그대로다.
        // 덩어리를 가리키는 텐서가 모두 사라진 뒤에만 여기에 온다.
        drop(unsafe { Box::from_raw(memory) });
    }
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Block({} / {})",
            self.used.load(Ordering::Relaxed),
            self.capacity
        )
    }
}
// ANCHOR_END: block

// ANCHOR: state
/// 스레드 하나의 아레나.
#[derive(Default)]
struct State {
    /// 지금 쓰는 덩어리.
    block: Option<Arc<Block>>,
    /// 겹쳐 들어온 [`scope`]의 수. 0이면 아레나를 쓰지 않는다.
    depth: usize,
    /// 이번 구간에서 달라고 한 양의 합. 덩어리에 못 들어간 것도 센다.
    wanted: usize,
    /// 지난 구간에서 달라고 한 양. 다음 덩어리의 크기를 정한다.
    last_wanted: usize,
    statistics: Statistics,
}

/// 아레나가 한 일의 기록.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Statistics {
    /// 덩어리에서 내준 구간의 수.
    pub taken: usize,
    /// 자리가 모자라 힙으로 보낸 요청의 수.
    pub overflowed: usize,
    /// 덩어리를 새로 얻은 횟수.
    pub blocks: usize,
    /// 지금 덩어리의 크기(`f32`의 수).
    pub capacity: usize,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
}
// ANCHOR_END: state

// ANCHOR: scope
/// `work`를 실행하는 동안 이 스레드에서 만드는 `f32` 텐서가 아레나에서
/// 자리를 얻게 한다.
///
/// 시작할 때 덩어리를 처음부터 다시 쓰도록 되돌린다. 지난번에 만든 텐서를
/// 누군가 아직 쥐고 있으면 되돌리지 못하므로, 그때는 새 덩어리를 얻는다.
/// 그래서 **구간 밖으로 내보낼 텐서는 [`Tensor::detached`]나
/// [`Tensor::assign`](crate::Tensor::assign)으로 힙에 옮겨 적어야** 덩어리를
/// 돌려 쓸 수 있다. 옮겨 적지 않아도 틀린 값이 나오지는 않는다.
/// 덩어리를 매번 새로 얻느라 느려질 뿐이다.
///
/// 안에서 다시 불러도 된다. 가장 바깥의 것만 덩어리를 되돌린다.
///
/// 환경 변수 `BADAK_ARENA`가 `off`이면 아무 일도 하지 않고 `work`만
/// 실행한다. 텐서는 모두 힙에서 자리를 얻는다. 아레나가 있을 때와 없을
/// 때를 견주려고 둔 것이다.
///
/// [`Tensor::detached`]: crate::Tensor::detached
pub fn scope<R>(work: impl FnOnce() -> R) -> R {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("BADAK_ARENA").as_deref() != Ok("off")) {
        return work();
    }

    /// 구간을 떠날 때(패닉으로 떠날 때도) 깊이를 되돌린다.
    struct Leave;
    impl Drop for Leave {
        fn drop(&mut self) {
            STATE.with_borrow_mut(|state| {
                state.depth -= 1;
                if state.depth == 0 {
                    state.last_wanted = state.wanted;
                }
            });
        }
    }

    STATE.with_borrow_mut(|state| {
        state.depth += 1;
        if state.depth > 1 {
            return;
        }
        state.wanted = 0;
        // 덩어리를 가리키는 것이 우리뿐이고 크기가 알맞으면 처음부터 다시 쓴다.
        // 지난번에 쓴 양보다 작으면 모자라고, 네 배가 넘으면 헛되이 크다.
        //
        // "우리뿐인가"는 `Arc::get_mut`에 묻는다. 덩어리의 구간을 가리키는
        // 텐서가 어느 스레드에든 하나라도 남아 있으면 `None`을 돌려준다.
        let wanted = state.last_wanted;
        let reused = match state.block.as_mut().and_then(Arc::get_mut) {
            Some(block) => {
                let fits = block.capacity >= wanted;
                let wasteful = block.capacity > INITIAL_CAPACITY && block.capacity / 4 > wanted;
                if fits && !wasteful {
                    *block.used.get_mut() = 0;
                }
                fits && !wasteful
            }
            None => false,
        };
        if !reused {
            // 지난번에 쓴 양보다 조금 넉넉하게 새로 얻는다. 옛 덩어리는 그것을
            // 가리키는 텐서가 모두 사라질 때 함께 사라진다.
            let capacity = (wanted + wanted / 4).max(INITIAL_CAPACITY);
            state.block = Some(Arc::new(Block::new(capacity)));
            state.statistics.blocks += 1;
            state.statistics.capacity = capacity;
        }
    });
    let _leave = Leave;
    work()
}
// ANCHOR_END: scope

// ANCHOR: take
/// 아레나에서 `f32` `len`개의 자리를 얻는다. 돌려주는 것은 그 자리를
/// 가리키는 (덩어리, 구간)이다.
///
/// [`scope`] 밖이거나 자리가 모자라면 `None`을 돌려준다. 그때는 부른 쪽이
/// 힙을 쓴다.
pub(crate) fn take(len: usize) -> Option<(Arc<Block>, Range<usize>)> {
    STATE.with_borrow_mut(|state| {
        if state.depth == 0 {
            return None;
        }
        state.wanted += len;
        let block = state.block.as_ref()?;
        match block.take(len) {
            Some(range) => {
                state.statistics.taken += 1;
                Some((Arc::clone(block), range))
            }
            None => {
                state.statistics.overflowed += 1;
                None
            }
        }
    })
}
// ANCHOR_END: take

// ANCHOR: scratch
thread_local! {
    /// 잠깐 쓰고 버리는 값들을 담는 자리. 스레드마다 하나를 두고 돌려 쓴다.
    static SCRATCH: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// `f32` `len`개의 자리를 빌려 `work`를 실행한다.
///
/// 자리는 이 스레드가 지난번에 쓴 것을 다시 쓴다. 지난번보다 길지 않으면
/// 힙에서 새로 얻지 않는다. **지난번에 쓴 값이 그대로 남아 있을 수
/// 있으므로**, 받은 쪽이 읽기 전에 모두 채워야 한다. 텐서가 되지 않고 함수
/// 안에서만 쓰는 값들(점수, 사인과 코사인)을 담는다. [`scope`] 밖에서도,
/// 스레드 풀의 스레드에서도 쓸 수 있다.
pub fn with_scratch<R>(len: usize, work: impl FnOnce(&mut [f32]) -> R) -> R {
    // 꺼내 가면 빈 벡터가 남는다. `work` 안에서 다시 부르면 그 빈 벡터가
    // 새로 자리를 얻으므로, 겹쳐 불러도 틀리지는 않는다.
    let mut buffer = SCRATCH.take();
    // 지금까지 쓴 가장 긴 길이를 그대로 두고 앞부분만 빌려준다. 길이를
    // 줄였다가 다시 늘리면 늘어난 부분을 매번 0으로 채우게 된다.
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    let result = work(&mut buffer[..len]);
    SCRATCH.set(buffer);
    result
}
// ANCHOR_END: scratch

/// 이 스레드의 아레나가 지금까지 한 일.
pub fn statistics() -> Statistics {
    STATE.with_borrow(|state| state.statistics)
}

/// 이 스레드의 아레나가 쥔 덩어리와, 돌려 쓰려고 쥐고 있던 자리를 놓는다.
/// 다음에 쓸 때 새로 얻는다.
pub fn release() {
    STATE.with_borrow_mut(|state| {
        state.block = None;
        state.last_wanted = 0;
        state.statistics.capacity = 0;
    });
    SCRATCH.take();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    fn filled(shape: &[usize], value: f32) -> Tensor {
        Tensor::build(shape, |data| {
            // 받은 자리는 0으로 채워져 있다.
            assert!(data.iter().all(|&v| v == 0.0));
            data.fill(value);
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn tensors_built_in_a_scope_live_in_the_arena() {
        release();
        let outside = filled(&[4], 1.0);
        assert!(!outside.is_in_arena());
        let before = statistics();
        let total = scope(|| {
            let a = filled(&[2, 3], 2.0);
            let b = filled(&[5], 3.0);
            assert!(a.is_in_arena() && b.is_in_arena());
            // 서로 다른 구간이라 값이 섞이지 않는다.
            assert_eq!(a.to_vec::<f32>().unwrap(), [2.0; 6]);
            assert_eq!(b.to_vec::<f32>().unwrap(), [3.0; 5]);
            // 뷰도 그대로 된다.
            assert_eq!(a.transpose(0, 1).unwrap().shape(), [3, 2]);
            a.len() + b.len()
        });
        assert_eq!(total, 11);
        assert_eq!(statistics().taken, before.taken + 2);
    }

    #[test]
    fn the_block_is_reused_when_nothing_points_into_it() {
        release();
        scope(|| drop(filled(&[100], 1.0)));
        let blocks = statistics().blocks;
        for round in 0..10 {
            scope(|| {
                let tensor = filled(&[100], round as f32);
                assert_eq!(tensor.as_slice::<f32>().unwrap()[99], round as f32);
            });
        }
        // 열 번을 더 돌았지만 덩어리는 새로 얻지 않았다.
        assert_eq!(statistics().blocks, blocks);
    }

    #[test]
    fn a_tensor_that_escapes_keeps_its_values() {
        release();
        // 옮겨 적지 않고 구간 밖으로 내보낸 텐서.
        let escaped = scope(|| filled(&[8], 7.0));
        let blocks = statistics().blocks;
        // 다음 구간은 그 덩어리를 되돌리지 못하고 새 덩어리를 얻는다.
        scope(|| drop(filled(&[8], 9.0)));
        assert_eq!(statistics().blocks, blocks + 1);
        assert_eq!(escaped.to_vec::<f32>().unwrap(), [7.0; 8]);

        // 옮겨 적어 내보내면 덩어리를 돌려 쓴다.
        let kept = scope(|| filled(&[8], 5.0).detached().unwrap());
        assert!(!kept.is_in_arena());
        let blocks = statistics().blocks;
        scope(|| drop(filled(&[8], 6.0)));
        assert_eq!(statistics().blocks, blocks);
        assert_eq!(kept.to_vec::<f32>().unwrap(), [5.0; 8]);
    }

    #[test]
    fn requests_that_do_not_fit_go_to_the_heap_and_the_block_grows() {
        release();
        let large = INITIAL_CAPACITY * 2;
        let overflowed = statistics().overflowed;
        scope(|| {
            let tensor = filled(&[large], 1.0);
            // 덩어리에 못 들어가 힙으로 갔다.
            assert!(!tensor.is_in_arena());
        });
        assert_eq!(statistics().overflowed, overflowed + 1);
        // 다음 구간에서는 덩어리가 커져 있어 들어간다.
        scope(|| assert!(filled(&[large], 1.0).is_in_arena()));
        assert!(statistics().capacity >= large);
    }

    #[test]
    fn a_block_far_larger_than_needed_is_replaced() {
        release();
        let large = INITIAL_CAPACITY * 16;
        scope(|| drop(filled(&[large], 1.0)));
        scope(|| drop(filled(&[large], 1.0)));
        assert!(statistics().capacity >= large);
        // 작은 일로 바뀐 첫 구간은 큰 덩어리를 그대로 쓴다. 그다음 구간에서
        // 지난번에 쓴 양을 보고 작은 덩어리로 바꾼다.
        scope(|| drop(filled(&[10], 1.0)));
        assert!(statistics().capacity >= large);
        scope(|| assert!(filled(&[10], 1.0).is_in_arena()));
        assert_eq!(statistics().capacity, INITIAL_CAPACITY);
    }

    #[test]
    fn scratch_is_reused() {
        let first = with_scratch(100, |values| {
            assert_eq!(values.len(), 100);
            values.fill(3.0);
            values.as_ptr() as usize
        });
        // 같은 자리를 다시 받는다. 지난번의 값이 남아 있다.
        let second = with_scratch(50, |values| {
            assert_eq!(values.len(), 50);
            assert!(values.iter().all(|&v| v == 3.0));
            // 겹쳐 불러도 서로 다른 자리를 받는다.
            with_scratch(10, |inner| assert_ne!(inner.as_ptr(), values.as_ptr()));
            values.as_ptr() as usize
        });
        assert_eq!(first, second);
    }

    #[test]
    fn scopes_nest_and_survive_a_panic() {
        release();
        scope(|| {
            let outer = filled(&[4], 1.0);
            scope(|| {
                // 안쪽 구간은 덩어리를 되돌리지 않는다.
                let inner = filled(&[4], 2.0);
                assert_eq!(outer.to_vec::<f32>().unwrap(), [1.0; 4]);
                assert_eq!(inner.to_vec::<f32>().unwrap(), [2.0; 4]);
            });
            assert!(filled(&[4], 3.0).is_in_arena());
        });
        let result = std::panic::catch_unwind(|| scope(|| panic!("일부러 낸 패닉")));
        assert!(result.is_err());
        // 패닉 뒤에 구간 밖에서 만든 텐서는 힙에 있다.
        assert!(!filled(&[4], 1.0).is_in_arena());
    }
}
