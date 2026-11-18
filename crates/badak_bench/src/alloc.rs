//! 메모리 할당을 세는 할당자 (22장).
//!
//! Rust 프로그램이 힙에서 메모리를 얻고 돌려줄 때는 전역 할당자를 거친다.
//! 그 자리에 세는 장치를 끼워 넣으면 프로그램이 할당을 몇 번, 몇 바이트나
//! 하는지 알 수 있다. 쓰려는 프로그램이 직접 등록해야 한다.
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: badak_bench::alloc::Counting = badak_bench::alloc::Counting;
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// 크기별로 나눠 세는 구간의 경계(바이트). 마지막 구간은 그 이상 전부다.
pub const SIZE_CLASSES: [usize; 3] = [1 << 10, 1 << 16, 1 << 20];

static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static BY_SIZE: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

// ANCHOR: counting
/// 운영체제의 할당자에 그대로 맡기되, 지나가는 요청을 센다.
pub struct Counting;

fn record(size: usize) {
    COUNT.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size, Ordering::Relaxed);
    let class = SIZE_CLASSES
        .iter()
        .take_while(|&&limit| size >= limit)
        .count();
    BY_SIZE[class].fetch_add(1, Ordering::Relaxed);
    let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(current, Ordering::Relaxed);
}

// SAFETY: 모든 요청을 `System`에 그대로 넘기고 그 결과를 그대로 돌려준다.
// 세는 일은 원자적 변수만 건드리고 스스로 메모리를 할당하지 않는다.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: 부르는 쪽이 지킨 조건을 그대로 넘긴다.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: 위와 같다.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: 위와 같다.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // 크기를 바꾸는 것은 새로 얻는 것으로 센다. 값을 옮겨야 할 수 있다.
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        record(new_size);
        // SAFETY: 위와 같다.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}
// ANCHOR_END: counting

// ANCHOR: snapshot
/// 어느 순간까지의 할당 기록.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// 할당한 횟수.
    pub count: usize,
    /// 할당한 바이트의 합. 돌려준 것을 빼지 않는다.
    pub bytes: usize,
    /// 지금 쥐고 있는 바이트.
    pub current: usize,
    /// 쥐고 있던 바이트의 최댓값.
    pub peak: usize,
    /// 크기별 횟수. [`SIZE_CLASSES`]의 경계로 나눈 네 구간이다.
    pub by_size: [usize; 4],
}

impl Snapshot {
    /// 지금까지의 기록. [`Counting`]을 등록하지 않았으면 모두 0이다.
    pub fn now() -> Self {
        Self {
            count: COUNT.load(Ordering::Relaxed),
            bytes: BYTES.load(Ordering::Relaxed),
            current: CURRENT.load(Ordering::Relaxed),
            peak: PEAK.load(Ordering::Relaxed),
            by_size: std::array::from_fn(|class| BY_SIZE[class].load(Ordering::Relaxed)),
        }
    }

    /// `earlier` 뒤로 일어난 일. 횟수와 바이트는 차이이고, `peak`는 그
    /// 사이의 최댓값에서 `earlier` 때 쥐고 있던 양을 뺀 것이다.
    ///
    /// 구간의 최댓값을 보려면 구간을 시작할 때 [`reset_peak`]를 불러야 한다.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            count: self.count - earlier.count,
            bytes: self.bytes - earlier.bytes,
            current: self.current,
            peak: self.peak.saturating_sub(earlier.current),
            by_size: std::array::from_fn(|class| self.by_size[class] - earlier.by_size[class]),
        }
    }
}

/// 최댓값의 기록을 지금 쥐고 있는 양으로 되돌린다.
pub fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}
// ANCHOR_END: snapshot
