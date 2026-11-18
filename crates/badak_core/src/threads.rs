//! 스레드 풀: 일을 여러 코어에 나눈다 (18장).
//!
//! 표준 라이브러리의 스레드, 원자적 변수, 뮤텍스, 조건 변수만으로 만든다.
//! 일꾼 스레드를 미리 만들어 두고, 일이 올 때마다 알려서 쓴다.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ANCHOR: waiting
/// 기다릴 일이 있을 때 스레드가 하는 일.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waiting {
    /// 곧바로 잠든다. 깨우는 것은 운영체제에 맡긴다.
    Sleep,
    /// 정한 시간 동안 깨어서 코어를 쥔 채로 기다리고, 그 뒤에 잠든다.
    Spin(Duration),
    /// 정한 시간 동안 깨어서 기다리되 다른 스레드에게 코어를 양보해 가며
    /// 기다리고, 그 뒤에 잠든다.
    Yield(Duration),
}

impl Waiting {
    /// `ready`가 참이 될 때까지, 또는 정한 시간이 다 될 때까지 깨어서 기다린다.
    fn stay_awake(self, ready: impl Fn() -> bool) {
        let (duration, yielding) = match self {
            Self::Sleep => return,
            Self::Spin(duration) => (duration, false),
            Self::Yield(duration) => (duration, true),
        };
        let deadline = Instant::now() + duration;
        while !ready() && Instant::now() < deadline {
            if yielding {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }
}
// ANCHOR_END: waiting

// ANCHOR: job
/// 일 하나: `0..count`의 번호마다 한 번씩 부를 함수.
///
/// 빌린 것을 가리키는 포인터에서 수명을 지운 것이다. 이 포인터가 가리키는
/// 함수는 [`ThreadPool::run`]이 돌아오기 전까지만 살아 있다.
#[derive(Clone, Copy)]
struct Job {
    task: *const (dyn Fn(usize) + Sync),
    count: usize,
}

/// 풀 하나가 가질 수 있는 스레드의 최대 수. 잠든 일꾼을 64비트 값의 비트
/// 하나씩으로 적기 때문에 생긴 한계다.
pub const MAX_THREADS: usize = 64;

/// 호출 하나를 나타내는 값: 일의 번호와, 그 일에 참여할 일꾼의 수.
///
/// 둘을 한 값에 담아야 일꾼이 "몇 번째 일인지"와 "내가 참여하는지"를 한 번에,
/// 어긋나지 않게 읽을 수 있다.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Call {
    epoch: u64,
    workers: usize,
}

impl Call {
    const WORKER_BITS: u32 = 8;

    fn pack(self) -> u64 {
        (self.epoch << Self::WORKER_BITS) | self.workers as u64
    }

    fn unpack(value: u64) -> Self {
        Self {
            epoch: value >> Self::WORKER_BITS,
            workers: (value & ((1 << Self::WORKER_BITS) - 1)) as usize,
        }
    }
}

/// 풀을 쥔 쪽과 일꾼들이 함께 보는 상태.
struct Shared {
    /// 지금의 일. `call`을 바꾸기 전에 적고, 참여한 일꾼이 모두 끝났다고
    /// 알릴 때까지 바꾸지 않는다.
    job: UnsafeCell<Option<Job>>,
    /// 지금의 호출([`Call`]). 일이 새로 올 때마다 번호가 1씩 는다.
    call: AtomicU64,
    /// 다음에 가져갈 번호. 스레드들이 여기서 번호를 하나씩 뽑아 간다.
    next: AtomicUsize,
    /// 지금의 일을 아직 끝냈다고 알리지 않은 일꾼의 수.
    running: AtomicUsize,
    /// 잠든 일꾼들. 일꾼 `id`가 잠들어 있으면 `id`번째 비트가 1이다.
    asleep: AtomicU64,
    /// 누군가 이 풀에 일을 맡겨 둔 동안 참이다. 일은 한 번에 하나만 맡는다.
    busy: AtomicBool,
    panicked: AtomicBool,
    shutdown: AtomicBool,
    /// 잠든 스레드를 깨울 때 쓰는 잠금과 조건 변수 둘. `wake`는 일꾼을,
    /// `done`은 일을 맡긴 쪽을 깨운다.
    lock: Mutex<()>,
    wake: Condvar,
    done: Condvar,
    /// 기다리는 방법.
    waiting: Waiting,
}

// SAFETY: `job`만이 스스로 동기화되지 않는 칸이다. 이 칸은 `busy`를 차지한
// `run` 하나만, 참여한 일꾼이 모두 끝난 뒤(`running == 0`)에만 쓴다. 일꾼은
// 자기가 참여하는 새 호출을 본 뒤부터 `running`을 줄이기 전까지만 읽는다.
// 쓰기와 읽기가 겹치지 않는다. 칸에 든 포인터가 가리키는 함수는 `Sync`여서
// 여러 스레드가 동시에 불러도 된다.
unsafe impl Sync for Shared {}
// SAFETY: 위와 같다. 포인터는 `run`이 돌아오기 전까지만 쓰인다.
unsafe impl Send for Shared {}

impl Shared {
    /// 일꾼 `id`가 참여해야 하는, `seen`번보다 새로운 호출이 있으면 그 번호를
    /// 돌려준다.
    fn pending(&self, id: usize, seen: u64) -> Option<u64> {
        let call = Call::unpack(self.call.load(Ordering::SeqCst));
        (call.epoch != seen && id < call.workers).then_some(call.epoch)
    }
}
// ANCHOR_END: job

// ANCHOR: pool
/// 스레드 풀. 일꾼 스레드들을 쥐고 있다가 일을 나누어 준다.
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// 스레드 `threads`개로 일하는 풀을 만든다. 부르는 스레드도 일하므로
    /// 일꾼은 `threads - 1`개를 만든다. [`MAX_THREADS`]개를 넘지 못한다.
    ///
    /// `waiting`은 일꾼이 다음 일을, 일을 맡긴 쪽이 일꾼을 기다리는 방법이다.
    pub fn new(threads: usize, waiting: Waiting) -> Self {
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            call: AtomicU64::new(0),
            next: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            asleep: AtomicU64::new(0),
            busy: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            done: Condvar::new(),
            waiting,
        });
        let workers = (0..threads.clamp(1, MAX_THREADS) - 1)
            .map(|id| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("badak-{id}"))
                    .spawn(move || work(&shared, id))
                    .expect("스레드를 만들 수 있다")
            })
            .collect();
        Self { shared, workers }
    }

    /// 이 풀이 쓸 수 있는 스레드의 수. 부르는 스레드를 포함한다.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }
    // ANCHOR_END: pool

    // ANCHOR: run
    /// `task(0)`, `task(1)`, …, `task(count - 1)`을 스레드 `threads`개에
    /// 나누어 실행하고, 모두 끝나면 돌아온다.
    ///
    /// 어느 번호를 어느 스레드가 맡는지는 정해져 있지 않다. 스레드마다
    /// 남은 번호를 하나씩 뽑아 간다. 부르는 스레드도 함께 일한다.
    ///
    /// 풀이 이미 다른 일을 하고 있으면(다른 스레드가 맡겼거나, 일 안에서
    /// 다시 불렀으면) 나누지 않고 부르는 스레드가 혼자 다 한다.
    pub fn run(&self, count: usize, threads: usize, task: &(dyn Fn(usize) + Sync)) {
        let shared = &*self.shared;
        let workers = threads.clamp(1, self.threads()).min(count.max(1)) - 1;
        // 풀을 차지한다. 차지하지 못하면 혼자 한다.
        let free = workers > 0
            && shared
                .busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok();
        if !free {
            (0..count).for_each(task);
            return;
        }

        // 1. 일을 적고, 새 일이 왔다고 알린다.
        // SAFETY: 수명을 지운다. 이 함수는 아래의 `Wait`가 일꾼이 모두
        // 끝났음을 확인할 때까지 돌아가지 않으므로, 일꾼이 `task`를 쓰는
        // 동안 `task`는 살아 있다.
        let task_pointer: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
        // SAFETY: `busy`를 차지했으므로 이 칸을 쓰는 것은 우리뿐이고, 앞서
        // 차지했던 `run`이 `running == 0`을 확인하고 놓았으므로 지금 이 칸을
        // 읽는 일꾼도 없다.
        unsafe {
            *shared.job.get() = Some(Job {
                task: task_pointer,
                count,
            });
        }
        shared.next.store(0, Ordering::Relaxed);
        shared.running.store(workers, Ordering::Relaxed);
        let epoch = Call::unpack(shared.call.load(Ordering::Relaxed)).epoch + 1;
        shared
            .call
            .store(Call { epoch, workers }.pack(), Ordering::SeqCst);
        // 참여할 일꾼 가운데 잠든 것이 있으면 깨운다. 잠금을 쥐고 알려야 막
        // 잠들려는 일꾼과 엇갈리지 않는다.
        let participants = (1_u64 << workers) - 1;
        if shared.asleep.load(Ordering::SeqCst) & participants != 0 {
            let _guard = shared.lock.lock().expect("잠금이 오염되지 않았다");
            shared.wake.notify_all();
        }

        // 2. 부르는 스레드도 번호를 뽑아 가며 일한다. 여기서 패닉이 나더라도
        //    일꾼들이 끝나기 전에는 이 함수를 떠나지 않는다(`Wait`의 `Drop`).
        let wait = Wait(shared);
        claim(&shared.next, count, task);
        drop(wait);

        if shared.panicked.swap(false, Ordering::Relaxed) {
            panic!("일꾼 스레드가 일하다 패닉을 일으켰습니다");
        }
    }
    // ANCHOR_END: run

    // ANCHOR: run_chunks
    /// `data`를 `size`개씩 조각내어, 조각마다 `task(조각의 번호, 조각)`을
    /// 스레드 `threads`개에 나누어 실행한다.
    ///
    /// 18장에서 `run_chunks`라는 이름으로 처음 짠 것이다. 22장에서 잠금 없는
    /// [`run_chunks`](Self::run_chunks)로 바꿨고, 둘을 견주려고 남겨 두었다.
    ///
    /// 스레드마다 다른 조각에 쓰므로 서로의 쓰기가 겹치지 않는다. 그 사실을
    /// 컴파일러에게는 조각마다 잠금을 하나씩 씌워서 보인다. 조각 하나는 한
    /// 번만 잠기므로 기다리는 일은 없다.
    pub fn run_chunks_with_locks<T: Send>(
        &self,
        data: &mut [T],
        size: usize,
        threads: usize,
        task: &(dyn Fn(usize, &mut [T]) + Sync),
    ) {
        let chunks: Vec<Mutex<&mut [T]>> = data.chunks_mut(size).map(Mutex::new).collect();
        self.run(chunks.len(), threads, &|index| {
            let mut chunk = chunks[index].lock().expect("조각마다 한 번만 잠근다");
            task(index, &mut chunk);
        });
    }
    // ANCHOR_END: run_chunks

    // ANCHOR: run_chunks_unlocked
    /// `data`를 `size`개씩 조각내어, 조각마다 `task(조각의 번호, 조각)`을
    /// 스레드 `threads`개에 나누어 실행한다.
    ///
    /// 조각마다 잠금을 씌우지 않는다(22장). 잠금 하나를 처음 잠글 때 메모리를
    /// 할당하는 운영체제가 있어서, 잠금을 쓰면 조각의 수만큼 할당이 생긴다.
    pub fn run_chunks<T: Send>(
        &self,
        data: &mut [T],
        size: usize,
        threads: usize,
        task: &(dyn Fn(usize, &mut [T]) + Sync),
    ) {
        let (len, size) = (data.len(), size.max(1));
        let start = Unshared(data.as_mut_ptr());
        self.run(len.div_ceil(size), threads, &|index| {
            let offset = index * size;
            let count = size.min(len - offset);
            // SAFETY: `run`은 번호마다 `task`를 정확히 한 번 부른다. 번호가
            // 다르면 구간 `offset..offset + count`가 겹치지 않으므로, 같은
            // 원소를 가리키는 `&mut`가 둘 생기지 않는다. 구간은 모두 `data`
            // 안이고, `data`는 이 함수가 끝날 때까지 빌려 있다.
            let chunk = unsafe { std::slice::from_raw_parts_mut(start.at(offset), count) };
            task(index, chunk);
        });
    }
    // ANCHOR_END: run_chunks_unlocked
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        {
            let _guard = self.shared.lock.lock().expect("잠금이 오염되지 않았다");
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            worker.join().ok();
        }
    }
}

// ANCHOR: claim
/// 남은 번호를 하나씩 뽑아 `task`를 부른다. 번호가 떨어지면 돌아온다.
fn claim(next: &AtomicUsize, count: usize, task: &(dyn Fn(usize) + Sync)) {
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        if index >= count {
            return;
        }
        task(index);
    }
}
// ANCHOR_END: claim

// ANCHOR: work
/// 일꾼 스레드가 하는 일. 자기를 부르는 호출을 기다리고, 일하고, 끝났다고
/// 알린다.
fn work(shared: &Shared, id: usize) {
    let mut seen = 0;
    loop {
        let called =
            || shared.pending(id, seen).is_some() || shared.shutdown.load(Ordering::SeqCst);

        // 1. 잠들지 않은 채로 잠깐 기다려 본다. 일이 곧 또 올 때는 이쪽이
        //    빠르다. 잠들었다 깨어나는 데는 시간이 걸린다.
        shared.waiting.stay_awake(called);

        // 2. 그래도 부르지 않으면 잠든다. 부르면 운영체제가 깨워 준다.
        if !called() {
            shared.asleep.fetch_or(1 << id, Ordering::SeqCst);
            let mut guard = shared.lock.lock().expect("잠금이 오염되지 않았다");
            while !called() {
                guard = shared.wake.wait(guard).expect("잠금이 오염되지 않았다");
            }
            drop(guard);
            shared.asleep.fetch_and(!(1 << id), Ordering::SeqCst);
        }
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let Some(epoch) = shared.pending(id, seen) else {
            continue;
        };
        seen = epoch;

        // 3. 일한다. 일이 패닉을 일으켜도 일꾼은 죽지 않는다.
        // SAFETY: 우리가 참여하는 호출을 봤고 아직 `running`을 줄이지
        // 않았으므로 `run`은 이 칸을 바꾸지 않으며, 칸의 포인터가 가리키는
        // 함수도 살아 있다.
        let job = unsafe { *shared.job.get() }.expect("호출이 있으면 일도 있다");
        // SAFETY: 위와 같다.
        let task = unsafe { &*job.task };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            claim(&shared.next, job.count, task);
        }));
        if outcome.is_err() {
            shared.panicked.store(true, Ordering::Relaxed);
        }

        // 4. 끝났다고 알린다. 마지막 일꾼은 맡긴 쪽이 잠들어 있을 수 있으니
        //    깨운다.
        if shared.running.fetch_sub(1, Ordering::SeqCst) == 1 {
            let _guard = shared.lock.lock().expect("잠금이 오염되지 않았다");
            shared.done.notify_all();
        }
    }
}
// ANCHOR_END: work

// ANCHOR: wait
/// 여러 스레드가 함께 보는 포인터. 스레드마다 서로 다른 구간만 건드린다는
/// 것을 쓰는 쪽이 보장한다.
struct Unshared<T>(*mut T);

// SAFETY: 포인터 자체는 값일 뿐이다. 그것으로 무엇을 읽고 쓰는지는
// `run_chunks`의 `unsafe` 블록이 책임진다. `T: Send`이므로 원소를 다른
// 스레드에서 건드려도 된다.
unsafe impl<T: Send> Sync for Unshared<T> {}

impl<T> Unshared<T> {
    /// 처음에서 `offset`번째 원소를 가리키는 포인터.
    ///
    /// # Safety
    ///
    /// `offset`이 처음에 받은 슬라이스의 길이를 넘지 않아야 한다.
    unsafe fn at(&self, offset: usize) -> *mut T {
        // SAFETY: 부르는 쪽이 범위 안임을 보장한다.
        unsafe { self.0.add(offset) }
    }
}

/// 일꾼들이 모두 끝나기를 기다리는 값. 사라질 때 기다린다.
///
/// `run`이 어떤 길로 끝나든(패닉으로 풀리는 중이어도) 일꾼들이 `task`를 다
/// 쓴 뒤에야 `run`을 떠나게 한다. 수명을 지운 포인터가 안전한 근거다.
struct Wait<'a>(&'a Shared);

impl Drop for Wait<'_> {
    fn drop(&mut self) {
        let shared = self.0;
        // 일꾼과 마찬가지로, 먼저 잠들지 않은 채로 잠깐 기다려 본다.
        shared
            .waiting
            .stay_awake(|| shared.running.load(Ordering::SeqCst) == 0);
        let mut guard = shared
            .lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while shared.running.load(Ordering::SeqCst) > 0 {
            guard = shared
                .done
                .wait(guard)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        drop(guard);
        // 일꾼이 모두 끝났다. 이제 다른 쪽이 풀을 차지해도 된다.
        shared.busy.store(false, Ordering::Release);
    }
}
// ANCHOR_END: wait

// ANCHOR: global
/// 엔진이 함께 쓰는 스레드 풀과, 일의 종류마다 쓸 스레드의 수.
pub struct Threads {
    pool: ThreadPool,
    /// 토큰을 하나씩 만들 때(입력 벡터가 하나일 때) 쓸 스레드의 수.
    generation: AtomicUsize,
    /// 프롬프트를 처리할 때(입력 벡터가 여럿일 때) 쓸 스레드의 수.
    batch: AtomicUsize,
}

static THREADS: OnceLock<Threads> = OnceLock::new();

/// 엔진이 함께 쓰는 스레드 풀. 처음 부를 때 만든다.
///
/// 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다. 없으면 이 기계가
/// 동시에 돌릴 수 있는 스레드의 수를 쓴다. 기다리는 방법은 `BADAK_WAITING`
/// (`sleep`, `spin`, `yield`)으로 바꿀 수 있다. 실험하려고 둔 것이다.
pub fn global() -> &'static Threads {
    THREADS.get_or_init(|| {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let threads = std::env::var("BADAK_THREADS")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(available)
            .max(1);
        let waiting = match std::env::var("BADAK_WAITING").as_deref() {
            Ok("sleep") => Waiting::Sleep,
            Ok("spin") => Waiting::Spin(DEFAULT_AWAKE),
            _ => DEFAULT_WAITING,
        };
        Threads {
            pool: ThreadPool::new(threads, waiting),
            generation: AtomicUsize::new(threads.min(DEFAULT_GENERATION_THREADS)),
            batch: AtomicUsize::new(threads),
        }
    })
}

impl Threads {
    /// 풀.
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }

    /// 입력 벡터가 `count`개인 일에 쓸 스레드의 수.
    pub fn for_vectors(&self, count: usize) -> usize {
        let limit = if count <= 1 {
            &self.generation
        } else {
            &self.batch
        };
        limit.load(Ordering::Relaxed)
    }

    /// 일의 종류마다 쓸 스레드의 수를 정한다. 풀의 크기를 넘을 수 없다.
    pub fn set(&self, generation: usize, batch: usize) {
        let limit = self.pool.threads();
        self.generation
            .store(generation.clamp(1, limit), Ordering::Relaxed);
        self.batch.store(batch.clamp(1, limit), Ordering::Relaxed);
    }
}
// ANCHOR_END: global

/// 기다리는 방법의 기본값(18.3절).
pub const DEFAULT_WAITING: Waiting = Waiting::Yield(DEFAULT_AWAKE);
/// 잠들기 전에 깨어서 기다리는 시간.
pub const DEFAULT_AWAKE: Duration = Duration::from_micros(200);
/// 토큰을 하나씩 만들 때 쓰는 스레드 수의 기본 상한. 18.10절에서 6으로
/// 정했고, 양자화한 가중치에 맞춰 20.7절에서 8로 올렸다.
pub const DEFAULT_GENERATION_THREADS: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_exactly_once() {
        let short = Duration::from_micros(200);
        for waiting in [Waiting::Sleep, Waiting::Spin(short), Waiting::Yield(short)] {
            let pool = ThreadPool::new(4, waiting);
            for count in [0, 1, 3, 4, 100, 1000] {
                let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
                pool.run(count, 4, &|index| {
                    hits[index].fetch_add(1, Ordering::Relaxed);
                });
                assert!(
                    hits.iter().all(|hit| hit.load(Ordering::Relaxed) == 1),
                    "count {count}"
                );
            }
        }
    }

    #[test]
    fn borrows_from_the_caller_and_can_be_reused() {
        let pool = ThreadPool::new(3, Waiting::Sleep);
        let values: Vec<u64> = (1..=1000).collect();
        for _ in 0..50 {
            let total = AtomicU64::new(0);
            pool.run(values.len(), 3, &|index| {
                total.fetch_add(values[index], Ordering::Relaxed);
            });
            assert_eq!(total.load(Ordering::Relaxed), 500_500);
        }
    }

    #[test]
    fn locked_and_unlocked_chunks_do_the_same() {
        let pool = ThreadPool::new(4, DEFAULT_WAITING);
        for (len, size) in [(0, 3), (1, 1), (10, 3), (1000, 7), (64, 64), (5, 100)] {
            let work = |index: usize, chunk: &mut [usize]| {
                let len = chunk.len();
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value += index * 1000 + offset + len;
                }
            };
            let mut locked = vec![1_usize; len];
            pool.run_chunks_with_locks(&mut locked, size, 4, &work);
            let mut unlocked = vec![1_usize; len];
            pool.run_chunks(&mut unlocked, size, 4, &work);
            assert_eq!(locked, unlocked, "길이 {len}, 조각 {size}");
        }
    }

    #[test]
    fn chunks_are_each_written_once() {
        let pool = ThreadPool::new(4, DEFAULT_WAITING);
        let mut data = vec![0_usize; 1003];
        pool.run_chunks(&mut data, 10, 4, &|index, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                *value += index * 10 + offset;
            }
        });
        assert!(
            data.iter()
                .enumerate()
                .all(|(index, &value)| index == value)
        );
    }

    #[test]
    fn uses_no_more_threads_than_asked() {
        let pool = ThreadPool::new(8, Waiting::Sleep);
        let names = Mutex::new(std::collections::BTreeSet::new());
        pool.run(64, 2, &|_| {
            let name = std::thread::current().name().map(str::to_string);
            names.lock().unwrap().insert(name);
            std::thread::sleep(Duration::from_millis(1));
        });
        assert!(names.lock().unwrap().len() <= 2);
        // 스레드 하나만 쓰면 부르는 스레드가 모두 한다.
        let caller = std::thread::current().id();
        pool.run(8, 1, &|_| assert_eq!(std::thread::current().id(), caller));
    }

    #[test]
    fn a_busy_pool_makes_the_caller_work_alone() {
        let pool = ThreadPool::new(4, DEFAULT_WAITING);
        // 일 안에서 같은 풀을 다시 부른다. 멈추지 않고 끝나야 한다.
        let total = AtomicUsize::new(0);
        pool.run(8, 4, &|_| {
            let inner = std::thread::current().id();
            pool.run(8, 4, &|index| {
                assert_eq!(std::thread::current().id(), inner);
                total.fetch_add(index, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 8 * 28);
        // 여러 스레드가 한꺼번에 맡겨도 번호마다 한 번씩 실행된다.
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..200 {
                        let hits: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();
                        pool.run(16, 4, &|index| {
                            hits[index].fetch_add(1, Ordering::Relaxed);
                        });
                        assert!(hits.iter().all(|hit| hit.load(Ordering::Relaxed) == 1));
                    }
                });
            }
        });
    }

    #[test]
    fn a_panic_in_a_task_reaches_the_caller_and_the_pool_survives() {
        let pool = ThreadPool::new(4, DEFAULT_WAITING);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(64, 4, &|index| {
                std::thread::sleep(Duration::from_millis(1));
                assert!(index != 13, "일부러 낸 패닉");
            });
        }));
        assert!(result.is_err());
        // 패닉 뒤에도 풀을 그대로 쓸 수 있다.
        let total = AtomicUsize::new(0);
        pool.run(10, 4, &|index| {
            total.fetch_add(index, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 45);
    }
}
