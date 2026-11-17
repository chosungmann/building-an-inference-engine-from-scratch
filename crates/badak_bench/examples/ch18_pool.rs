//! 18장: 일을 스레드에 나누는 일 자체의 값을 잰다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch18_pool`
//!
//! 스레드 12개에 아무 일도 하지 않는 일을 나누어 주고, 모두 끝났다는 것을
//! 확인할 때까지의 시간을 잰다. 스레드를 마련하고 기다리는 방법 넷을 견준다.
//! 일과 일 사이의 간격도 바꿔 본다. 일꾼이 잠들 틈이 있는지에 따라 값이
//! 달라진다. 풀이 깨어서 기다리는 시간은 0.2밀리초다.

use std::time::{Duration, Instant};

use badak_bench::Summary;
use badak_core::threads::{ThreadPool, Waiting};

const THREADS: usize = 12;

// ANCHOR: spawn
/// 방법 1: 일이 올 때마다 스레드를 새로 만든다.
fn spawn_each_time(task: &(dyn Fn(usize) + Sync)) {
    std::thread::scope(|scope| {
        for index in 1..THREADS {
            scope.spawn(move || task(index));
        }
        task(0);
    });
}
// ANCHOR_END: spawn

/// 일 사이에 `gap`만큼 쉬면서 `run`을 여러 번 부르고, 한 번의 시간을 모은다.
fn measure(gap: Duration, mut run: impl FnMut()) -> Summary {
    let rounds = if gap > Duration::from_millis(1) {
        200
    } else {
        2000
    };
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        // 쉬는 동안 CPU를 쓰지 않는다. 일꾼이 잠들 수 있다.
        if gap > Duration::ZERO {
            std::thread::sleep(gap);
        }
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64());
    }
    Summary::new(&samples)
}

fn main() {
    let nothing = |_: usize| {};
    let short = Duration::from_micros(200);
    let sleeping = ThreadPool::new(THREADS, Waiting::Sleep);
    let spinning = ThreadPool::new(THREADS, Waiting::Spin(short));
    let yielding = ThreadPool::new(THREADS, Waiting::Yield(short));

    println!("스레드 {THREADS}개에 빈 일을 나누어 주는 데 걸리는 시간 (마이크로초)");
    println!(
        "{:<22} {:<28} {:>9} {:>9} {:>9}",
        "일 사이의 간격", "방법", "중앙값", "p90", "p99"
    );
    for (label, gap) in [
        ("없음", Duration::ZERO),
        ("0.1밀리초", Duration::from_micros(100)),
        ("10밀리초", Duration::from_millis(10)),
    ] {
        let rows: [(&str, Summary); 4] = [
            (
                "매번 스레드를 만든다",
                measure(gap, || spawn_each_time(&nothing)),
            ),
            (
                "풀: 곧바로 잠든다",
                measure(gap, || sleeping.run(THREADS, THREADS, &nothing)),
            ),
            (
                "풀: 코어를 쥐고 기다린다",
                measure(gap, || spinning.run(THREADS, THREADS, &nothing)),
            ),
            (
                "풀: 양보하며 기다린다",
                measure(gap, || yielding.run(THREADS, THREADS, &nothing)),
            ),
        ];
        for (method, summary) in rows {
            println!(
                "{:<22} {:<28} {:>9.1} {:>9.1} {:>9.1}",
                label,
                method,
                summary.median * 1e6,
                summary.p90 * 1e6,
                summary.p99 * 1e6
            );
        }
    }
}
