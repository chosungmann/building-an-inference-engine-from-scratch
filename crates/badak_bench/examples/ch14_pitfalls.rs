//! 14장: 측정을 망치는 것들을 하나씩 겪어 본다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch14_pitfalls -- [실험]`
//!
//! 실험은 `clock`, `optimizer`, `cold`, `noise`, `work` 가운데 하나다. 주지
//! 않으면 앞의 넷을 차례로 한다. `work`는 일 하나의 시간만 재며,
//! `taskpolicy -b`로 효율 코어에서 돌려 볼 때 쓴다.

use std::hint::black_box;
use std::time::{Duration, Instant};

use badak_bench::{Counters, Plan, Summary, measure};
use badak_core::linalg::matvec;

fn microseconds(seconds: f64) -> f64 {
    seconds * 1e6
}

fn print_summary(label: &str, samples: &[f64]) {
    let s = Summary::new(samples);
    println!(
        "{:<26} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
        label,
        microseconds(s.min),
        microseconds(s.median),
        microseconds(s.mean),
        microseconds(s.p90),
        microseconds(s.p99),
        microseconds(s.max)
    );
}

fn summary_header() {
    println!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "(단위: 마이크로초)", "최소", "중앙값", "평균", "p90", "p99", "최대"
    );
}

/// 재 볼 일: `512 × 512` 행렬과 벡터의 곱. 0.2밀리초쯤 걸린다.
struct Work {
    matrix: Vec<f32>,
    vector: Vec<f32>,
}

impl Work {
    const SIZE: usize = 512;

    fn new() -> Self {
        Self {
            matrix: (0..Self::SIZE * Self::SIZE)
                .map(|i| (i % 7) as f32 * 0.25)
                .collect(),
            vector: (0..Self::SIZE).map(|i| (i % 5) as f32 * 0.5).collect(),
        }
    }

    fn run(&self) -> Vec<f32> {
        matvec(
            black_box(&self.matrix),
            Self::SIZE,
            Self::SIZE,
            black_box(&self.vector),
        )
    }
}

// ANCHOR: clock
/// 시계 자체의 값: 시각을 한 번 읽는 데 드는 시간과, 구별할 수 있는 가장
/// 짧은 간격.
fn clock() {
    let rounds = 1_000_000;
    let start = Instant::now();
    for _ in 0..rounds {
        black_box(Instant::now());
    }
    let cost = start.elapsed().as_secs_f64() / f64::from(rounds);

    // 시각을 연달아 읽어, 0이 아닌 차이 가운데 가장 작은 것을 찾는다.
    let mut finest = Duration::MAX;
    for _ in 0..rounds {
        let (a, b) = (Instant::now(), Instant::now());
        let gap = b - a;
        if gap > Duration::ZERO {
            finest = finest.min(gap);
        }
    }
    println!("시각을 한 번 읽는 시간     {:>8.1} 나노초", cost * 1e9);
    println!("구별되는 가장 짧은 간격    {:>8} 나노초", finest.as_nanos());
}
// ANCHOR_END: clock

// ANCHOR: optimizer
/// 1부터 `n`까지 더한다.
fn sum_to(n: u64) -> u64 {
    (1..=n).sum()
}

/// 컴파일러가 측정하려던 일을 지워 버리는 경우.
fn optimizer() {
    let n = 100_000_000_u64;
    let plan = Plan {
        warmup: 1,
        repetitions: 5,
    };

    // 입력이 상수이고 결과를 쓰지 않으면 컴파일러가 계산을 미리 해 둔다.
    let start = Instant::now();
    let total = sum_to(n);
    let naive = start.elapsed();

    // 입력을 가리면 컴파일러는 `n`이 무엇인지 모르는 채로 코드를 만들어야 한다.
    let guarded = Summary::new(&measure(plan, || sum_to(black_box(n))));

    // 반복문 안의 값을 하나하나 가리면 정말로 1억 번 돈다.
    let looped = Summary::new(&measure(plan, || {
        let mut sum = 0_u64;
        for i in 1..=n {
            sum = black_box(sum + i);
        }
        sum
    }));

    println!("1부터 1억까지의 합 {total}");
    println!(
        "그냥 잰 시간                {:>12.0} 나노초",
        naive.as_secs_f64() * 1e9
    );
    println!(
        "입력을 가린 시간            {:>12.0} 나노초",
        guarded.median * 1e9
    );
    println!(
        "반복마다 값을 가린 시간     {:>12.0} 나노초",
        looped.median * 1e9
    );
}
// ANCHOR_END: optimizer

// ANCHOR: cold
/// 처음 몇 번은 느리다.
fn cold() {
    // 0으로 채운 큰 메모리를 달라고 하면 운영체제는 주소만 내주고, 실제
    // 메모리는 그 쪽에 처음 닿는 순간에 붙여 준다. 그 일이 첫 실행의 시간에
    // 섞인다.
    let size = 4096;
    let matrix = vec![0.0_f32; size * size];
    let vector = vec![1.0_f32; size];
    println!("{:<8} {:>12}", "차례", "밀리초");
    for round in 1..=6 {
        let start = Instant::now();
        black_box(matvec(black_box(&matrix), size, size, black_box(&vector)));
        println!("{:<8} {:>12.3}", round, start.elapsed().as_secs_f64() * 1e3);
    }
}
// ANCHOR_END: cold

// ANCHOR: noise
/// 같은 일을 조용할 때와 시끄러울 때 잰다.
fn noise() {
    let work = Work::new();
    let plan = Plan {
        warmup: 20,
        repetitions: 2000,
    };
    summary_header();
    print_summary("조용할 때", &measure(plan, || work.run()));

    // 논리 코어 수의 두 배만큼 스레드를 띄워 CPU를 다투게 한다.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let threads = 2 * std::thread::available_parallelism().map_or(8, usize::from);
    let spinners: Vec<_> = (0..threads)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut x = 0_u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    x = black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
                }
            })
        })
        .collect();
    print_summary("다른 일이 CPU를 다툴 때", &measure(plan, || work.run()));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for spinner in spinners {
        spinner.join().expect("스레드가 정상 종료한다");
    }
}
// ANCHOR_END: noise

fn work() {
    let work = Work::new();
    let plan = Plan {
        warmup: 20,
        repetitions: 2000,
    };
    let (before, start) = (Counters::now(), Instant::now());
    let samples = measure(plan, || work.run());
    let elapsed = start.elapsed().as_secs_f64();
    summary_header();
    print_summary("행렬과 벡터의 곱", &samples);
    // 사이클 수를 시간으로 나누면 프로세서가 실제로 돈 속도가 나온다.
    if let Some(spent) = before
        .zip(Counters::now())
        .map(|(before, after)| after.since(before))
    {
        println!(
            "프로세서가 돈 속도         {:>8.2} GHz",
            spent.cycles as f64 / elapsed / 1e9
        );
        println!(
            "사이클당 명령 (IPC)        {:>8.2}",
            spent.instructions_per_cycle()
        );
    }
}

fn main() {
    let experiment = std::env::args().nth(1);
    let all = experiment.is_none();
    let wants = |name: &str| all || experiment.as_deref() == Some(name);
    if wants("clock") {
        println!("[시계]");
        clock();
        println!();
    }
    if wants("optimizer") {
        println!("[컴파일러]");
        optimizer();
        println!();
    }
    if wants("cold") {
        println!("[첫 실행]");
        cold();
        println!();
    }
    if wants("noise") {
        println!("[소음]");
        noise();
        println!();
    }
    if experiment.as_deref() == Some("work") {
        work();
    }
}
