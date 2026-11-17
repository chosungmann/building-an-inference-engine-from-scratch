//! 18장: 선형 층의 일을 스레드에 나누는 방법을 견준다.
//!
//! 실행: `cargo run --release -p badak_bench --example ch18_split`
//!
//! 세 가지를 잰다. 무엇을 나눌지(가중치의 행, 입력 벡터), 몇 조각으로
//! 나눌지, 얼마나 작은 일부터 나누지 않을지.

use std::hint::black_box;

use badak_bench::{Plan, Summary, measure};
use badak_core::simd::{Backend, Split, Walk, linear_split, linear_with};
use badak_core::threads::{DEFAULT_WAITING, ThreadPool};

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 31 + seed * 17) % 19) as f32 * 0.125 - 1.0)
        .collect()
}

/// 선형 층 하나의 입력과 가중치와 출력.
struct Layer {
    x: Vec<f32>,
    w: Vec<f32>,
    y: Vec<f32>,
    count: usize,
    rows: usize,
    cols: usize,
}

impl Layer {
    fn new(count: usize, rows: usize, cols: usize) -> Self {
        Self {
            x: values(count * cols, 1),
            w: values(rows * cols, 2),
            y: vec![0.0; count * rows],
            count,
            rows,
            cols,
        }
    }

    /// 1초에 하는 곱셈의 수(십억).
    fn speed(&mut self, repetitions: usize, mut run: impl FnMut(&mut Self)) -> f64 {
        let plan = Plan {
            warmup: 2,
            repetitions,
        };
        let seconds = Summary::new(&measure(plan, || {
            run(self);
            black_box(self.y[0])
        }))
        .median;
        (self.count * self.rows * self.cols) as f64 / seconds / 1e9
    }

    fn split(&mut self, pool: &ThreadPool, threads: usize, pieces: usize, split: Split) {
        let (x, w, y) = (&self.x, &self.w, &mut self.y);
        linear_split(
            pool, threads, pieces, split, x, self.count, w, self.rows, self.cols, y,
        );
    }
}

fn main() {
    let pool = ThreadPool::new(16, DEFAULT_WAITING);
    // Llama 3.2 1B의 MLP에서 가장 큰 가중치와 같은 모양이다.
    let (rows, cols) = (8192, 2048);

    println!("1. 무엇을 나눌까: 가중치 ({rows}, {cols}), 스레드 12개, 스레드마다 4조각");
    println!(
        "{:>10} {:>22} {:>22}",
        "벡터", "행을 나눈다", "벡터를 나눈다"
    );
    for count in [16, 64, 256] {
        let mut layer = Layer::new(count, rows, cols);
        let by_rows = layer.speed(7, |layer| layer.split(&pool, 12, 4, Split::Rows));
        let by_vectors = layer.speed(7, |layer| layer.split(&pool, 12, 4, Split::Vectors));
        println!("{count:>10} {by_rows:>12.1}십억 곱셈/초 {by_vectors:>12.1}십억 곱셈/초");
    }

    println!();
    println!("2. 몇 조각으로 나눌까: 가중치 ({rows}, {cols}), 십억 곱셈/초");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "벡터", "스레드", "1조각", "2조각", "4조각", "8조각", "16조각"
    );
    for count in [1, 256] {
        let mut layer = Layer::new(count, rows, cols);
        let repetitions = if count == 1 { 200 } else { 7 };
        for threads in [6, 12, 16] {
            print!("{count:>10} {threads:>10}");
            for pieces in [1, 2, 4, 8, 16] {
                let speed = layer.speed(repetitions, |layer| {
                    layer.split(&pool, threads, pieces, Split::Rows);
                });
                print!(" {speed:>10.1}");
            }
            println!();
        }
    }

    println!();
    println!("3. 얼마나 작은 일부터 나누지 않을까: 벡터 하나, 마이크로초");
    println!(
        "{:>14} {:>12} {:>12} {:>12} {:>10}",
        "가중치", "곱셈", "스레드 1개", "스레드 6개", "배수"
    );
    let backend = Backend::detect();
    for size in [128, 256, 384, 512, 768, 1024, 1536, 2048] {
        let mut layer = Layer::new(1, size, size);
        let work = size * size;
        let repetitions = 2000;
        let alone = work as f64
            / layer.speed(repetitions, |layer| {
                let (x, w, y) = (&layer.x, &layer.w, &mut layer.y);
                linear_with(backend, Walk::FourStreams, x, 1, w, size, size, y);
            })
            / 1e3;
        let shared = work as f64
            / layer.speed(repetitions, |layer| layer.split(&pool, 6, 4, Split::Rows))
            / 1e3;
        println!(
            "{:>14} {:>12} {:>12.1} {:>12.1} {:>9.2}배",
            format!("({size}, {size})"),
            work,
            alone,
            shared,
            alone / shared
        );
    }
}
