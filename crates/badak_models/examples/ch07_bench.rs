//! 7장: 첫 벤치마크. 시험 데이터 전체를 분류하는 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_models --example ch07_bench`
//!
//! 초시계 수준의 거친 측정이다. 제대로 된 측정 방법은 14장에서 다룬다.

use std::hint::black_box;
use std::time::Instant;

use badak_formats::idx;
use badak_models::mnist::MnistMlp;
use badak_models::models_dir;

/// 여러 번 실행해 가장 빠른 시간을 초 단위로 돌려준다.
fn best_of<R>(repeats: usize, mut f: impl FnMut() -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let (load, model) = {
        let start = Instant::now();
        let model = MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?;
        (start.elapsed().as_secs_f64(), model)
    };
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte"))?;
    let count = images.shape()[0];
    let input = model.preprocess(&images)?;

    let preprocess = best_of(5, || model.preprocess(&images).unwrap());
    let batch = best_of(5, || model.forward(&input).unwrap());
    // 한 장씩 따로 넣는 경우: 앞의 1000장.
    let singles: Vec<_> = (0..1000)
        .map(|i| input.slice(0, i..i + 1).unwrap())
        .collect();
    let one_by_one = best_of(5, || {
        for x in &singles {
            black_box(model.forward(x).unwrap());
        }
    });

    println!(
        "badak, 스레드 {}개",
        badak_core::threads::global().pool().threads()
    );
    println!("{:<34} {:>10.3} ms", "모델 읽기", load * 1e3);
    println!(
        "{:<34} {:>10.3} ms",
        format!("전처리, {count}장"),
        preprocess * 1e3
    );
    println!(
        "{:<34} {:>10.3} ms",
        format!("순전파, {count}장을 한 번에"),
        batch * 1e3
    );
    println!(
        "{:<34} {:>10.1} 장",
        "  1초에 처리하는 사진",
        count as f64 / batch
    );
    println!(
        "{:<34} {:>10.3} µs",
        "순전파, 한 장씩 (장당)",
        one_by_one / 1000.0 * 1e6
    );
    Ok(())
}
