//! 26장: 해석기가 세운 계획을 본다. 실행할 순서, 값마다의 수명, 자리.
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_plan -- residual 1`
//!
//! 둘째 인자는 한 번에 넣을 사진의 수다.

use badak_formats::tflite::Model;
use badak_graph::memory::{self, Strategy};
use badak_graph::{Graph, Interpreter, Registry, models_dir};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "residual".into());
    let batch: usize = arguments.next().map_or(Ok(1), |text| text.parse())?;
    let model = Model::open(models_dir().join(format!("mnist_tflite/{name}.tflite")))?;
    let graph = Graph::from_tflite(&model)?;
    let registry = Registry::builtin();
    let shape: &[usize] = &[batch, 784];
    let interpreter = Interpreter::with(&graph, &registry, &[shape], Strategy::BySize)?;

    println!("{name}.tflite, 한 번에 {batch}장");
    println!("\n실행 계획");
    for (index, step) in interpreter.steps().enumerate() {
        // 입력 가운데 앞의 것만 계산되는 값이고 나머지는 가중치다.
        let (_, input) = step.inputs[0];
        let (_, output) = step.outputs[0];
        println!("  {index:>2} {:<16} {input:?} → {output:?}", step.name);
    }

    println!("\n값의 수명과 자리 (큰 것부터 놓기)");
    println!(
        "  {:<4} {:>8} {:>6} {:>6} {:>10}",
        "값", "크기", "처음", "끝", "자리"
    );
    for (index, (_, lifetime, offset)) in interpreter.placements().enumerate() {
        println!(
            "  t{index:<3} {:>8} {:>6} {:>6} {:>10}",
            lifetime.size, lifetime.first, lifetime.last, offset
        );
    }

    let lifetimes = interpreter.lifetimes();
    println!("\n자리 전체의 크기 (f32의 수, 바이트)");
    let report = |label: &str, total: usize| {
        println!("  {label:<28} {total:>10} {:>12}", total * 4);
    };
    report(
        "값마다 따로",
        memory::plan(&lifetimes, Strategy::Separate).total,
    );
    report(
        "순서대로 놓기",
        memory::plan(&lifetimes, Strategy::InOrder).total,
    );
    report(
        "큰 것부터 놓기",
        memory::plan(&lifetimes, Strategy::BySize).total,
    );
    report("이보다 작을 수 없는 크기", memory::lower_bound(&lifetimes));
    Ok(())
}
