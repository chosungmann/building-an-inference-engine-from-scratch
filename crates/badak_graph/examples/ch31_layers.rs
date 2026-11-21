//! 31장: 층 하나짜리 그래프의 시간을 잰다. `reference/ch31_layers.py`가 다른
//! 엔진들로 재는 것과 같은 층들이다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch31_layers`
//!
//! 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다. 합성곱은 28장의 커널
//! (NEON)과 이 장의 커널(SME)로 한 번씩 잰다. 깊이별 합성곱에는 SME 커널이
//! 없다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::{Tensor, threads};
use badak_graph::{Activation, Graph, Interpreter, Op, Padding, Registry};

/// (이름, 입력 채널, 출력 채널, 한 변, 창, 스트라이드, 깊이별인지)
const LAYERS: [(&str, usize, usize, usize, usize, usize, bool); 9] = [
    ("conv 7×7 스트라이드 2, 3→64 @224", 3, 64, 224, 7, 2, false),
    ("conv 3×3, 64→64 @56", 64, 64, 56, 3, 1, false),
    ("conv 1×1, 64→256 @56", 64, 256, 56, 1, 1, false),
    ("conv 3×3, 256→256 @14", 256, 256, 14, 3, 1, false),
    ("conv 1×1, 1024→256 @14", 1024, 256, 14, 1, 1, false),
    ("conv 3×3, 512→512 @7", 512, 512, 7, 3, 1, false),
    ("conv 1×1, 16→96 @112", 16, 96, 112, 1, 1, false),
    ("depthwise 3×3, 144 @56", 144, 144, 56, 3, 1, true),
    ("depthwise 3×3, 960 @7", 960, 960, 7, 3, 1, true),
];

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.07 - 1.0)
        .collect()
}

fn time(graph: &Graph, registry: &Registry) -> Result<f64, Box<dyn std::error::Error>> {
    let mut interpreter = Interpreter::new(graph, registry)?;
    let input = values(interpreter.input_mut(0).len(), 3);
    let plan = Plan {
        warmup: 5,
        repetitions: 51,
    };
    let samples = measure(plan, || {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().expect("실행");
        black_box(interpreter.output(0)[0]);
    });
    Ok(Summary::new(&samples).median * 1e6)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plain = Registry::builtin();
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut outer: Option<Registry> = None;
    #[cfg(target_arch = "aarch64")]
    if badak_core::simd::sme::available() {
        let mut registry = Registry::builtin();
        registry.register("CONV_2D", badak_graph::outer::conv_2d_outer);
        outer = Some(registry);
    }
    println!(
        "층 하나의 시간 (마이크로초), 스레드 {}개",
        threads::global().pool().threads()
    );
    println!("{:<28} {:>9} {:>9}", "층", "NEON", "SME");
    for (name, in_channels, out_channels, side, window, stride, depthwise) in LAYERS {
        let pad = window / 2;
        let padding = Padding::Explicit([[pad, pad]; 2]);
        let mut graph = Graph::new(name);
        let x = graph.value("x", &[1, side, side, in_channels]);
        let y = graph.value("y", &[]);
        let (op, shape) = if depthwise {
            let op = Op::DepthwiseConv2d {
                padding,
                stride: [stride; 2],
                dilation: [1, 1],
                multiplier: 1,
                activation: Activation::None,
            };
            (op, [1, window, window, out_channels])
        } else {
            let op = Op::Conv2d {
                padding,
                stride: [stride; 2],
                dilation: [1, 1],
                activation: Activation::None,
            };
            (op, [out_channels, window, window, in_channels])
        };
        let filter = Tensor::from_vec(values(shape.iter().product(), 1), &shape)?;
        let w = graph.constant("w", filter);
        let bias = Tensor::from_vec(values(out_channels, 2), &[out_channels])?;
        let b = graph.constant("b", bias);
        graph.node(op, &[x, w, b], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        graph.infer_shapes(&plain)?;
        let neon = time(&graph, &plain)?;
        let fast = match (&outer, depthwise) {
            (Some(registry), false) => format!("{:>9.0}", time(&graph, registry)?),
            _ => format!("{:>9}", "-"),
        };
        println!("{name:<28} {neon:>9.0} {fast}");
    }
    Ok(())
}
