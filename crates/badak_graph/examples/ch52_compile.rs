//! 52장: 같은 그래프를 해석기로 돌린 것과, Rust 소스로 옮겨 컴파일한 것을 견준다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch52_compile -- [mlp|glue|kernels ...]`
//!
//! 소스는 `target/ch52/` 아래에 적고 `rustc`로 컴파일한다(`rustc`가 길에
//! 있어야 한다). 옮기는 코드는 `crates/badak_graph/src/codegen.rs`다.
//!
//!     mlp      7장의 신경망(784 → 128 → 10)
//!     glue     작은 연산 예순넷이 한 줄로 이어진 그래프. 길이 64와 65,536
//!     kernels  내적 커널의 생김새를 바꿔 가며 가장 빠른 것을 찾는다
//!     ir       7장의 신경망이 그래프, 반복문, 소스를 지나는 모습을 적는다

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use badak_core::Tensor;
use badak_formats::safetensors::SafeTensors;
use badak_graph::codegen::{self, Options};
use badak_graph::{Activation, Graph, Interpreter, Op, Registry, models_dir};

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

/// 생성된 소스 뒤에 붙이는 `main`. 입력 파일을 읽어 `rounds`번씩 아홉 묶음을
/// 돌리고, 가장 빨랐던 묶음의 한 번의 시간과 결과의 비트를 적는다.
const HARNESS: &str = r#"
fn main() {
    let mut arguments = std::env::args().skip(1);
    let bytes = std::fs::read(arguments.next().unwrap()).unwrap();
    let rounds: usize = arguments.next().unwrap().parse().unwrap();
    let values: Vec<f32> = bytes.chunks_exact(4).map(|four| f32::from_le_bytes(four.try_into().unwrap())).collect();
    let input: &[f32; INPUT] = (&values[..]).try_into().unwrap();
    let mut state = State::new();
    let mut output = Box::new([0.0_f32; OUTPUT]);
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            run(&mut state, std::hint::black_box(input), &mut output);
            std::hint::black_box(&output);
        }
        best = best.min(started.elapsed().as_secs_f64() / rounds as f64);
    }
    println!("ns {:.1}", best * 1e9);
    let bits: Vec<String> = output.iter().map(|value| format!("{:08x}", value.to_bits())).collect();
    println!("bits {}", bits.join(" "));
}
"#;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ch52")
}

/// 소스를 `rustc`로 컴파일하고 실행 파일의 길과 걸린 시간을 돌려준다.
fn build(name: &str, source: &str) -> Fallible<(PathBuf, f64)> {
    let directory = workspace().join(name);
    std::fs::create_dir_all(&directory)?;
    let (file, binary) = (directory.join("main.rs"), directory.join("main"));
    std::fs::write(&file, source)?;
    let started = Instant::now();
    let status = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "-C",
            "opt-level=3",
            "-C",
            "debuginfo=0",
            "-A",
            "warnings",
            "-o",
        ])
        .arg(&binary)
        .arg(&file)
        .status()?;
    if !status.success() {
        return Err(format!("rustc가 {} 를 컴파일하지 못했습니다", file.display()).into());
    }
    Ok((binary, started.elapsed().as_secs_f64()))
}

/// 컴파일한 것을 돌려 한 번의 시간(ns)과 결과를 돌려준다.
fn execute(binary: &Path, input: &[f32], rounds: usize) -> Fallible<(f64, Vec<f32>)> {
    let file = binary.with_file_name("input.bin");
    let bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
    std::fs::write(&file, bytes)?;
    let output = Command::new(binary)
        .arg(&file)
        .arg(rounds.to_string())
        .output()?;
    let text = String::from_utf8(output.stdout)?;
    let mut lines = text.lines();
    let nanoseconds: f64 = lines
        .next()
        .and_then(|line| line.strip_prefix("ns "))
        .ok_or("시간이 없습니다")?
        .parse()?;
    let bits = lines
        .next()
        .and_then(|line| line.strip_prefix("bits "))
        .ok_or("결과가 없습니다")?;
    let values = bits
        .split(' ')
        .map(|word| u32::from_str_radix(word, 16).map(f32::from_bits))
        .collect::<Result<_, _>>()?;
    Ok((nanoseconds, values))
}

/// 해석기로 돌려 한 번의 시간(ns)과 결과를 돌려준다. 재는 방법은 위와 같다.
fn interpret(
    graph: &Graph,
    input: &[f32],
    rounds: usize,
) -> Fallible<(f64, Vec<f32>, usize, usize)> {
    let registry = Registry::builtin();
    let mut interpreter = Interpreter::new(graph, &registry)?;
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let started = Instant::now();
        for _ in 0..rounds {
            interpreter.input_mut(0).copy_from_slice(black_box(input));
            interpreter.run()?;
            black_box(interpreter.output(0));
        }
        best = best.min(started.elapsed().as_secs_f64() / rounds as f64);
    }
    Ok((
        best * 1e9,
        interpreter.output(0).to_vec(),
        interpreter.step_count(),
        interpreter.arena_len(),
    ))
}

/// 7장의 신경망을 그래프로 짓는다. 선형 층, ReLU, 선형 층.
fn mlp() -> Fallible<Graph> {
    let weights = SafeTensors::open(models_dir().join("mnist_mlp/model.safetensors"))?;
    let mut graph = Graph::new("mlp");
    let x = graph.value("x", &[1, 784]);
    let constant = |graph: &mut Graph, name: &str| -> Fallible<usize> {
        Ok(graph.constant(name, weights.tensor(name)?))
    };
    let (w1, b1) = (
        constant(&mut graph, "fc1.weight")?,
        constant(&mut graph, "fc1.bias")?,
    );
    let (w2, b2) = (
        constant(&mut graph, "fc2.weight")?,
        constant(&mut graph, "fc2.bias")?,
    );
    let (hidden, active, y) = (
        graph.value("hidden", &[1, 128]),
        graph.value("active", &[1, 128]),
        graph.value("y", &[1, 10]),
    );
    let plain = Op::FullyConnected {
        activation: Activation::None,
        keep_num_dims: false,
    };
    graph.node(plain.clone(), &[x, w1, b1], &[hidden]);
    graph.node(Op::Relu, &[hidden], &[active]);
    graph.node(plain, &[active, w2, b2], &[y]);
    graph.inputs = vec![x];
    graph.outputs = vec![y];
    Ok(graph)
}

// ANCHOR: glue
/// 작은 연산 `depth`개가 한 줄로 이어진 그래프. 값은 `len`개다.
///
/// 더하기, 곱하기, ReLU, 자르기를 돌아가며 놓는다. 큰 모델에서 선형 층과
/// 선형 층 사이를 잇는 일들(정규화의 조각, 활성화 함수, 잔차 더하기)을
/// 흉내 낸 것이다.
fn glue(len: usize, depth: usize) -> Fallible<Graph> {
    let mut graph = Graph::new("glue");
    let mut current = graph.value("x", &[len]);
    graph.inputs = vec![current];
    // 더할 값과 곱할 값. 마디들이 함께 쓴다.
    let offsets: Vec<f32> = (0..len)
        .map(|i| ((i * 7) % 13) as f32 * 0.01 - 0.06)
        .collect();
    let scales: Vec<f32> = (0..len)
        .map(|i| 1.0 + ((i * 5) % 7) as f32 * 0.004)
        .collect();
    let offsets = graph.constant("offsets", Tensor::from_vec(offsets, &[len])?);
    let scales = graph.constant("scales", Tensor::from_vec(scales, &[len])?);
    let none = Activation::None;
    for index in 0..depth {
        let next = graph.value(&format!("v{index}"), &[len]);
        match index % 4 {
            0 => graph.node(Op::Add { activation: none }, &[current, offsets], &[next]),
            1 => graph.node(Op::Mul { activation: none }, &[current, scales], &[next]),
            2 => graph.node(Op::Relu, &[current], &[next]),
            _ => graph.node(Op::Clip { min: 0.0, max: 2.0 }, &[current], &[next]),
        }
        current = next;
    }
    graph.outputs = vec![current];
    Ok(graph)
}
// ANCHOR_END: glue

fn difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// 그래프 하나를 여러 가지로 돌려 적는다: 해석기, 옮긴 것(합치지 않고), 옮긴
/// 것(합쳐서), 그리고 `tuned`를 주면 그 생김새의 커널로 옮긴 것.
fn compare(
    label: &str,
    graph: &Graph,
    input: &[f32],
    rounds: usize,
    tuned: Option<Options>,
) -> Fallible<()> {
    let (interpreted, expected, steps, arena) = interpret(graph, input, rounds)?;
    // 해석기의 자리에는 입력과 출력도 놓인다. 원소 수를 바이트로 바꿔 적는다.
    println!(
        "{label} 해석기 마디 {steps} 값의 자리(입력과 출력 포함) {}바이트 한 번에 {interpreted:.0}ns",
        arena * 4
    );
    let plain = Options {
        fuse: false,
        ..Options::default()
    };
    let mut variants = vec![("마디마다", plain), ("합쳐서", Options::default())];
    variants.extend(tuned.map(|options| ("합치고, 찾은 커널로", options)));
    for (index, (name, options)) in variants.into_iter().enumerate() {
        let (program, source) = codegen::compile(graph, &options)?;
        let (binary, seconds) = build(&format!("{label}_{index}"), &(source.clone() + HARNESS))?;
        let (compiled, got) = execute(&binary, input, rounds)?;
        // 옮긴 프로그램이 쥐는 가운데 값의 자리. 입력과 출력은 뺀다.
        let middle: usize = program
            .lengths
            .iter()
            .filter(|(value, _)| **value != program.input && **value != program.output)
            .map(|(_, len)| len * 4)
            .sum();
        println!(
            "{label} 컴파일({name}) 반복문 {} 가운데 값의 자리 {middle}바이트 소스 {}줄 컴파일 {seconds:.1}초 한 번에 {compiled:.0}ns 해석기의 {:.2}배 시간 가장 큰 차이 {:.1e}",
            program.stages.len(),
            source.lines().count(),
            compiled / interpreted,
            difference(&expected, &got)
        );
    }
    Ok(())
}

// ANCHOR: search
/// 내적 커널의 생김새(나누어 담는 합의 수, 함께 읽는 행의 수)를 바꿔 가며 선형 층
/// 하나의 시간을 잰다. 가장 빠른 것을 고르는 것이 "찾기"의 전부다.
fn kernels(rows: usize, cols: usize) -> Fallible<Options> {
    let mut graph = Graph::new("linear");
    let x = graph.value("x", &[1, cols]);
    let weights: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 37 + 404) % 29) as f32 * 0.037 - 0.5)
        .collect();
    let w = graph.constant("w", Tensor::from_vec(weights, &[rows, cols])?);
    let y = graph.value("y", &[1, rows]);
    graph.node(
        Op::FullyConnected {
            activation: Activation::None,
            keep_num_dims: false,
        },
        &[x, w],
        &[y],
    );
    graph.inputs = vec![x];
    graph.outputs = vec![y];
    let input: Vec<f32> = (0..cols)
        .map(|i| ((i * 37 + 303) % 29) as f32 * 0.037 - 0.5)
        .collect();
    // 묶음 하나가 수십 ms는 되게 한다. 짧으면 프로세서가 속도를 올리기 전에 끝난다.
    let rounds = (400_000_000 / (rows * cols)).max(20);
    let (interpreted, expected, _, _) = interpret(&graph, &input, rounds)?;
    println!("kernels {rows}x{cols} 해석기(손으로 짠 커널) {interpreted:.0}ns");
    let mut best = (f64::INFINITY, 0, 0);
    for accumulators in [1, 2, 4, 8, 16, 32] {
        let mut line = format!("kernels {rows}x{cols} 합 {accumulators:>2}개:");
        for rows_at_once in [1, 2, 4, 8] {
            let options = Options {
                fuse: true,
                accumulators,
                rows_at_once,
            };
            let (_, source) = codegen::compile(&graph, &options)?;
            let (binary, _) = build(
                &format!("kernel_{rows}x{cols}_{accumulators}_{rows_at_once}"),
                &(source + HARNESS),
            )?;
            let (nanoseconds, got) = execute(&binary, &input, rounds)?;
            assert!(difference(&expected, &got) < 1e-3, "답이 다릅니다");
            let _ = write!(line, " 행 {rows_at_once}개씩 {nanoseconds:>8.0}ns");
            if nanoseconds < best.0 {
                best = (nanoseconds, accumulators, rows_at_once);
            }
        }
        println!("{line}");
    }
    println!(
        "kernels {rows}x{cols} 가장 빠른 것: 합 {}개, 행 {}개씩, {:.0}ns (해석기의 커널의 {:.2}배 시간)",
        best.1,
        best.2,
        best.0,
        best.0 / interpreted
    );
    Ok(Options {
        fuse: true,
        accumulators: best.1,
        rows_at_once: best.2,
    })
}
// ANCHOR_END: search

fn main() -> Fallible<()> {
    // 스레드를 하나만 쓴다. 컴파일한 쪽이 스레드를 쓰지 않으므로 같은 조건으로 견준다.
    badak_core::threads::configure(1);
    let chosen: Vec<String> = std::env::args().skip(1).collect();
    let wants = |name: &str| chosen.is_empty() || chosen.iter().any(|one| one == name);
    // 커널을 먼저 찾는다. 찾은 생김새를 신경망을 옮길 때 쓴다.
    let mut tuned = None;
    if wants("kernels") {
        tuned = Some(kernels(128, 784)?);
        kernels(1536, 576)?;
    }
    if wants("mlp") {
        let images = std::fs::read(models_dir().join("mnist/t10k-images-idx3-ubyte"))?;
        let input: Vec<f32> = images[16..16 + 784]
            .iter()
            .map(|&pixel| f32::from(pixel) / 255.0)
            .collect();
        compare("mlp", &mlp()?, &input, 20_000, tuned)?;
    }
    if chosen.iter().any(|one| one == "ir") {
        // 신경망이 세 층의 표현을 지나는 모습을 적는다.
        let graph = mlp()?;
        for index in graph.order()? {
            let node = &graph.nodes[index];
            println!(
                "그래프 {} {:?} -> {:?}",
                node.op.name(),
                node.inputs,
                node.outputs
            );
        }
        let mut program = codegen::lower(&graph)?;
        for stage in &program.stages {
            println!("낮춘 뒤 {stage:?}");
        }
        codegen::fuse(&mut program);
        for stage in &program.stages {
            println!("합친 뒤 {stage:?}");
        }
        let options = tuned.unwrap_or_default();
        for line in codegen::emit(&program, &options).lines() {
            if !line.starts_with("static") {
                println!("소스 {line}");
            }
        }
    }
    if wants("glue") {
        for len in [64, 65_536] {
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 31) % 17) as f32 * 0.05 - 0.3)
                .collect();
            compare(
                &format!("glue_{len}"),
                &glue(len, 64)?,
                &input,
                (4_000_000 / len).max(20),
                None,
            )?;
        }
    }
    Ok(())
}
