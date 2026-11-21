//! 28장: 합성곱 하나하나가 곱셈을 얼마나 빨리 하는지 본다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch28_layers -- [모델] [스레드 수] [방법]`
//!
//! 방법은 `whole`(27장), `tiled`(조각마다 펼치기), `winograd`(쓸 수 있는
//! 곳에 위노그라드 변환), `auto`(기본값)다.
//!
//! 노드마다 곱셈의 수와 걸린 시간을 적고, 같은 모양의 합성곱끼리 묶는다.
//! 1초에 하는 곱셈의 수가 낮은 줄이 고칠 곳이다.

use std::collections::BTreeMap;
use std::time::Duration;

use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry, conv, models_dir};

const RUNS: usize = 40;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "resnet50".into());
    let pool = threads::global();
    let count: usize = arguments
        .next()
        .map_or(Ok(pool.pool().threads()), |text| text.parse())?;
    pool.set(count, count);

    let directory = models_dir().join("vision_tflite");
    let reference = SafeTensors::open(directory.join("reference.safetensors"))?;
    let input = reference.tensor("cup.input")?.to_vec::<f32>()?;
    let model = Model::open(directory.join(format!("{name}.tflite")))?;
    let graph = Graph::from_tflite(&model)?;
    // 셋째 인자로 합성곱의 방법을 고른다. 없으면 엔진이 고르는 대로.
    let method = arguments.next().unwrap_or_else(|| "auto".into());
    let mut registry = Registry::builtin();
    registry.register(
        "CONV_2D",
        match method.as_str() {
            "whole" => conv::conv_2d_whole,
            "tiled" => conv::conv_2d_tiled,
            "indirect" => conv::conv_2d_indirect,
            "winograd" => badak_graph::winograd::conv_2d_winograd,
            _ => conv::conv_2d,
        },
    );
    let mut interpreter = Interpreter::new(&graph, &registry)?;

    let mut times = vec![Duration::ZERO; interpreter.step_count()];
    for run in 0..RUNS + 3 {
        if run == 3 {
            times.fill(Duration::ZERO);
        }
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run_timed(&mut times)?;
    }

    // (연산, 입력의 높이 × 너비 × 채널, 창, 스트라이드, 출력 채널) → (개수, 곱셈, 초)
    let mut groups: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    let mut order = Vec::new();
    for (step, time) in interpreter.steps().zip(&times) {
        if !step.name.contains("CONV") {
            continue;
        }
        let (input, filter, output) = (step.inputs[0].1, step.inputs[1].1, step.outputs[0].1);
        let positions = (output[1] * output[2]) as f64;
        let (label, multiplies) = if step.name == "CONV_2D" {
            let stride = (input[1] as f64 / output[1] as f64).round();
            (
                format!(
                    "{:>3}×{:<3} {:>4} → {:<4} {}×{} 스트라이드 {}",
                    input[1], input[2], input[3], output[3], filter[1], filter[2], stride
                ),
                positions * (filter[1] * filter[2] * filter[3] * filter[0]) as f64,
            )
        } else {
            (
                format!(
                    "{:>3}×{:<3} {:>4} 깊이별   {}×{} → {}×{}",
                    input[1], input[2], input[3], filter[1], filter[2], output[1], output[2]
                ),
                positions * (filter[1] * filter[2] * filter[3]) as f64,
            )
        };
        if !groups.contains_key(&label) {
            order.push(label.clone());
        }
        let entry = groups.entry(label).or_default();
        entry.0 += 1;
        entry.1 += multiplies;
        entry.2 += time.as_secs_f64() / RUNS as f64;
    }
    let total: f64 = times.iter().map(Duration::as_secs_f64).sum::<f64>() / RUNS as f64;
    println!(
        "{name}.tflite, 스레드 {count}개, 사진 한 장 {:.2} ms",
        total * 1e3
    );
    println!(
        "{:<36} {:>3} {:>10} {:>9} {:>6} {:>12}",
        "입력, 채널, 창", "개", "곱셈 (백만)", "시간 (ms)", "비율", "억 번/초"
    );
    let (mut all_multiplies, mut all_seconds) = (0.0, 0.0);
    for label in &order {
        let (count, multiplies, seconds) = groups[label];
        all_multiplies += multiplies;
        all_seconds += seconds;
        println!(
            "{label:<36} {count:>3} {:>10.0} {:>9.3} {:>5.1}% {:>12.0}",
            multiplies / 1e6,
            seconds * 1e3,
            100.0 * seconds / total,
            multiplies / seconds / 1e8
        );
    }
    println!(
        "{:<36} {:>3} {:>10.0} {:>9.3} {:>5.1}% {:>12.0}",
        "합성곱 전체",
        "",
        all_multiplies / 1e6,
        all_seconds * 1e3,
        100.0 * all_seconds / total,
        all_multiplies / all_seconds / 1e8
    );
    Ok(())
}
