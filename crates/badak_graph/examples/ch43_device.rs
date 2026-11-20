//! 43장: 사진 한 장을 분류하는 가장 작은 프로그램.
//!
//! 실행: `cargo run --release -p badak_graph --example ch43_device -- \
//!        [모델] [--int8] [--read] [--hold 초]`
//!
//! 모델은 `mobilenetv2`(기본값)나 `resnet50`이다. 프로그램이 시작한 때부터
//! 단계마다의 시각을 적는다. 크기, 시작 시간, 메모리를 재는 데 쓴다.
//!
//! `--read`를 주면 모델 파일을 메모리에 비추지 않고 통째로 읽어 들인다.
//! `--hold`를 주면 끝난 뒤에 그만큼 기다린다. 그동안 밖에서 이 프로세스의
//! 메모리를 들여다볼 수 있다.

use std::sync::Arc;
use std::time::{Duration, Instant};

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::quantized::QuantizedInterpreter;
use badak_graph::{Graph, Interpreter, Registry, models_dir};

fn best(scores: impl Iterator<Item = f32>) -> usize {
    let scores: Vec<f32> = scores.collect();
    (0..scores.len()).fold(0, |best, index| {
        if scores[index] > scores[best] {
            index
        } else {
            best
        }
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let since = || start.elapsed().as_secs_f64() * 1e3;
    let (mut name, mut int8, mut read, mut hold) = ("mobilenetv2".to_owned(), false, false, 0.0);
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--int8" => int8 = true,
            "--read" => read = true,
            "--hold" => {
                hold = arguments
                    .next()
                    .ok_or("--hold 뒤에 값이 없습니다")?
                    .parse()?
            }
            _ => name = argument,
        }
    }
    let directory = models_dir().join("vision_tflite");
    let file = if int8 {
        format!("{name}_int8.tflite")
    } else {
        format!("{name}.tflite")
    };
    let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
    let input = SafeTensors::open(directory.join("reference.safetensors"))?
        .tensor("cup.input")?
        .to_vec::<f32>()?;

    // 1. 모델 파일을 연다. 비추거나(기본), 통째로 읽거나.
    let model = if read {
        Model::parse(Arc::new(std::fs::read(directory.join(&file))?))?
    } else {
        Model::open(directory.join(&file))?
    };
    let opened = since();

    // 2. 실행할 준비를 하고, 사진을 두 번 넣는다.
    let graph = Graph::from_tflite(&model)?;
    let (ready, first, second, answer);
    if int8 {
        let mut interpreter = QuantizedInterpreter::new(&graph)?;
        let (scale, zero) = interpreter.input_quantization(0);
        ready = since();
        let run =
            |interpreter: &mut QuantizedInterpreter| -> Result<(), Box<dyn std::error::Error>> {
                for (target, &value) in interpreter.input_mut(0).iter_mut().zip(&input) {
                    *target = ((value / scale).round() as i32 + zero).clamp(-128, 127) as i8;
                }
                Ok(interpreter.run()?)
            };
        run(&mut interpreter)?;
        first = since();
        run(&mut interpreter)?;
        second = since();
        answer = best(interpreter.output(0).iter().map(|&score| f32::from(score)));
    } else {
        let mut interpreter = Interpreter::new(&graph, &Registry::builtin())?;
        ready = since();
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run()?;
        first = since();
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run()?;
        second = since();
        answer = best(interpreter.output(0).iter().copied());
    }
    println!(
        "device {file} {} 파일 {:.2}MB 열기 {opened:.2} 준비 {ready:.2} 첫추론끝 {first:.2} 둘째추론 {:.2} 답 {}",
        if read { "읽기" } else { "비추기" },
        model.file_len() as f64 / 1e6,
        second - first,
        labels.lines().nth(answer).unwrap_or("?").replace(' ', "_"),
    );
    if hold > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(hold));
    }
    Ok(())
}
