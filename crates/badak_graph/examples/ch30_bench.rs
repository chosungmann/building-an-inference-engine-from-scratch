//! 30장: 최적화를 한 단계씩 더해 가며 사진 한 장의 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch30_bench <onnx|tflite>
//! <모델> <단계>`
//!
//! 단계는 0부터다. 번호가 클수록 패스가 더 들어간다. 한 번 실행에 한 단계만
//! 잰다. 단계들을 번갈아 재려는 것이다(22.12절). `scripts/bench_ch30.sh`가
//! 그렇게 부른다. 스레드의 수는 환경 변수 `BADAK_THREADS`로 정한다.

use std::hint::black_box;

use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::threads;
use badak_formats::safetensors::SafeTensors;
use badak_formats::{onnx, tflite};
use badak_graph::optimize::{self, PASSES, Pass};
use badak_graph::{Graph, Interpreter, Registry, Strategy, models_dir};

/// 단계마다 더해지는 패스들의 이름과, 그 단계의 이름.
const ONNX_STAGES: [(&str, &[&str]); 5] = [
    ("옮긴 그대로", &[]),
    ("+ 상수 접기", &["상수 접기", "IDENTITY 지우기"]),
    (
        "+ TRANSPOSE 없애기",
        &[
            "TRANSPOSE 뒤로 옮기기",
            "TRANSPOSE 상쇄",
            "형상 바꾸기 줄이기",
        ],
    ),
    ("+ 배치 정규화 접기", &["배치 정규화 접기"]),
    ("+ 활성화 함수 붙이기", &["활성화 함수 붙이기"]),
];
const TFLITE_STAGES: [(&str, &[&str]); 2] = [
    ("변환기가 준 그대로", &[]),
    ("+ PAD 합치기", &["PAD 합치기"]),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [source, name, stage] = &arguments[..] else {
        return Err("사용법: ch30_bench <onnx|tflite> <모델> <단계>".into());
    };
    let stage: usize = stage.parse()?;
    let registry = Registry::builtin();
    let (mut graph, stages): (Graph, &[(&str, &[&str])]) = match source.as_str() {
        "onnx" => {
            let path = models_dir().join(format!("vision_onnx/{name}.onnx"));
            let mut graph = Graph::from_onnx(&onnx::Model::open(path)?)?;
            graph.infer_shapes(&registry)?;
            (graph, &ONNX_STAGES)
        }
        "tflite" => {
            let path = models_dir().join(format!("vision_tflite/{name}.tflite"));
            (
                Graph::from_tflite(&tflite::Model::open(path)?)?,
                &TFLITE_STAGES,
            )
        }
        other => return Err(format!("모르는 형식 {other}").into()),
    };
    // 마지막 단계 다음의 번호는 "모든 패스 + 제자리 계산"이다.
    let in_place = stage == stages.len();
    let included = &stages[..stages.len().min(stage + 1)];
    let wanted: Vec<&str> = included
        .iter()
        .flat_map(|(_, names)| names.iter().copied())
        .collect();
    let passes: Vec<(&'static str, Pass)> = PASSES
        .iter()
        .copied()
        .filter(|(name, _)| wanted.contains(name))
        .collect();
    optimize::run(&mut graph, &registry, &passes)?;
    let label = if in_place {
        "+ 제자리 계산"
    } else {
        included.last().expect("단계가 있다").0
    };

    let mut interpreter = if in_place {
        Interpreter::in_place(&graph, &registry, &[], Strategy::BySize)?
    } else {
        Interpreter::new(&graph, &registry)?
    };
    let inputs = SafeTensors::open(models_dir().join("vision_tflite/reference.safetensors"))?;
    let input = inputs.tensor("cup.input")?.to_vec::<f32>()?;
    let plan = Plan {
        warmup: 5,
        repetitions: 51,
    };
    let samples = measure(plan, || {
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().expect("실행");
        black_box(interpreter.output(0)[0]);
    });
    let summary = Summary::new(&samples);
    println!(
        "{source} {name} 스레드 {:>2} 단계 {stage} {label:<22} 노드 {:>3} 자리 {:>5.2}MB 중앙값 {:>7.2}ms 최소 {:>7.2}ms",
        threads::global().pool().threads(),
        interpreter.step_count(),
        interpreter.arena_len() as f64 * 4.0 / 1e6,
        summary.median * 1e3,
        summary.min * 1e3
    );
    Ok(())
}
