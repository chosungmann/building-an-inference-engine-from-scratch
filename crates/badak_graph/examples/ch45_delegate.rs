//! 45장: 그래프를 나눠 Core ML에 넘기고, CPU에서 돌린 것과 견준다.
//!
//! 실행: `sh scripts/build_coreml.sh` 뒤에
//!       `cargo run --release -p badak_graph --example ch45_delegate -- [모델]`
//!
//! macOS에서만 된다. 모델은 `mobilenetv2`(기본값)나 `resnet50`이다.

#[cfg(target_os = "macos")]
mod mac {
    use std::time::{Duration, Instant};

    use badak_formats::safetensors::SafeTensors;
    use badak_formats::tflite::Model;
    use badak_graph::coreml::{CoreMl, Units, default_library};
    use badak_graph::delegate::{Hybrid, Options};
    use badak_graph::{Graph, Interpreter, Registry, models_dir};

    fn best(scores: &[f32]) -> usize {
        (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap_or(0)
    }

    fn median(mut times: Vec<Duration>) -> f64 {
        times.sort();
        times[times.len() / 2].as_secs_f64() * 1e3
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let name = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "mobilenetv2".into());
        let directory = models_dir().join("vision_tflite");
        let labels = std::fs::read_to_string(directory.join("labels.txt"))?;
        let labels: Vec<&str> = labels.lines().collect();
        let input = SafeTensors::open(directory.join("reference.safetensors"))?
            .tensor("cup.input")?
            .to_vec::<f32>()?;
        let model = Model::open(directory.join(format!("{name}.tflite")))?;
        let graph = Graph::from_tflite(&model)?;
        let registry = Registry::builtin();

        let mut cpu = Interpreter::new(&graph, &registry)?;
        cpu.input_mut(0).copy_from_slice(&input);
        cpu.run()?;
        let expected = cpu.output(0).to_vec();
        println!("{name}.tflite, 마디 {}개", graph.nodes.len());
        println!("badak (CPU)          {} ", labels[best(&expected)]);

        let scratch = std::env::temp_dir().join("badak_ch45");
        let base = CoreMl::new(default_library(), Units::All, &scratch)?;
        for units in Units::ALL {
            let delegate = base.with_units(units);
            let started = Instant::now();
            let mut hybrid = Hybrid::new(&graph, &registry, &delegate, Options::default())?;
            let prepared = started.elapsed();
            hybrid.input_mut(0).copy_from_slice(&input);
            let started = Instant::now();
            hybrid.run()?;
            let first = started.elapsed();
            let mut times = Vec::new();
            for _ in 0..30 {
                let started = Instant::now();
                hybrid.run()?;
                times.push(started.elapsed());
            }
            let gap = hybrid
                .output(0)
                .iter()
                .zip(&expected)
                .map(|(ours, theirs)| (ours - theirs).abs())
                .fold(0.0_f32, f32::max);
            println!(
                "{:<22} 조각 {} 준비 {:>7.1}ms 첫 실행 {:>6.1}ms 가운데 {:>6.2}ms  {}  점수의 차이 {gap:.4}",
                units.label(),
                hybrid.segments().len(),
                prepared.as_secs_f64() * 1e3,
                first.as_secs_f64() * 1e3,
                median(times),
                labels[best(hybrid.output(0))],
            );
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Core ML은 macOS에만 있습니다.");
}
