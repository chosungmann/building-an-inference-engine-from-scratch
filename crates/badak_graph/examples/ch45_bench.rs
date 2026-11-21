//! 45장: 그래프를 통째로 넘길 때와 나눠서 넘길 때를 잰다.
//!
//! 실행: `sh scripts/build_coreml.sh` 뒤에
//!       `cargo run --release -p badak_graph --example ch45_bench -- <부분> <모델>`
//!
//! 부분은 `whole`(통째로), `plan`(Core ML이 층마다 고른 장치), `split`(나눠서)
//! 가운데 하나다. 결과는 한 줄에 하나씩 적는다. `scripts/bench_ch45.sh`가
//! 모아서 `bench/ch45`에 둔다. macOS에서만 된다.

#[cfg(target_os = "macos")]
mod mac {
    use std::time::{Duration, Instant};

    use badak_formats::safetensors::SafeTensors;
    use badak_formats::tflite::Model;
    use badak_graph::coreml::{self, CoreMl, Units, default_library};
    use badak_graph::delegate::{Delegate, Hybrid, Options, partition};
    use badak_graph::{Graph, Interpreter, Registry, models_dir};

    /// 재기 전에 버리는 실행의 수와 재는 실행의 수.
    const WARM: usize = 5;
    const RUNS: usize = 50;

    fn best(scores: &[f32]) -> usize {
        (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap_or(0)
    }

    fn milliseconds(time: Duration) -> f64 {
        time.as_secs_f64() * 1e3
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    fn gap(ours: &[f32], theirs: &[f32]) -> f32 {
        ours.iter()
            .zip(theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn tag(units: Units) -> &'static str {
        match units {
            Units::CpuOnly => "coreml-cpu",
            Units::CpuAndGpu => "coreml-gpu",
            Units::CpuAndNeuralEngine => "coreml-npu",
            Units::All => "coreml-all",
        }
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut arguments = std::env::args().skip(1);
        let part = arguments
            .next()
            .ok_or("부분(whole, plan, split)을 주십시오")?;
        let name = arguments.next().unwrap_or_else(|| "mobilenetv2".into());
        let directory = models_dir().join("vision_tflite");
        let input = SafeTensors::open(directory.join("reference.safetensors"))?
            .tensor("cup.input")?
            .to_vec::<f32>()?;
        let model = Model::open(directory.join(format!("{name}.tflite")))?;
        let graph = Graph::from_tflite(&model)?;
        let registry = Registry::builtin();
        let scratch = std::env::temp_dir().join("badak_ch45");
        let base = CoreMl::new(default_library(), Units::All, &scratch)?;

        // 견줄 답: 이 엔진이 CPU에서 낸 점수.
        let started = Instant::now();
        let mut cpu = Interpreter::new(&graph, &registry)?;
        let prepared = started.elapsed();
        cpu.input_mut(0).copy_from_slice(&input);
        let started = Instant::now();
        cpu.run()?;
        let first = started.elapsed();
        let expected = cpu.output(0).to_vec();

        match part.as_str() {
            "whole" => {
                let mut times = Vec::new();
                for index in 0..WARM + RUNS {
                    let started = Instant::now();
                    cpu.run()?;
                    if index >= WARM {
                        times.push(milliseconds(started.elapsed()));
                    }
                }
                let fastest = times.iter().copied().fold(f64::INFINITY, f64::min);
                println!(
                    "whole {name} badak-cpu 준비 {:.2} 첫실행 {:.2} 가운데 {:.3} 가장빠름 {:.3} 차이 0 답 {}",
                    milliseconds(prepared),
                    milliseconds(first),
                    median(times),
                    fastest,
                    best(&expected)
                );
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
                    for index in 0..WARM + RUNS {
                        let started = Instant::now();
                        hybrid.run()?;
                        if index >= WARM {
                            times.push(milliseconds(started.elapsed()));
                        }
                    }
                    let fastest = times.iter().copied().fold(f64::INFINITY, f64::min);
                    println!(
                        "whole {name} {} 준비 {:.1} 첫실행 {:.2} 가운데 {:.3} 가장빠름 {:.3} 차이 {:.4} 답 {}",
                        tag(units),
                        milliseconds(prepared),
                        milliseconds(first),
                        median(times),
                        fastest,
                        gap(hybrid.output(0), &expected),
                        best(hybrid.output(0))
                    );
                }
            }
            "plan" => {
                for units in Units::ALL {
                    let delegate = base.with_units(units);
                    let supported: Vec<bool> = graph
                        .nodes
                        .iter()
                        .map(|node| delegate.supports(&graph, node))
                        .collect();
                    let segments = partition(&graph, &supported, Options::default())?;
                    let plan = delegate.plan(&graph, &segments[0])?;
                    let count = |device: &str| {
                        plan.iter()
                            .filter(|(_, chosen)| chosen.starts_with(device))
                            .count()
                    };
                    println!(
                        "plan {name} {} 층 {} CPU {} GPU {} NPU {} 모름 {}",
                        tag(units),
                        plan.len(),
                        count("CPU"),
                        count("GPU"),
                        count("NPU"),
                        count("-")
                    );
                    // 고른 장치가 그 설정에서 가장 좋은 것이 아닌 층들을 적는다.
                    let wanted = match units {
                        Units::CpuOnly => "CPU",
                        Units::CpuAndGpu => "GPU",
                        _ => "NPU",
                    };
                    for (layer, device) in &plan {
                        if !device.starts_with(wanted) {
                            println!(
                                "plan-layer {name} {} {layer} {}",
                                tag(units),
                                device.replace('\t', " ")
                            );
                        }
                    }
                }
            }
            "split" => {
                // (이름, 넘기지 않을 연산들, 나누는 방법).
                let all = Options::default();
                let cases: [(&str, &[&str], Options); 6] = [
                    ("전부", &[], all),
                    ("ADD빼고", &["ADD"], all),
                    ("DW빼고", &["DEPTHWISE_CONV_2D"], all),
                    ("ADD와DW빼고", &["ADD", "DEPTHWISE_CONV_2D"], all),
                    (
                        "ADD빼고_큰것하나만",
                        &["ADD"],
                        Options {
                            max_delegated: 1,
                            ..all
                        },
                    ),
                    (
                        "ADD빼고_노드넷이상만",
                        &["ADD"],
                        Options {
                            min_nodes: 4,
                            ..all
                        },
                    ),
                ];
                for units in [Units::CpuOnly, Units::CpuAndGpu, Units::CpuAndNeuralEngine] {
                    for (case, refused, options) in cases {
                        let delegate = base.with_units(units).refusing(refused);
                        let started = Instant::now();
                        let mut hybrid = Hybrid::new(&graph, &registry, &delegate, options)?;
                        let prepared = started.elapsed();
                        let segments = hybrid.segments().to_vec();
                        hybrid.input_mut(0).copy_from_slice(&input);
                        let mut stage = vec![Duration::ZERO; segments.len()];
                        let (mut totals, mut handed, mut kept) =
                            (Vec::new(), Vec::new(), Vec::new());
                        for index in 0..WARM + RUNS {
                            let started = Instant::now();
                            hybrid.run_timed(&mut stage)?;
                            let total = started.elapsed();
                            if index >= WARM {
                                totals.push(milliseconds(total));
                                let sum = |delegated: bool| -> f64 {
                                    segments
                                        .iter()
                                        .zip(&stage)
                                        .filter(|(segment, _)| segment.delegated == delegated)
                                        .fold(0.0, |sum, (_, time)| sum + milliseconds(*time))
                                };
                                handed.push(sum(true));
                                kept.push(sum(false));
                            }
                        }
                        let delegated: Vec<_> = segments
                            .iter()
                            .filter(|segment| segment.delegated)
                            .collect();
                        // 넘긴 조각들을 드나드는 값의 수.
                        let crossings: usize = delegated
                            .iter()
                            .map(|segment| segment.inputs.len() + segment.outputs.len())
                            .sum();
                        // 경계에서 값의 레이아웃을 바꾸는 일만 따로 잰다. 넘긴 조각을
                        // 드나드는 값마다 한 번씩이다.
                        let ports: Vec<(coreml::Port, bool)> = delegated
                            .iter()
                            .map(|segment| coreml::lower(&graph, segment))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .flat_map(|lowered| {
                                let inputs = lowered.inputs.into_iter().map(|port| (port, true));
                                inputs.chain(lowered.outputs.into_iter().map(|port| (port, false)))
                            })
                            .filter(|(port, _)| port.channels_last)
                            .collect();
                        let length = |port: &coreml::Port| port.shape.iter().product::<usize>();
                        let longest = ports
                            .iter()
                            .map(|(port, _)| length(port))
                            .max()
                            .unwrap_or(0);
                        let (from, mut into) = (vec![1.0_f32; longest], vec![0.0_f32; longest]);
                        let mut layout = Vec::new();
                        for _ in 0..RUNS {
                            let started = Instant::now();
                            for (port, inward) in &ports {
                                let len = length(port);
                                if *inward {
                                    coreml::to_channels_first(port, &from[..len], &mut into[..len]);
                                } else {
                                    coreml::to_channels_last(port, &from[..len], &mut into[..len]);
                                }
                                std::hint::black_box(&mut into);
                            }
                            layout.push(milliseconds(started.elapsed()));
                        }
                        let bytes: usize = ports.iter().map(|(port, _)| length(port) * 4).sum();
                        println!(
                            "split {name} {} {case} 조각 {} 넘긴조각 {} 넘긴노드 {} 드나드는값 {} 준비 {:.1} 가운데 {:.3} 넘긴쪽 {:.3} CPU쪽 {:.3} 레이아웃바꾸기 {:.3} 옮긴바이트 {} 차이 {:.4} 답 {}",
                            tag(units),
                            segments.len(),
                            delegated.len(),
                            delegated
                                .iter()
                                .map(|segment| segment.nodes.len())
                                .sum::<usize>(),
                            crossings,
                            milliseconds(prepared),
                            median(totals),
                            median(handed),
                            median(kept),
                            median(layout),
                            bytes,
                            gap(hybrid.output(0), &expected),
                            best(hybrid.output(0))
                        );
                    }
                }
            }
            other => return Err(format!("모르는 부분 {other}").into()),
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
