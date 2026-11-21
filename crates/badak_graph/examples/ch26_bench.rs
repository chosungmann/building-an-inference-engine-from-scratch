//! 26장: 그래프 해석기의 시간을 잰다.
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_bench`
//!
//! 재는 것은 넷이다. 파일을 열어 실행할 준비를 마치기까지, 사진 한 장의
//! 추론, 만 장을 한 번에, 그리고 해석기가 노드 하나를 부르는 데 드는 수고.
//! 한 장의 추론은 7장의 엔진(코드로 적은 모델)과 견준다.

use std::hint::black_box;
use std::time::Duration;

use badak_bench::alloc::{Counting, Snapshot};
use badak_bench::stats::Summary;
use badak_bench::timing::{Plan, measure};
use badak_core::ops;
use badak_formats::idx;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Op, Registry, Strategy, models_dir};
use badak_models::mnist::MnistMlp;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const PIXELS: usize = 784;
/// 한 장씩 잴 때 쓰는 사진의 수.
const SINGLES: usize = 1000;

fn median(plan: Plan, work: impl FnMut()) -> f64 {
    Summary::new(&measure(plan, work)).median
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = models_dir();
    let images = idx::open(models.join("mnist/t10k-images-idx3-ubyte"))?;
    let count = images.shape()[0];
    let pixels: Vec<f32> = images
        .to_vec::<u8>()?
        .iter()
        .map(|&pixel| f32::from(pixel) / 255.0)
        .collect();
    let registry = Registry::builtin();
    let quick = Plan {
        warmup: 3,
        repetitions: 31,
    };
    badak_bench::environment::print();
    println!();

    for name in ["mlp", "residual"] {
        let path = models.join(format!("mnist_tflite/{name}.tflite"));
        println!("{name}.tflite");

        // 1. 준비. 파일을 열고, 그래프로 옮기고, 계획을 세운다.
        let open = median(quick, || {
            black_box(Model::open(&path).unwrap());
        });
        let model = Model::open(&path)?;
        let convert = median(quick, || {
            black_box(Graph::from_tflite(&model).unwrap());
        });
        let graph = Graph::from_tflite(&model)?;
        let plan = median(quick, || {
            black_box(Interpreter::new(&graph, &registry).unwrap());
        });
        println!("  파일 열기와 해석          {:>9.1} µs", open * 1e6);
        println!("  그래프로 옮기기           {:>9.1} µs", convert * 1e6);
        println!("  계획 세우기               {:>9.1} µs", plan * 1e6);

        // 2. 한 장씩.
        let mut single = Interpreter::new(&graph, &registry)?;
        let mut checksum = 0.0_f32;
        let one = median(quick, || {
            for image in pixels.as_chunks::<PIXELS>().0.iter().take(SINGLES) {
                single.input_mut(0).copy_from_slice(image);
                single.run().unwrap();
                checksum += single.output(0)[0];
            }
        });
        black_box(checksum);
        println!(
            "  추론, 한 장               {:>9.2} µs",
            one / SINGLES as f64 * 1e6
        );

        // 실행 중에 메모리를 얻는지 센다.
        let before = Snapshot::now();
        for image in pixels.as_chunks::<PIXELS>().0.iter().take(SINGLES) {
            single.input_mut(0).copy_from_slice(image);
            single.run().unwrap();
        }
        let used = Snapshot::now().since(before);
        println!("  {SINGLES}장을 추론하는 동안의 할당 {:>6} 번", used.count);

        // 단계마다의 시간. 시계를 읽는 시간이 섞이므로 합은 위보다 크다.
        let mut times = vec![Duration::ZERO; single.step_count()];
        for _ in 0..20 {
            for image in pixels.as_chunks::<PIXELS>().0.iter().take(SINGLES) {
                single.input_mut(0).copy_from_slice(image);
                single.run_timed(&mut times).unwrap();
            }
        }
        let runs = (20 * SINGLES) as f64;
        let total: f64 = times.iter().map(Duration::as_secs_f64).sum();
        for ((index, step), time) in single.steps().enumerate().zip(&times) {
            let seconds = time.as_secs_f64();
            println!(
                "    {index:>2} {:<16} {:>8.3} µs {:>5.1}%",
                step.name,
                seconds / runs * 1e6,
                100.0 * seconds / total
            );
        }

        // 3. 만 장을 한 번에.
        let shape: &[usize] = &[count, PIXELS];
        let mut batch = Interpreter::with(&graph, &registry, &[shape], Strategy::BySize)?;
        let slow = Plan {
            warmup: 2,
            repetitions: 15,
        };
        let all = median(slow, || {
            batch.input_mut(0).copy_from_slice(&pixels);
            batch.run().unwrap();
            black_box(batch.output(0)[0]);
        });
        println!(
            "  추론, {count}장을 한 번에  {:>9.2} ms (장당 {:.2} µs)",
            all * 1e3,
            all / count as f64 * 1e6
        );

        if name == "mlp" {
            // 7장의 엔진. 같은 가중치이고, 전처리는 미리 해 둔다.
            let hand_written = MnistMlp::load(models.join("mnist_mlp/model.safetensors"))?;
            let load = median(quick, || {
                black_box(MnistMlp::load(models.join("mnist_mlp/model.safetensors")).unwrap());
            });
            let input = hand_written.preprocess(&images)?;
            let singles: Vec<_> = (0..SINGLES)
                .map(|index| input.slice(0, index..index + 1).unwrap())
                .collect();
            // 그래프에는 소프트맥스까지 들어 있으므로 여기서도 붙인다.
            let forward = |x| ops::softmax(&hand_written.forward(x).unwrap(), 1).unwrap();
            let one = median(quick, || {
                for x in &singles {
                    black_box(forward(x));
                }
            });
            let before = Snapshot::now();
            for x in &singles {
                black_box(forward(x));
            }
            let used = Snapshot::now().since(before);
            let all = median(slow, || {
                black_box(forward(&input));
            });
            println!("  7장의 엔진");
            println!("    모델 읽기               {:>9.1} µs", load * 1e6);
            println!(
                "    추론, 한 장             {:>9.2} µs",
                one / SINGLES as f64 * 1e6
            );
            println!(
                "    {SINGLES}장을 추론하는 동안의 할당 {:>4} 번",
                used.count
            );
            println!("    추론, {count}장을 한 번에 {:>8.2} ms", all * 1e3);
        }
        println!();
    }

    // 4. 노드 하나를 부르는 수고. 원소 여덟 개에 ReLU를 거듭 적용한다.
    //    커널이 하는 일이 거의 없으므로 남는 것이 해석기의 몫이다.
    const CHAIN: usize = 1024;
    let mut graph = Graph::new("사슬");
    let mut previous = graph.value("x", &[8]);
    graph.inputs = vec![previous];
    for index in 0..CHAIN {
        let next = graph.value(&format!("v{index}"), &[8]);
        graph.node(Op::Relu, &[previous], &[next]);
        previous = next;
    }
    graph.outputs = vec![previous];
    let mut chain = Interpreter::new(&graph, &registry)?;
    let dispatch = median(quick, || {
        for _ in 0..100 {
            chain.input_mut(0).fill(1.0);
            chain.run().unwrap();
        }
        black_box(chain.output(0)[0]);
    });
    println!(
        "ReLU {CHAIN}개의 사슬: 노드 하나에 {:.2} ns, 자리 {}원소",
        dispatch / (100 * CHAIN) as f64 * 1e9,
        chain.arena_len()
    );
    Ok(())
}
