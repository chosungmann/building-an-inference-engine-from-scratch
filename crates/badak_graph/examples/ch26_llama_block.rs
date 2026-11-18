//! 26장: 메모리 계획을 Llama의 블록 하나에 적용해 본다 (22.13절의 숙제).
//!
//! 실행: `cargo run --release -p badak_graph --example ch26_llama_block`
//!
//! 블록을 실행하지는 않는다. 24장까지의 `block_cached`가 만드는 텐서들을
//! 그래프로 옮겨 적고, 수명을 읽어 자리를 계획한다. 모델 파일은 필요 없다.

use badak_graph::memory::{self, Lifetime, Strategy};
use badak_graph::{Graph, Op};

/// Llama 3.2 1B의 크기들 (9장).
const HIDDEN: usize = 2048;
const KV: usize = 512;
const INTERMEDIATE: usize = 8192;

/// `block_cached`(11장, 24장)가 하는 일을 그래프로 적는다. `fused`가 거짓이면
/// 22장 때처럼 `silu`와 곱셈을 따로 한다.
fn block(tokens: usize, fused: bool) -> Graph {
    let mut graph = Graph::new("llama_block");
    let op = |name: &str| Op::Custom { name: name.into() };
    let value = |graph: &mut Graph, name: &str, width: usize| graph.value(name, &[tokens, width]);

    let x = value(&mut graph, "x", HIDDEN);
    let normed = value(&mut graph, "normed", HIDDEN);
    let q = value(&mut graph, "q", HIDDEN);
    let k = value(&mut graph, "k", KV);
    let v = value(&mut graph, "v", KV);
    let q_rotated = value(&mut graph, "q_rotated", HIDDEN);
    let k_rotated = value(&mut graph, "k_rotated", KV);
    let mixed = value(&mut graph, "mixed", HIDDEN);
    let attended = value(&mut graph, "attended", HIDDEN);
    let h = value(&mut graph, "h", HIDDEN);
    let normed2 = value(&mut graph, "normed2", HIDDEN);
    let gate = value(&mut graph, "gate", INTERMEDIATE);
    let up = value(&mut graph, "up", INTERMEDIATE);
    let hidden = value(&mut graph, "hidden", INTERMEDIATE);
    let down = value(&mut graph, "down", HIDDEN);
    let out = value(&mut graph, "out", HIDDEN);
    graph.inputs = vec![x];
    graph.outputs = vec![out];

    graph.node(op("RMS_NORM"), &[x], &[normed]);
    graph.node(op("LINEAR"), &[normed], &[q]);
    graph.node(op("LINEAR"), &[normed], &[k]);
    graph.node(op("LINEAR"), &[normed], &[v]);
    graph.node(op("ROPE"), &[q], &[q_rotated]);
    graph.node(op("ROPE"), &[k], &[k_rotated]);
    // 키와 값을 캐시에 덧붙이고, 질의로 캐시를 읽어 값을 섞는다.
    graph.node(op("ATTENTION"), &[q_rotated, k_rotated, v], &[mixed]);
    graph.node(op("LINEAR"), &[mixed], &[attended]);
    graph.node(op("ADD"), &[x, attended], &[h]);
    graph.node(op("RMS_NORM"), &[h], &[normed2]);
    graph.node(op("LINEAR"), &[normed2], &[gate]);
    graph.node(op("LINEAR"), &[normed2], &[up]);
    if fused {
        graph.node(op("SILU_MUL"), &[gate, up], &[hidden]);
    } else {
        let activated = value(&mut graph, "activated", INTERMEDIATE);
        graph.node(op("SILU"), &[gate], &[activated]);
        graph.node(op("MUL"), &[activated, up], &[hidden]);
    }
    graph.node(op("LINEAR"), &[hidden], &[down]);
    graph.node(op("ADD"), &[h, down], &[out]);
    graph
}

fn megabytes(elements: usize) -> f64 {
    (elements * 4) as f64 / 1e6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for (label, tokens, fused) in [
        ("22장의 블록, 256토큰", 256, false),
        ("24장의 블록, 256토큰", 256, true),
        ("24장의 블록, 1토큰", 1, true),
    ] {
        let graph = block(tokens, fused);
        let order = graph.order()?;
        let shapes: Vec<Vec<usize>> = graph.values.iter().map(|v| v.shape.clone()).collect();
        // 블록의 입력 x는 앞 블록이 준 것이다. 이 블록이 만드는 것만 센다.
        let made: Vec<(usize, Lifetime)> = memory::lifetimes(&graph, &order, &shapes)?
            .into_iter()
            .filter(|&(value, _)| value != graph.inputs[0])
            .collect();
        let lifetimes: Vec<Lifetime> = made.iter().map(|&(_, lifetime)| lifetime).collect();
        println!("{label}: 텐서 {}개", lifetimes.len());
        for strategy in [Strategy::Separate, Strategy::InOrder, Strategy::BySize] {
            let layout = memory::plan(&lifetimes, strategy);
            memory::check(&lifetimes, &layout)?;
            println!(
                "  {:<10} {:>8.2} MB",
                format!("{strategy:?}"),
                megabytes(layout.total)
            );
        }
        println!(
            "  {:<10} {:>8.2} MB",
            "하한",
            megabytes(memory::lower_bound(&lifetimes))
        );
        if tokens == 256 && fused {
            println!("  값마다의 수명 (단계)");
            for &(value, lifetime) in &made {
                println!(
                    "    {:<10} {:>6.2} MB  {:>2} ~ {:>2}",
                    graph.values[value].name,
                    megabytes(lifetime.size),
                    lifetime.first,
                    lifetime.last
                );
            }
        }
    }
    Ok(())
}
