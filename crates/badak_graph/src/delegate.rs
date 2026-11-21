//! 그래프의 일부를 다른 실행기에 넘긴다 (45장).
//!
//! 폰의 GPU나 NPU는 모든 연산을 돌리지 못한다. 그래서 엔진은 그래프를
//! **나눈다**. 가속기가 돌릴 수 있는 노드들을 묶어 통째로 넘기고, 나머지는
//! CPU에서 돌린다. 넘겨받는 쪽을 델리게이트(delegate)라고 부른다.
//!
//! 이 모듈에는 셋이 있다. 나누는 함수([`partition`]), 넘겨받는 쪽이 지킬
//! 약속([`Delegate`], [`Compiled`]), 나눈 조각들을 차례로 돌리는
//! 실행기([`Hybrid`]).

use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::graph::{Graph, Node};
use crate::interpreter::Interpreter;
use crate::registry::Registry;

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

// ANCHOR: segment
/// 한 실행기가 이어서 돌리는 노드들.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// 델리게이트에 넘기는 조각인지. 아니면 CPU에서 돈다.
    pub delegated: bool,
    /// 노드들의 번호. 실행할 순서대로다.
    pub nodes: Vec<usize>,
    /// 조각 밖에서 만들어져 들어오는 값들. 가중치는 들어 있지 않다.
    pub inputs: Vec<usize>,
    /// 조각 안에서 만들어져 밖에서 읽히는 값들.
    pub outputs: Vec<usize>,
}

/// 나누는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// 노드가 이보다 적은 조각은 넘기지 않는다.
    pub min_nodes: usize,
    /// 넘기는 조각의 수의 한도. 넘치면 큰 것부터 남긴다.
    pub max_delegated: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            min_nodes: 1,
            max_delegated: usize::MAX,
        }
    }
}
// ANCHOR_END: segment

/// 값마다 그 값을 만드는 노드.
fn producers(graph: &Graph) -> Vec<Option<usize>> {
    let mut producer = vec![None; graph.values.len()];
    for (index, node) in graph.nodes.iter().enumerate() {
        for &output in &node.outputs {
            producer[output] = Some(index);
        }
    }
    producer
}

// ANCHOR: color
/// 노드들을 같은 종류끼리 묶는다. 노드마다 조각의 번호를 돌려준다.
///
/// 한 번에 한 조각씩 키운다. 조각의 종류(넘길 것인가)는 지금 실행할 수 있는
/// 노드 가운데 번호가 가장 작은 것이 정한다. 그 뒤로는 실행할 수 있게 된
/// 같은 종류의 노드를 더 넣을 수 없을 때까지 넣는다. 다른 종류의 노드는
/// 다음 조각을 기다린다. LiteRT가 쓰는 방법이다.
fn color(graph: &Graph, supported: &[bool]) -> Result<Vec<usize>> {
    let producer = producers(graph);
    const WAITING: usize = usize::MAX;
    let mut segment_of = vec![WAITING; graph.nodes.len()];
    let ready = |index: usize, segment_of: &[usize]| {
        graph.nodes[index]
            .inputs
            .iter()
            .all(|&input| producer[input].is_none_or(|before| segment_of[before] != WAITING))
    };
    let mut left = graph.nodes.len();
    let mut current = 0;
    while left > 0 {
        let first = (0..graph.nodes.len())
            .find(|&index| segment_of[index] == WAITING && ready(index, &segment_of))
            .ok_or_else(|| invalid("그래프에 돌고 도는 길이 있습니다"))?;
        let kind = supported[first];
        loop {
            let mut grew = false;
            for index in 0..graph.nodes.len() {
                if segment_of[index] == WAITING
                    && supported[index] == kind
                    && ready(index, &segment_of)
                {
                    segment_of[index] = current;
                    left -= 1;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        current += 1;
    }
    Ok(segment_of)
}
// ANCHOR_END: color

/// 조각의 번호들로 조각들을 만든다. 조각의 번호가 실행할 순서다.
fn gather(graph: &Graph, segment_of: &[usize], delegated: &[bool]) -> Result<Vec<Segment>> {
    let producer = producers(graph);
    let order = graph.order()?;
    let count = segment_of.iter().max().map_or(0, |last| last + 1);
    let mut segments: Vec<Segment> = (0..count)
        .map(|index| Segment {
            delegated: delegated[index],
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        })
        .collect();
    for &index in &order {
        segments[segment_of[index]].nodes.push(index);
    }
    let push = |list: &mut Vec<usize>, value: usize| {
        if !list.contains(&value) {
            list.push(value);
        }
    };
    for &index in &order {
        let home = segment_of[index];
        for &input in &graph.nodes[index].inputs {
            if graph.values[input].constant.is_some() {
                continue;
            }
            match producer[input] {
                Some(before) if segment_of[before] == home => {}
                Some(before) => {
                    push(&mut segments[home].inputs, input);
                    push(&mut segments[segment_of[before]].outputs, input);
                }
                None => push(&mut segments[home].inputs, input),
            }
        }
    }
    for &output in &graph.outputs {
        if let Some(before) = producer[output] {
            push(&mut segments[segment_of[before]].outputs, output);
        }
    }
    Ok(segments)
}

// ANCHOR: partition
/// 그래프를 조각들로 나눈다. 조각들은 실행할 순서대로 놓인다.
///
/// `supported[i]`는 `i`번 노드를 델리게이트가 돌릴 수 있는지다.
pub fn partition(graph: &Graph, supported: &[bool], options: Options) -> Result<Vec<Segment>> {
    if supported.len() != graph.nodes.len() {
        return Err(invalid("supported의 길이가 노드의 수와 다릅니다"));
    }
    // 1. 같은 종류끼리 묶는다.
    let segment_of = color(graph, supported)?;
    let count = segment_of.iter().max().map_or(0, |last| last + 1);
    let mut sizes = vec![0_usize; count];
    let mut kind = vec![false; count];
    for (index, &segment) in segment_of.iter().enumerate() {
        sizes[segment] += 1;
        kind[segment] = supported[index];
    }
    // 2. 너무 작은 조각은 넘기지 않는다. 넘기는 조각이 너무 많으면 큰 것부터
    //    남긴다. 조각을 넘나들 때마다 값이 든다.
    let mut candidates: Vec<usize> = (0..count)
        .filter(|&segment| kind[segment] && sizes[segment] >= options.min_nodes)
        .collect();
    candidates.sort_by_key(|&segment| (std::cmp::Reverse(sizes[segment]), segment));
    candidates.truncate(options.max_delegated);
    let keep: Vec<bool> = (0..count)
        .map(|segment| candidates.contains(&segment))
        .collect();
    // 3. 넘기지 않기로 한 노드들을 CPU 쪽으로 돌리고 다시 묶는다. 이웃한 CPU
    //    조각들이 하나로 합쳐진다.
    let narrowed: Vec<bool> = segment_of.iter().map(|&segment| keep[segment]).collect();
    let segment_of = color(graph, &narrowed)?;
    let count = segment_of.iter().max().map_or(0, |last| last + 1);
    let mut delegated = vec![false; count];
    for (index, &segment) in segment_of.iter().enumerate() {
        delegated[segment] = narrowed[index];
    }
    gather(graph, &segment_of, &delegated)
}
// ANCHOR_END: partition

// ANCHOR: subgraph
/// 조각 하나를 혼자 설 수 있는 그래프로 떼어 낸다.
///
/// 조각의 입력이 새 그래프의 입력이, 조각의 출력이 새 그래프의 출력이 된다.
/// 가중치는 함께 옮긴다(텐서는 파일을 비춘 메모리를 빌린 것이어서 복사되지
/// 않는다).
pub fn subgraph(graph: &Graph, segment: &Segment) -> Graph {
    let mut part = Graph::new(&graph.name);
    let mut moved = vec![None; graph.values.len()];
    let mut carry = |part: &mut Graph, value: usize| -> usize {
        *moved[value].get_or_insert_with(|| {
            part.values.push(graph.values[value].clone());
            part.values.len() - 1
        })
    };
    for &input in &segment.inputs {
        let index = carry(&mut part, input);
        part.inputs.push(index);
    }
    for &index in &segment.nodes {
        let node = &graph.nodes[index];
        let inputs: Vec<usize> = node
            .inputs
            .iter()
            .map(|&value| carry(&mut part, value))
            .collect();
        let outputs: Vec<usize> = node
            .outputs
            .iter()
            .map(|&value| carry(&mut part, value))
            .collect();
        part.nodes.push(Node {
            op: node.op.clone(),
            inputs,
            outputs,
        });
    }
    for &output in &segment.outputs {
        let index = carry(&mut part, output);
        part.outputs.push(index);
    }
    part
}
// ANCHOR_END: subgraph

// ANCHOR: delegate
/// 그래프의 일부를 넘겨받아 돌리는 쪽.
pub trait Delegate {
    /// 이름. 사람이 읽는 출력에 쓴다.
    fn name(&self) -> String;

    /// 이 노드를 돌릴 수 있는지.
    fn supports(&self, graph: &Graph, node: &Node) -> bool;

    /// 조각 하나를 돌릴 준비를 한다. 시간이 걸릴 수 있다. 가속기가 쓸 꼴로
    /// 다시 컴파일하는 일이 여기에서 일어난다.
    fn compile(&self, graph: &Graph, segment: &Segment) -> Result<Box<dyn Compiled>>;
}

/// 돌릴 준비가 된 조각.
pub trait Compiled {
    /// 조각의 입력들을 받아 출력들을 채운다. 순서는 [`Segment`]의 `inputs`,
    /// `outputs`와 같고, 값의 레이아웃도 그래프의 것 그대로다.
    fn run(&mut self, inputs: &[&[f32]], outputs: &mut [Vec<f32>]) -> Result<()>;
}
// ANCHOR_END: delegate

/// 조각을 돌리는 것.
enum Runner {
    Cpu(Box<Interpreter>),
    Delegated(Box<dyn Compiled>),
}

struct Stage {
    runner: Runner,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

/// 나눈 그래프를 조각마다 맞는 실행기로 돌린다.
pub struct Hybrid {
    stages: Vec<Stage>,
    /// 조각 사이를 오가는 값들. 값의 번호로 찾는다.
    values: Vec<Vec<f32>>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    /// 조각마다 준비하는 데 걸린 시간.
    prepared: Vec<Duration>,
    segments: Vec<Segment>,
}

impl std::fmt::Debug for Hybrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hybrid")
            .field("segments", &self.segments)
            .finish_non_exhaustive()
    }
}

impl Hybrid {
    // ANCHOR: hybrid_new
    /// 그래프를 나누고 조각마다 실행기를 준비한다.
    pub fn new(
        graph: &Graph,
        registry: &Registry,
        delegate: &dyn Delegate,
        options: Options,
    ) -> Result<Self> {
        let supported: Vec<bool> = graph
            .nodes
            .iter()
            .map(|node| delegate.supports(graph, node))
            .collect();
        let segments = partition(graph, &supported, options)?;
        Self::from_segments(graph, registry, delegate, segments)
    }

    /// 이미 나눈 조각들로 실행기를 준비한다.
    pub fn from_segments(
        graph: &Graph,
        registry: &Registry,
        delegate: &dyn Delegate,
        segments: Vec<Segment>,
    ) -> Result<Self> {
        let mut values = vec![Vec::new(); graph.values.len()];
        let mut stages = Vec::with_capacity(segments.len());
        let mut prepared = Vec::with_capacity(segments.len());
        for segment in &segments {
            let started = Instant::now();
            let runner = if segment.delegated {
                Runner::Delegated(delegate.compile(graph, segment)?)
            } else {
                Runner::Cpu(Box::new(Interpreter::new(
                    &subgraph(graph, segment),
                    registry,
                )?))
            };
            prepared.push(started.elapsed());
            for &value in segment.inputs.iter().chain(&segment.outputs) {
                values[value] = vec![0.0; graph.values[value].shape.iter().product()];
            }
            stages.push(Stage {
                runner,
                inputs: segment.inputs.clone(),
                outputs: segment.outputs.clone(),
            });
        }
        for &value in graph.inputs.iter().chain(&graph.outputs) {
            if values[value].is_empty() {
                values[value] = vec![0.0; graph.values[value].shape.iter().product()];
            }
        }
        Ok(Self {
            stages,
            values,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            prepared,
            segments,
        })
    }
    // ANCHOR_END: hybrid_new

    /// 나눈 조각들.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// 조각마다 준비하는 데 걸린 시간.
    pub fn prepared(&self) -> &[Duration] {
        &self.prepared
    }

    /// `index`번째 입력을 쓸 자리.
    pub fn input_mut(&mut self, index: usize) -> &mut [f32] {
        &mut self.values[self.inputs[index]]
    }

    /// `index`번째 출력.
    pub fn output(&self, index: usize) -> &[f32] {
        &self.values[self.outputs[index]]
    }

    /// 한 번 실행한다.
    pub fn run(&mut self) -> Result<()> {
        self.run_timed(&mut [])
    }

    // ANCHOR: hybrid_run
    /// 한 번 실행한다. `times`가 조각의 수만큼 길면 조각마다 걸린 시간을
    /// 채운다. 조각의 경계에서 값을 옮기는 시간이 그 조각의 시간에 든다.
    pub fn run_timed(&mut self, times: &mut [Duration]) -> Result<()> {
        for (index, stage) in self.stages.iter_mut().enumerate() {
            let started = Instant::now();
            // 출력들을 잠깐 꺼내 둔다. 입력들을 빌린 채로 출력에 쓰려는 것이다.
            let mut outputs: Vec<Vec<f32>> = stage
                .outputs
                .iter()
                .map(|&value| std::mem::take(&mut self.values[value]))
                .collect();
            let outcome = match &mut stage.runner {
                Runner::Cpu(interpreter) => {
                    for (position, &value) in stage.inputs.iter().enumerate() {
                        interpreter
                            .input_mut(position)
                            .copy_from_slice(&self.values[value]);
                    }
                    interpreter.run().map(|()| {
                        for (position, output) in outputs.iter_mut().enumerate() {
                            output.copy_from_slice(interpreter.output(position));
                        }
                    })
                }
                Runner::Delegated(compiled) => {
                    let inputs: Vec<&[f32]> = stage
                        .inputs
                        .iter()
                        .map(|&value| self.values[value].as_slice())
                        .collect();
                    compiled.run(&inputs, &mut outputs)
                }
            };
            for (&value, output) in stage.outputs.iter().zip(outputs) {
                self.values[value] = output;
            }
            outcome?;
            if let Some(time) = times.get_mut(index) {
                *time = started.elapsed();
            }
        }
        Ok(())
    }
    // ANCHOR_END: hybrid_run
}

// ANCHOR: cpu_delegate
/// 넘겨받은 조각을 이 엔진의 해석기로 돌리는 델리게이트.
///
/// 가속기가 없어도 나누기와 넘기기를 시험할 수 있게 한다. 어떤 연산을
/// 돌릴 수 있다고 답할지를 함수로 받으므로, "이 연산을 못 돌리는 가속기"를
/// 흉내 낼 수 있다. 결과는 나누지 않고 돌린 것과 비트까지 같아야 한다.
pub struct Reference<F> {
    supports: F,
}

impl<F: Fn(&Graph, &Node) -> bool> Reference<F> {
    /// `supports`가 참이라고 답한 노드를 넘겨받는 델리게이트.
    pub fn new(supports: F) -> Self {
        Self { supports }
    }
}

struct ReferenceCompiled(Interpreter);

impl Compiled for ReferenceCompiled {
    fn run(&mut self, inputs: &[&[f32]], outputs: &mut [Vec<f32>]) -> Result<()> {
        for (position, input) in inputs.iter().enumerate() {
            self.0.input_mut(position).copy_from_slice(input);
        }
        self.0.run()?;
        for (position, output) in outputs.iter_mut().enumerate() {
            output.copy_from_slice(self.0.output(position));
        }
        Ok(())
    }
}

impl<F: Fn(&Graph, &Node) -> bool> Delegate for Reference<F> {
    fn name(&self) -> String {
        "reference".to_owned()
    }

    fn supports(&self, graph: &Graph, node: &Node) -> bool {
        (self.supports)(graph, node)
    }

    fn compile(&self, graph: &Graph, segment: &Segment) -> Result<Box<dyn Compiled>> {
        let interpreter = Interpreter::new(&subgraph(graph, segment), &Registry::builtin())?;
        Ok(Box::new(ReferenceCompiled(interpreter)))
    }
}
// ANCHOR_END: cpu_delegate

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Activation, Op};

    /// 값 `count`개가 한 줄로 이어진 그래프. `skips`의 `(a, b)`마다 `a`번
    /// 노드의 출력을 `b`번 노드가 함께 읽는다(잔차 연결).
    fn chain(count: usize, skips: &[(usize, usize)]) -> Graph {
        let mut graph = Graph::new("chain");
        let mut values = vec![graph.value("x", &[4])];
        graph.inputs.push(values[0]);
        for index in 0..count {
            let output = graph.value(&format!("v{index}"), &[4]);
            let extra = skips
                .iter()
                .find(|&&(_, to)| to == index)
                .map(|&(from, _)| values[from + 1]);
            match extra {
                Some(extra) => graph.node(
                    Op::Add {
                        activation: Activation::None,
                    },
                    &[values[index], extra],
                    &[output],
                ),
                None => graph.node(Op::Relu, &[values[index]], &[output]),
            }
            values.push(output);
        }
        graph.outputs.push(values[count]);
        graph
    }

    fn shape(segments: &[Segment]) -> Vec<(bool, Vec<usize>)> {
        segments
            .iter()
            .map(|segment| (segment.delegated, segment.nodes.clone()))
            .collect()
    }

    #[test]
    fn splits_a_chain_where_support_changes() {
        let graph = chain(6, &[]);
        let all = partition(&graph, &[true; 6], Options::default()).unwrap();
        assert_eq!(shape(&all), [(true, vec![0, 1, 2, 3, 4, 5])]);
        assert_eq!(
            (all[0].inputs.clone(), all[0].outputs.clone()),
            (graph.inputs.clone(), graph.outputs.clone())
        );

        let none = partition(&graph, &[false; 6], Options::default()).unwrap();
        assert_eq!(shape(&none), [(false, vec![0, 1, 2, 3, 4, 5])]);

        let holes = [true, true, false, true, true, true];
        let split = partition(&graph, &holes, Options::default()).unwrap();
        assert_eq!(
            shape(&split),
            [(true, vec![0, 1]), (false, vec![2]), (true, vec![3, 4, 5])]
        );
        // 경계를 넘는 값은 하나씩이다.
        assert_eq!(split[0].outputs, split[1].inputs);
        assert_eq!(split[1].outputs, split[2].inputs);
    }

    #[test]
    fn small_and_surplus_partitions_stay_on_the_cpu() {
        let graph = chain(8, &[]);
        let holes = [true, false, true, true, false, true, true, true];
        let all = partition(&graph, &holes, Options::default()).unwrap();
        assert_eq!(all.iter().filter(|segment| segment.delegated).count(), 3);
        // 노드가 둘보다 적은 조각은 넘기지 않는다. 이웃한 CPU 조각과 합쳐진다.
        let bigger = partition(
            &graph,
            &holes,
            Options {
                min_nodes: 2,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(
            shape(&bigger),
            [
                (false, vec![0, 1]),
                (true, vec![2, 3]),
                (false, vec![4]),
                (true, vec![5, 6, 7])
            ]
        );
        // 하나만 넘긴다면 가장 큰 것을.
        let one = partition(
            &graph,
            &holes,
            Options {
                min_nodes: 1,
                max_delegated: 1,
            },
        )
        .unwrap();
        assert_eq!(
            shape(&one),
            [(false, vec![0, 1, 2, 3, 4]), (true, vec![5, 6, 7])]
        );
    }

    #[test]
    fn a_skip_connection_crosses_the_boundary_as_an_extra_value() {
        // 0 → 1 → 2 → 3(= 2의 출력 + 0의 출력). 1을 못 돌린다.
        let graph = chain(4, &[(0, 3)]);
        let split = partition(&graph, &[true, false, true, true], Options::default()).unwrap();
        assert_eq!(
            shape(&split),
            [(true, vec![0]), (false, vec![1]), (true, vec![2, 3])]
        );
        // 마지막 조각은 값 둘을 받는다. 바로 앞 조각의 것과, 첫 조각의 것.
        assert_eq!(split[2].inputs.len(), 2);
        assert_eq!(split[0].outputs.len(), 1);
        // 모든 노드가 꼭 한 번씩 들어 있고, 조각의 입력은 앞에서 만들어진다.
        let mut seen = vec![false; graph.values.len()];
        seen[graph.inputs[0]] = true;
        for segment in &split {
            assert!(segment.inputs.iter().all(|&value| seen[value]));
            for &node in &segment.nodes {
                seen[graph.nodes[node].outputs[0]] = true;
            }
        }
    }

    #[test]
    fn the_hybrid_executor_matches_the_interpreter_bit_for_bit() {
        let graph = chain(7, &[(1, 4), (2, 6)]);
        let registry = Registry::builtin();
        let input = [-1.5_f32, 0.25, 3.0, -0.0];
        let mut plain = Interpreter::new(&graph, &registry).unwrap();
        plain.input_mut(0).copy_from_slice(&input);
        plain.run().unwrap();
        for holes in [0b0000000_u32, 0b1111111, 0b0010100, 0b1010101, 0b0001000] {
            let delegate = Reference::new(|graph: &Graph, node: &Node| {
                let index = graph
                    .nodes
                    .iter()
                    .position(|other| std::ptr::eq(other, node))
                    .unwrap();
                holes >> index & 1 == 0
            });
            let mut hybrid = Hybrid::new(&graph, &registry, &delegate, Options::default()).unwrap();
            hybrid.input_mut(0).copy_from_slice(&input);
            let mut times = vec![Duration::ZERO; hybrid.segments().len()];
            hybrid.run_timed(&mut times).unwrap();
            assert_eq!(hybrid.output(0), plain.output(0), "구멍 {holes:07b}");
            assert_eq!(hybrid.prepared().len(), times.len());
        }
    }
}
