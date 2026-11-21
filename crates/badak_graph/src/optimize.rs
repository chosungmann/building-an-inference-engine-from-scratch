//! 그래프 최적화 (30장).
//!
//! 그래프를 실행하기 전에, 같은 답을 내는 더 싼 그래프로 고쳐 쓴다. 고쳐
//! 쓰는 규칙 하나하나를 **패스**(pass)라고 한다. 패스는 그래프에서 정해진
//! 모양을 찾아 다른 모양으로 바꾸고, 바꾼 횟수를 돌려준다.
//!
//! 패스는 서로에게 일거리를 만든다. 축을 바꾸는 마디 둘이 지워져야 합성곱과
//! 배치 정규화가 이웃이 되고, 필터가 상수가 되어야 배치 정규화를 접을 수
//! 있다. 그래서 어느 패스도 더 바꿀 것이 없을 때까지 모든 패스를 되풀이한다
//! ([`optimize`]).

use badak_core::Tensor;

use crate::error::{Error, Result};
use crate::graph::{Activation, Graph, Node, Op, Padding};
use crate::kernels::{View, ViewMut};
use crate::registry::Registry;

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

// ANCHOR: uses
/// 값마다, 그 값을 만드는 마디와 읽는 마디들. 패스가 이웃을 찾을 때 쓴다.
struct Uses {
    /// 값을 만드는 마디의 번호.
    producer: Vec<Option<usize>>,
    /// 값을 읽는 마디들의 번호. 한 마디가 두 번 읽으면 두 번 적힌다.
    consumers: Vec<Vec<usize>>,
    /// 그래프의 출력인지.
    is_output: Vec<bool>,
}

impl Uses {
    fn of(graph: &Graph) -> Self {
        let count = graph.values.len();
        let mut uses = Self {
            producer: vec![None; count],
            consumers: vec![Vec::new(); count],
            is_output: vec![false; count],
        };
        for (index, node) in graph.nodes.iter().enumerate() {
            for &output in &node.outputs {
                uses.producer[output] = Some(index);
            }
            for &input in &node.inputs {
                uses.consumers[input].push(index);
            }
        }
        for &output in &graph.outputs {
            uses.is_output[output] = true;
        }
        uses
    }

    /// 값을 읽는 것이 마디 `consumer` 하나뿐인지. 그래야 그 값을 없애도
    /// 된다.
    fn only(&self, value: usize, consumer: usize) -> bool {
        self.consumers[value] == [consumer] && !self.is_output[value]
    }
}
// ANCHOR_END: uses

impl Graph {
    /// 값 `old`를 읽던 것들이 모두 `new`를 읽게 한다.
    fn redirect(&mut self, old: usize, new: usize) {
        let all = self
            .nodes
            .iter_mut()
            .flat_map(|node| &mut node.inputs)
            .chain(&mut self.outputs);
        for value in all {
            if *value == old {
                *value = new;
            }
        }
    }

    /// 계산될 값을 하나 더한다. 형상을 알고 있을 때 쓴다.
    fn fresh(&mut self, name: String, shape: Vec<usize>) -> usize {
        self.value(&name, &shape)
    }

    /// 값이 `f32` 상수이면 그 원소들.
    fn floats(&self, value: usize) -> Option<Vec<f32>> {
        self.values[value].constant.as_ref()?.to_vec::<f32>().ok()
    }

    // ANCHOR: prune
    /// 아무도 읽지 않는 마디와 값을 지운다. 지운 마디의 수를 돌려준다.
    ///
    /// 패스는 마디를 지우지 않는다. 읽는 쪽의 연결만 바꿔 놓는다. 그러면
    /// 예전 마디는 결과를 아무도 읽지 않게 되고, 여기서 한꺼번에 사라진다.
    /// 마디 하나를 지우면 그 입력을 만들던 마디도 쓸모없어질 수 있으므로
    /// 더 지울 것이 없을 때까지 되풀이한다. 값의 번호는 다시 매긴다.
    pub fn prune(&mut self) -> usize {
        let before = self.nodes.len();
        loop {
            let uses = Uses::of(self);
            let live = |node: &Node| {
                node.outputs
                    .iter()
                    .any(|&value| uses.is_output[value] || !uses.consumers[value].is_empty())
            };
            let count = self.nodes.len();
            self.nodes.retain(live);
            if self.nodes.len() == count {
                break;
            }
        }

        // 남은 마디와 그래프의 입출력이 가리키는 값만 남긴다.
        let mut kept = vec![false; self.values.len()];
        let referenced = self
            .nodes
            .iter()
            .flat_map(|node| node.inputs.iter().chain(&node.outputs))
            .chain(&self.inputs)
            .chain(&self.outputs);
        for &value in referenced {
            kept[value] = true;
        }
        let mut renumbered = vec![usize::MAX; self.values.len()];
        let mut next = 0;
        for (value, &keep) in kept.iter().enumerate() {
            if keep {
                renumbered[value] = next;
                next += 1;
            }
        }
        let mut index = 0;
        self.values.retain(|_| {
            index += 1;
            kept[index - 1]
        });
        let all = self
            .nodes
            .iter_mut()
            .flat_map(|node| node.inputs.iter_mut().chain(&mut node.outputs))
            .chain(&mut self.inputs)
            .chain(&mut self.outputs);
        for value in all {
            *value = renumbered[*value];
        }
        before - self.nodes.len()
    }
    // ANCHOR_END: prune
}

// ANCHOR: fold_constants
/// 상수 접기. 입력이 모두 상수인 마디를 지금 실행해 결과를 상수로 둔다.
///
/// 마디를 실행하는 데는 해석기가 쓰는 바로 그 커널을 쓴다. 그래서 미리
/// 계산한 값과 실행할 때 계산했을 값이 같다.
pub fn fold_constants(graph: &mut Graph, registry: &Registry) -> Result<usize> {
    let mut folded = 0;
    for index in graph.order()? {
        let node = &graph.nodes[index];
        let constants: Option<Vec<&Tensor>> = node
            .inputs
            .iter()
            .map(|&value| graph.values[value].constant.as_ref())
            .collect();
        let (Some(constants), [output]) = (constants, &node.outputs[..]) else {
            continue;
        };
        // 커널이 없는 연산이나 f32가 아닌 상수는 건드리지 않는다.
        let Ok(kernel) = registry.resolve(&node.op) else {
            continue;
        };
        let tensors: Vec<Tensor> = constants.iter().map(|tensor| tensor.contiguous()).collect();
        let Ok(slices) = tensors
            .iter()
            .map(|tensor| tensor.as_slice::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
        else {
            continue;
        };
        let shapes: Vec<&[usize]> = tensors.iter().map(Tensor::shape).collect();
        let [shape] = &kernel.shapes(&shapes)?[..] else {
            continue;
        };
        let views: Vec<View<'_>> = shapes
            .iter()
            .zip(&slices)
            .map(|(&shape, &data)| View { shape, data })
            .collect();
        let mut data = vec![0.0_f32; shape.iter().product()];
        kernel.run(
            &views,
            &mut [ViewMut {
                shape,
                data: &mut data,
            }],
        )?;
        let output = *output;
        graph.values[output].shape.clone_from(shape);
        graph.values[output].constant = Some(Tensor::from_vec(data, shape)?);
        // 이 마디의 출력은 이제 상수다. 마디가 남아 있으면 상수에 쓰려는
        // 마디가 되므로 출력을 떼어 둔다. `prune`이 지운다.
        graph.nodes[index].outputs.clear();
        folded += 1;
    }
    graph.nodes.retain(|node| !node.outputs.is_empty());
    Ok(folded)
}
// ANCHOR_END: fold_constants

/// `IDENTITY`를 지운다. 그 결과를 읽던 마디들이 입력을 바로 읽게 한다.
pub fn remove_identities(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut removed = 0;
    for index in 0..graph.nodes.len() {
        let node = &graph.nodes[index];
        if node.op == Op::Identity
            && let (&[input], &[output]) = (&node.inputs[..], &node.outputs[..])
        {
            graph.redirect(output, input);
            removed += 1;
        }
    }
    Ok(removed)
}

// ANCHOR: merge_duplicates
/// 같은 입력에 같은 연산을 하는 마디들을 하나로 합친다 (32장).
///
/// 기계적으로 옮기면 같은 계산이 여러 번 생긴다. 한 값을 읽는 마디가
/// 넷이고 넷 모두 축의 순서를 따지면, 그 값을 NHWC로 바꾸는 `TRANSPOSE`가
/// 넷 생긴다. 하나만 남기고 나머지를 읽던 마디들이 그것을 읽게 한다.
pub fn merge_duplicates(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut merged = 0;
    for later in 0..graph.nodes.len() {
        let node = &graph.nodes[later];
        if node.outputs.len() != 1 || matches!(node.op, Op::Custom { .. }) {
            continue;
        }
        let earlier = graph.nodes[..later].iter().find(|other| {
            other.op == node.op && other.inputs == node.inputs && other.outputs.len() == 1
        });
        if let Some(earlier) = earlier {
            let (old, new) = (node.outputs[0], earlier.outputs[0]);
            if old != new {
                graph.redirect(old, new);
                merged += 1;
            }
        }
    }
    Ok(merged)
}
// ANCHOR_END: merge_duplicates

/// 순서를 잇달아 바꾼 결과. `first`를 한 다음 `second`를 한 것과 같은 순서다.
fn compose(first: &[usize], second: &[usize]) -> Vec<usize> {
    second.iter().map(|&axis| first[axis]).collect()
}

fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(index, &axis)| index == axis)
}

/// 마디가 `TRANSPOSE`이면 입력 값과 축의 순서.
fn as_transpose(node: &Node) -> Option<(usize, &[usize])> {
    match (&node.op, &node.inputs[..]) {
        (Op::Transpose { perm }, &[input]) => Some((input, perm)),
        _ => None,
    }
}

// ANCHOR: cancel_transposes
/// 잇달아 있는 `TRANSPOSE` 둘을 하나로 합친다. 합친 것이 아무것도 바꾸지
/// 않으면 지운다.
pub fn cancel_transposes(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut changed = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let Some((middle, second)) = as_transpose(&graph.nodes[index]) else {
            continue;
        };
        let output = graph.nodes[index].outputs[0];
        if is_identity(second) {
            graph.redirect(output, middle);
            changed += 1;
            continue;
        }
        let Some((source, first)) = uses.producer[middle]
            .and_then(|producer| as_transpose(&graph.nodes[producer]))
            .filter(|(_, first)| first.len() == second.len())
        else {
            continue;
        };
        let perm = compose(first, second);
        if is_identity(&perm) {
            // 갔다가 돌아온 것이다. 처음 값을 바로 읽는다.
            graph.redirect(output, source);
        } else {
            graph.nodes[index].op = Op::Transpose { perm };
            graph.nodes[index].inputs = vec![source];
        }
        changed += 1;
    }
    Ok(changed)
}
// ANCHOR_END: cancel_transposes

// ANCHOR: sink_transposes
/// `TRANSPOSE`를 원소마다 하는 연산의 뒤로 옮긴다.
///
/// 원소마다 하는 연산은 원소가 어디에 놓였는지를 따지지 않는다. 그래서
/// "축을 바꾸고 ReLU"와 "ReLU를 하고 축을 바꾸기"는 같다. 뒤로 옮긴
/// `TRANSPOSE`는 다음 마디 앞의 `TRANSPOSE`와 만나 [`cancel_transposes`]에서
/// 지워진다. 입력이 둘인 연산은 두 입력이 같은 순서로 바뀌어 있을 때만
/// 옮길 수 있다.
pub fn sink_transposes(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut moved = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        // 32장: 시그모이드도 원소마다 하는 연산이다. 이어 붙이기는 원소를
        // 계산하지 않고 옮기기만 하므로 역시 뒤로 보낼 수 있다. 다만 이어
        // 붙이는 축이 달라진다.
        let elementwise = matches!(
            node.op,
            Op::Relu
                | Op::Clip { .. }
                | Op::Add { .. }
                | Op::Mul { .. }
                | Op::Sigmoid
                | Op::Concat { .. }
        );
        if !elementwise || node.outputs.len() != 1 {
            continue;
        }
        // 모든 입력이 같은 순서의 TRANSPOSE에서 와야 한다.
        let sources: Option<Vec<(usize, &[usize])>> = node
            .inputs
            .iter()
            .map(|&input| uses.producer[input].and_then(|p| as_transpose(&graph.nodes[p])))
            .collect();
        let Some(sources) = sources.filter(|sources| !sources.is_empty()) else {
            continue;
        };
        let perm = sources[0].1.to_vec();
        if sources.iter().any(|(_, other)| *other != perm) {
            continue;
        }
        let inputs: Vec<usize> = sources.iter().map(|&(source, _)| source).collect();
        let output = node.outputs[0];
        // 바뀐 뒤의 축 `axis`는 바뀌기 전의 축 `perm[axis]`다.
        let concat = match node.op {
            Op::Concat { axis } if axis < perm.len() => Some(perm[axis]),
            Op::Concat { .. } => continue,
            _ => None,
        };
        // 연산은 바꾸기 전의 값들을 읽어 새 값을 만들고, 그 새 값의 축을
        // 바꾼 것이 예전의 출력이 된다. 예전 출력을 읽던 마디들은 그대로다.
        let name = format!("{}/sunk", graph.values[output].name);
        let mut shape = graph.values[inputs[0]].shape.clone();
        if let Some(axis) = concat {
            shape[axis] = inputs
                .iter()
                .map(|&input| graph.values[input].shape[axis])
                .sum();
            graph.nodes[index].op = Op::Concat { axis };
        }
        let inner = graph.fresh(name, shape);
        graph.nodes[index].inputs = inputs;
        graph.nodes[index].outputs = vec![inner];
        graph.node(Op::Transpose { perm }, &[inner], &[output]);
        moved += 1;
    }
    Ok(moved)
}
// ANCHOR_END: sink_transposes

/// 길이가 1인 축을 빼고 보면 순서가 그대로인지. 그런 `TRANSPOSE`는 원소를
/// 옮기지 않는다. 형상만 달라진다.
fn keeps_order(perm: &[usize], shape: &[usize]) -> bool {
    let moved: Vec<usize> = perm
        .iter()
        .copied()
        .filter(|&axis| shape.get(axis).is_some_and(|&length| length != 1))
        .collect();
    perm.len() == shape.len() && moved.is_sorted()
}

// ANCHOR: simplify_reshapes
/// 형상만 바꾸는 마디들을 줄인다.
///
/// - `FLATTEN` 앞의, 원소를 옮기지 않는 `TRANSPOSE`를 지운다.
/// - "축을 남기는 `MEAN`" 뒤의 `FLATTEN`을, 축을 남기지 않는 `MEAN` 하나로
///   바꾼다.
///
/// 둘 다 값의 형상을 알아야 판단할 수 있다.
pub fn simplify_reshapes(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut changed = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (&Op::Flatten { axis }, &[input], &[output]) =
            (&node.op, &node.inputs[..], &node.outputs[..])
        else {
            continue;
        };
        let Some(producer) = uses.producer[input] else {
            continue;
        };
        let wanted = &graph.values[output].shape;
        let before = &graph.nodes[producer];
        if let Some((source, perm)) = as_transpose(before) {
            let shape = &graph.values[source].shape;
            // 바꾸기 전의 형상을 어느 축에서 끊으면 같은 행렬이 되는지.
            let split = (0..=shape.len()).find(|&split| {
                let (rows, columns) = shape.split_at(split);
                [rows.iter().product::<usize>(), columns.iter().product()] == wanted[..]
            });
            if let Some(split) = split.filter(|_| keeps_order(perm, shape)) {
                graph.nodes[index].op = Op::Flatten { axis: split };
                graph.nodes[index].inputs = vec![source];
                changed += 1;
            }
        } else if let Op::Mean {
            axes,
            keep_dims: true,
        } = &before.op
            && uses.only(input, index)
        {
            let source = &graph.values[before.inputs[0]].shape;
            let squeezed: Vec<usize> = (0..source.len())
                .filter(|axis| !axes.contains(axis))
                .map(|axis| source[axis])
                .collect();
            if squeezed == *wanted && axis <= source.len() {
                let axes = axes.clone();
                graph.nodes[producer].op = Op::Mean {
                    axes,
                    keep_dims: false,
                };
                graph.nodes[producer].outputs = vec![output];
                graph.nodes[index].outputs.clear();
                changed += 1;
            }
        }
    }
    graph.nodes.retain(|node| !node.outputs.is_empty());
    Ok(changed)
}
// ANCHOR_END: simplify_reshapes

// ANCHOR: swap_reshapes
/// "NCHW로 되돌린 뒤 높이와 너비를 한 축으로 합치기"를 "합친 뒤에 축
/// 바꾸기"로 고친다 (32장).
///
/// 검출 모델의 끝에서 특징 지도 `(장 수, 채널, 높이, 너비)`를 칸의 목록
/// `(장 수, 채널, 칸 수)`로 다시 묶는다. 우리 그래프에서 그 앞에는 NHWC를
/// NCHW로 되돌리는 `TRANSPOSE`가 있다. NHWC `(장 수, 높이, 너비, 채널)`의
/// 가운데 두 축을 합치면 `(장 수, 칸 수, 채널)`이고, 그것의 뒤 두 축을
/// 바꾼 것이 원래 원하던 텐서다. 이렇게 고쳐 두면 `TRANSPOSE`가 `RESHAPE`
/// 뒤로 나와서 더 뒤로 갈 수 있다.
pub fn swap_reshapes(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut swapped = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (Op::Reshape { shape }, &[input], &[output]) =
            (&node.op, &node.inputs[..], &node.outputs[..])
        else {
            continue;
        };
        let Some((source, perm)) =
            uses.producer[input].and_then(|producer| as_transpose(&graph.nodes[producer]))
        else {
            continue;
        };
        let &[batch, height, width, channels] = &graph.values[source].shape[..] else {
            continue;
        };
        if perm != [0, 3, 1, 2] || shape[..] != [batch, channels, height * width] {
            continue;
        }
        let name = format!("{}/cells", graph.values[output].name);
        let inner = graph.fresh(name, vec![batch, height * width, channels]);
        graph.nodes[index].op = Op::Reshape {
            shape: vec![batch, height * width, channels],
        };
        graph.nodes[index].inputs = vec![source];
        graph.nodes[index].outputs = vec![inner];
        graph.node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            &[inner],
            &[output],
        );
        swapped += 1;
    }
    Ok(swapped)
}
// ANCHOR_END: swap_reshapes

// ANCHOR: fuse_silu
/// `x × sigmoid(x)`를 앞 연산의 활성화 함수로 붙인다 (32장).
///
/// SiLU는 ONNX에 연산으로 없어서 `Sigmoid`와 `Mul` 두 마디로 적혀 있다.
/// 모양은 마름모다. `x`를 시그모이드와 곱셈이 함께 읽고, 곱셈이 시그모이드의
/// 결과를 읽는다. `x`를 읽는 것이 그 둘뿐이고 시그모이드의 결과를 읽는 것이
/// 그 곱셈뿐일 때 세 마디를 하나로 합친다.
pub fn fuse_silu(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut fused = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (
            Op::Mul {
                activation: Activation::None,
            },
            &[a, b],
            &[output],
        ) = (&node.op, &node.inputs[..], &node.outputs[..])
        else {
            continue;
        };
        // 두 입력 가운데 하나가 다른 하나의 시그모이드여야 한다.
        let sigmoid_of = |gate: usize, x: usize| {
            uses.producer[gate].filter(|&producer| {
                graph.nodes[producer].op == Op::Sigmoid && graph.nodes[producer].inputs == [x]
            })
        };
        let Some((x, gate, sigmoid)) = sigmoid_of(a, b)
            .map(|sigmoid| (b, a, sigmoid))
            .or_else(|| sigmoid_of(b, a).map(|sigmoid| (a, b, sigmoid)))
        else {
            continue;
        };
        let mut readers = uses.consumers[x].clone();
        readers.sort_unstable();
        let mut expected = [sigmoid, index];
        expected.sort_unstable();
        if readers != expected || uses.is_output[x] || !uses.only(gate, index) {
            continue;
        }
        let Some(producer) = uses.producer[x] else {
            continue;
        };
        let slot = match &mut graph.nodes[producer].op {
            Op::Conv2d { activation, .. }
            | Op::DepthwiseConv2d { activation, .. }
            | Op::FullyConnected { activation, .. }
            | Op::Add { activation } => activation,
            _ => continue,
        };
        if *slot != Activation::None {
            continue;
        }
        *slot = Activation::Silu;
        graph.nodes[producer].outputs = vec![output];
        graph.nodes[index].outputs.clear();
        graph.nodes[sigmoid].outputs.clear();
        fused += 1;
    }
    graph.nodes.retain(|node| !node.outputs.is_empty());
    Ok(fused)
}
// ANCHOR_END: fuse_silu

// ANCHOR: fold_batch_norms
/// 배치 정규화를 앞의 합성곱에 접어 넣는다.
///
/// 합성곱의 출력 채널 `c`는 `Σ w·x + bias`이고, 배치 정규화는 거기에
/// `a`를 곱하고 `b`를 더한다(`a = 배율 / √(분산 + ε)`, `b = 치우침 −
/// 평균·a`). 곱셈을 합 안으로 넣으면 `Σ (a·w)·x + (a·bias + b)`다. 필터와
/// 치우침을 미리 그렇게 고쳐 두면 배치 정규화 마디는 필요 없다.
///
/// 조건이 있다. 필터와 배치 정규화의 값들이 상수여야 하고, 합성곱의 결과를
/// 배치 정규화만 읽어야 하고(다른 마디가 정규화 전의 값을 읽으면 그 값을
/// 없앨 수 없다), 합성곱에 활성화 함수가 붙어 있지 않아야 한다(사이에
/// 직선이 아닌 것이 끼면 곱셈을 안으로 넣을 수 없다).
pub fn fold_batch_norms(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut folded = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (&Op::BatchNorm { epsilon }, &[x, scale, bias, mean, variance]) =
            (&node.op, &node.inputs[..])
        else {
            continue;
        };
        let Some(producer) = uses.producer[x].filter(|_| uses.only(x, index)) else {
            continue;
        };
        let conv = &graph.nodes[producer];
        // 필터에서 출력 채널이 놓인 축. 합성곱은 맨 앞, 깊이별 합성곱은 맨 뒤다.
        let leading = match conv.op {
            Op::Conv2d {
                activation: Activation::None,
                ..
            } => true,
            Op::DepthwiseConv2d {
                activation: Activation::None,
                ..
            } => false,
            _ => continue,
        };
        let filter = conv.inputs[1];
        let old_bias = conv.inputs.get(2).copied();
        let parameters = [scale, bias, mean, variance].map(|value| graph.floats(value));
        let [Some(scale), Some(bias), Some(mean), Some(variance)] = parameters else {
            continue;
        };
        let Some(mut weights) = graph.floats(filter) else {
            continue;
        };
        let channels = scale.len();
        let mut shifts = match old_bias {
            Some(value) => match graph.floats(value) {
                Some(values) => values,
                None => continue,
            },
            None => vec![0.0; channels],
        };
        let shape = graph.values[filter].shape.clone();
        let out_axis = if leading { 0 } else { shape.len() - 1 };
        if shape[out_axis] != channels || shifts.len() != channels || channels == 0 {
            return Err(invalid("배치 정규화와 합성곱의 채널 수가 다릅니다"));
        }

        let factors: Vec<f32> = (0..channels)
            .map(|c| scale[c] / (variance[c] + epsilon).sqrt())
            .collect();
        if leading {
            // (출력 채널, 높이, 너비, 입력 채널): 채널마다 한 덩어리다.
            let block = weights.len() / channels;
            for (chunk, &factor) in weights.chunks_exact_mut(block).zip(&factors) {
                chunk.iter_mut().for_each(|weight| *weight *= factor);
            }
        } else {
            // (1, 높이, 너비, 출력 채널): 채널이 번갈아 놓여 있다.
            for pixel in weights.chunks_exact_mut(channels) {
                pixel
                    .iter_mut()
                    .zip(&factors)
                    .for_each(|(weight, &factor)| *weight *= factor);
            }
        }
        for c in 0..channels {
            shifts[c] = (shifts[c] - mean[c]) * factors[c] + bias[c];
        }

        let name = &graph.values[filter].name;
        let new_filter_name = format!("{name}/folded");
        let new_bias_name = format!("{name}/folded_bias");
        let new_filter = graph.constant(&new_filter_name, Tensor::from_vec(weights, &shape)?);
        let new_bias = graph.constant(&new_bias_name, Tensor::from_vec(shifts, &[channels])?);
        let output = graph.nodes[index].outputs[0];
        let input = graph.nodes[producer].inputs[0];
        graph.nodes[producer].inputs = vec![input, new_filter, new_bias];
        graph.nodes[producer].outputs = vec![output];
        graph.nodes[index].outputs.clear();
        folded += 1;
    }
    graph.nodes.retain(|node| !node.outputs.is_empty());
    Ok(folded)
}
// ANCHOR_END: fold_batch_norms

/// 따로 있는 활성화 함수 마디가, 연산에 붙일 수 있는 것이면 그 종류.
fn as_activation(op: &Op) -> Option<Activation> {
    match *op {
        Op::Relu => Some(Activation::Relu),
        Op::Clip { min, max } if min == 0.0 && max == 6.0 => Some(Activation::Relu6),
        Op::Clip { min, max } if min == -1.0 && max == 1.0 => Some(Activation::ReluN1To1),
        Op::Clip { min, max } if min == 0.0 && max == f32::INFINITY => Some(Activation::Relu),
        _ => None,
    }
}

// ANCHOR: fuse_activations
/// 활성화 함수 마디를 앞의 연산에 붙인다.
///
/// 붙일 수 있는 연산은 커널이 결과를 적으면서 곧바로 활성화 함수를
/// 적용하는 것들이다(24.7절, 28장). 앞 연산의 결과를 활성화 함수만 읽어야
/// 한다.
pub fn fuse_activations(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut fused = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (Some(wanted), &[input], &[output]) =
            (as_activation(&node.op), &node.inputs[..], &node.outputs[..])
        else {
            continue;
        };
        let Some(producer) = uses.producer[input].filter(|_| uses.only(input, index)) else {
            continue;
        };
        let slot = match &mut graph.nodes[producer].op {
            Op::Conv2d { activation, .. }
            | Op::DepthwiseConv2d { activation, .. }
            | Op::FullyConnected { activation, .. }
            | Op::Add { activation }
            | Op::Mul { activation } => activation,
            _ => continue,
        };
        if *slot != Activation::None {
            continue;
        }
        *slot = wanted;
        graph.nodes[producer].outputs = vec![output];
        graph.nodes[index].outputs.clear();
        fused += 1;
    }
    graph.nodes.retain(|node| !node.outputs.is_empty());
    Ok(fused)
}
// ANCHOR_END: fuse_activations

// ANCHOR: fuse_pads
/// 0을 패딩하는 `PAD`를 뒤의 합성곱에 합친다.
///
/// 합성곱은 패딩한 자리를 0으로 읽는다(27장). 그러니 0을 패딩한 텐서를 따로
/// 만들 필요가 없다. 합성곱에 덧댈 길이를 알려 주면 된다. 풀링에는 합치지
/// 않는다. 최댓값 풀링은 패딩한 자리를 아예 보지 않으므로 0을 패딩한 것과
/// 다르다.
pub fn fuse_pads(graph: &mut Graph, _: &Registry) -> Result<usize> {
    let mut fused = 0;
    for index in 0..graph.nodes.len() {
        let uses = Uses::of(graph);
        let node = &graph.nodes[index];
        let (Op::Pad { amounts, value }, &[source], &[padded]) =
            (&node.op, &node.inputs[..], &node.outputs[..])
        else {
            continue;
        };
        // 높이와 너비에만 0을 패딩하는 것이어야 한다.
        let &[[0, 0], rows, columns, [0, 0]] = &amounts[..] else {
            continue;
        };
        if *value != 0.0 {
            continue;
        }
        let Some(&consumer) = uses.consumers[padded]
            .first()
            .filter(|&&c| uses.only(padded, c))
        else {
            continue;
        };
        let conv = &mut graph.nodes[consumer];
        if conv.inputs[0] != padded {
            continue;
        }
        let (Op::Conv2d { padding, .. } | Op::DepthwiseConv2d { padding, .. }) = &mut conv.op
        else {
            continue;
        };
        *padding = match *padding {
            Padding::Valid => Padding::Explicit([rows, columns]),
            Padding::Explicit([r, c]) => Padding::Explicit([
                [r[0] + rows[0], r[1] + rows[1]],
                [c[0] + columns[0], c[1] + columns[1]],
            ]),
            // 규칙으로 정한 길이에 더하려면 그 길이를 먼저 숫자로 바꿔야
            // 한다. 이 책의 모델에는 그런 경우가 없어 두었다.
            Padding::Same => continue,
        };
        conv.inputs[0] = source;
        fused += 1;
    }
    Ok(fused)
}
// ANCHOR_END: fuse_pads

// ANCHOR: passes
/// 패스 하나. 그래프를 고치고, 고친 횟수를 돌려준다.
pub type Pass = fn(&mut Graph, &Registry) -> Result<usize>;

/// 이 크레이트의 패스들과 그 이름. 적힌 순서대로 실행한다.
pub const PASSES: [(&str, Pass); 11] = [
    ("상수 접기", fold_constants),
    // 32장에서 더했다.
    ("같은 계산 합치기", merge_duplicates),
    ("IDENTITY 지우기", remove_identities),
    ("TRANSPOSE 뒤로 옮기기", sink_transposes),
    ("TRANSPOSE 상쇄", cancel_transposes),
    ("형상 바꾸기 줄이기", simplify_reshapes),
    // 32장에서 더했다.
    ("RESHAPE와 TRANSPOSE 바꾸기", swap_reshapes),
    ("배치 정규화 접기", fold_batch_norms),
    ("활성화 함수 붙이기", fuse_activations),
    // 32장에서 더했다.
    ("SiLU 붙이기", fuse_silu),
    ("PAD 합치기", fuse_pads),
];

/// 패스 하나가 한 일.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    /// 패스의 이름.
    pub name: &'static str,
    /// 고친 횟수.
    pub changes: usize,
}

/// 주어진 패스들을, 어느 것도 더 고칠 것이 없을 때까지 되풀이한다. 패스마다
/// 고친 횟수를 모두 더해 돌려준다.
///
/// 8비트 정수로 계산하는 그래프(29장)는 건드리지 않는다. 여기의 패스들은
/// 값이 실수라고 가정한다. 예를 들어 정수 그래프에서 패딩하는 값은 0이 아니라
/// 영점이다.
pub fn run(
    graph: &mut Graph,
    registry: &Registry,
    passes: &[(&'static str, Pass)],
) -> Result<Vec<Applied>> {
    let mut applied: Vec<Applied> = passes
        .iter()
        .map(|&(name, _)| Applied { name, changes: 0 })
        .collect();
    if graph.is_quantized()
        || graph
            .values
            .iter()
            .any(|value| value.quantization.is_some())
    {
        return Ok(applied);
    }
    // 패스가 형상을 본다. 비어 있을 수 있으므로 먼저 채운다.
    graph.infer_shapes(registry)?;
    // 패스 하나는 마디 하나를 많아야 한 번 고치고, 고칠 때마다 마디가
    // 줄거나 TRANSPOSE가 출력 쪽으로 한 칸 간다. 그래도 끝나지 않는
    // 규칙의 짝이 생기면 알 수 있게 횟수를 막아 둔다.
    let limit = 16 + graph.nodes.len();
    for _ in 0..limit {
        let mut changed = 0;
        for (&(_, pass), record) in passes.iter().zip(&mut applied) {
            let changes = pass(graph, registry)?;
            graph.prune();
            record.changes += changes;
            changed += changes;
        }
        if changed == 0 {
            // 고쳐 쓴 그래프가 여전히 올바른지 확인하고 형상을 다시 구한다.
            graph.infer_shapes(registry)?;
            return Ok(applied);
        }
    }
    Err(invalid("그래프 최적화가 끝나지 않습니다"))
}

/// 모든 패스로 그래프를 최적화한다.
pub fn optimize(graph: &mut Graph, registry: &Registry) -> Result<Vec<Applied>> {
    run(graph, registry, &PASSES)
}
// ANCHOR_END: passes

#[cfg(test)]
mod tests {
    use badak_formats::onnx;

    use super::*;
    use crate::interpreter::Interpreter;
    use crate::memory::Strategy;

    /// 시험용 값들. 식으로 정해지고 0을 가운데 두고 퍼져 있다.
    fn noise(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn constant(graph: &mut Graph, name: &str, shape: &[usize], seed: u32) -> usize {
        let data = noise(shape.iter().product(), seed);
        graph.constant(name, Tensor::from_vec(data, shape).unwrap())
    }

    /// 양수인 상수. 분산으로 쓴다.
    fn positive(graph: &mut Graph, name: &str, count: usize, seed: u32) -> usize {
        let data = noise(count, seed).iter().map(|v| v.abs() + 0.1).collect();
        graph.constant(name, Tensor::from_vec(data, &[count]).unwrap())
    }

    /// 그래프를 실행해 첫 출력을 돌려준다. 입력은 식으로 정한 값이다.
    fn run_graph(graph: &Graph, in_place: bool) -> Vec<f32> {
        let registry = Registry::builtin();
        let mut graph = graph.clone();
        graph.infer_shapes(&registry).unwrap();
        let mut interpreter = if in_place {
            Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap()
        } else {
            Interpreter::new(&graph, &registry).unwrap()
        };
        // 남아 있던 값에 기대는 커널이 있으면 드러나게 한다.
        interpreter.scribble(f32::NAN);
        for index in 0..graph.inputs.len() {
            let input = interpreter.input_mut(index);
            let values = noise(input.len(), 7 + index as u32);
            input.copy_from_slice(&values);
        }
        interpreter.run().unwrap();
        interpreter.output(0).to_vec()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance * e.abs().max(1.0),
                "{index}번째: {a} 대 {e}"
            );
        }
    }

    fn count(graph: &Graph, name: &str) -> usize {
        graph
            .nodes
            .iter()
            .filter(|node| node.op.name() == name)
            .count()
    }

    fn apply(graph: &mut Graph, pass: Pass) -> usize {
        let registry = Registry::builtin();
        graph.infer_shapes(&registry).unwrap();
        let changes = pass(graph, &registry).unwrap();
        graph.prune();
        graph.order().unwrap();
        changes
    }

    fn conv_op(padding: Padding, activation: Activation) -> Op {
        Op::Conv2d {
            padding,
            stride: [1, 1],
            dilation: [1, 1],
            activation,
        }
    }

    const ADD: Op = Op::Add {
        activation: Activation::None,
    };

    /// 합성곱(또는 깊이별 합성곱) 뒤에 배치 정규화가 있는 그래프.
    fn conv_then_norm(depthwise: bool, with_bias: bool, activation: Activation) -> Graph {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[1, 5, 4, 3]);
        let (op, filter, channels) = if depthwise {
            let op = Op::DepthwiseConv2d {
                padding: Padding::Same,
                stride: [1, 1],
                dilation: [1, 1],
                multiplier: 2,
                activation,
            };
            (op, constant(&mut graph, "w", &[1, 3, 3, 6], 1), 6)
        } else {
            let op = conv_op(Padding::Same, activation);
            (op, constant(&mut graph, "w", &[4, 3, 3, 3], 1), 4)
        };
        let mut inputs = vec![x, filter];
        if with_bias {
            inputs.push(constant(&mut graph, "b", &[channels], 2));
        }
        let h = graph.value("h", &[]);
        graph.node(op, &inputs, &[h]);
        let scale = constant(&mut graph, "scale", &[channels], 3);
        let bias = constant(&mut graph, "bias", &[channels], 4);
        let mean = constant(&mut graph, "mean", &[channels], 5);
        let variance = positive(&mut graph, "variance", channels, 6);
        let y = graph.value("y", &[]);
        graph.node(
            Op::BatchNorm { epsilon: 1e-5 },
            &[h, scale, bias, mean, variance],
            &[y],
        );
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        graph
    }

    #[test]
    fn folding_batch_norm_keeps_the_answer() {
        for depthwise in [false, true] {
            for with_bias in [false, true] {
                let mut graph = conv_then_norm(depthwise, with_bias, Activation::None);
                let expected = run_graph(&graph, false);
                assert_eq!(apply(&mut graph, fold_batch_norms), 1);
                assert_eq!(graph.nodes.len(), 1);
                // 예전 필터와 배치 정규화의 값들은 사라지고, 새 필터와
                // 치우침, 입력과 출력만 남는다.
                assert_eq!(graph.values.len(), 4);
                assert_close(&run_graph(&graph, false), &expected, 1e-5);
            }
        }
    }

    #[test]
    fn batch_norm_is_not_folded_when_it_would_change_the_answer() {
        // 합성곱에 활성화 함수가 붙어 있다.
        let mut fused = conv_then_norm(false, true, Activation::Relu);
        assert_eq!(apply(&mut fused, fold_batch_norms), 0);
        // 합성곱의 결과를 다른 마디도 읽는다.
        let mut shared = conv_then_norm(false, true, Activation::None);
        let h = shared.nodes[0].outputs[0];
        let (y, z) = (shared.outputs[0], shared.value("z", &[]));
        shared.node(ADD, &[h, y], &[z]);
        shared.outputs = vec![z];
        assert_eq!(apply(&mut shared, fold_batch_norms), 0);
        // 합성곱의 결과가 그래프의 출력이기도 하다.
        let mut exposed = conv_then_norm(false, true, Activation::None);
        exposed.outputs.push(h);
        assert_eq!(apply(&mut exposed, fold_batch_norms), 0);
        // 필터가 상수가 아니다.
        let mut computed = conv_then_norm(false, false, Activation::None);
        let filter = computed.nodes[0].inputs[1];
        computed.values[filter].constant = None;
        computed.inputs.push(filter);
        assert_eq!(apply(&mut computed, fold_batch_norms), 0);
    }

    #[test]
    fn constants_are_folded_with_the_real_kernels() {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[2, 3]);
        let w = constant(&mut graph, "w", &[3, 2], 1);
        let turned = graph.value("turned", &[]);
        let same = graph.value("same", &[]);
        let y = graph.value("y", &[]);
        graph.node(Op::Transpose { perm: vec![1, 0] }, &[w], &[turned]);
        graph.node(Op::Identity, &[turned], &[same]);
        graph.node(ADD, &[x, same], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        let expected = run_graph(&graph, false);
        let original = graph.floats(w).unwrap();

        // 한 번 훑으면서 둘 다 접힌다. 앞의 것이 상수가 되면 뒤의 것도
        // 입력이 모두 상수다.
        assert_eq!(apply(&mut graph, fold_constants), 2);
        assert_eq!(graph.nodes.len(), 1);
        let folded = graph.nodes[0].inputs[1];
        assert_eq!(graph.values[folded].shape, [2, 3]);
        let values = graph.floats(folded).unwrap();
        assert_eq!(values[1], original[2]);
        assert_eq!(run_graph(&graph, false), expected);
        // 입력에서 오는 값을 읽는 마디는 접지 않는다.
        assert_eq!(apply(&mut graph, fold_constants), 0);
    }

    #[test]
    fn identities_between_computed_values_are_removed() {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[4]);
        let (a, b, y) = (
            graph.value("a", &[]),
            graph.value("b", &[]),
            graph.value("y", &[]),
        );
        graph.node(Op::Relu, &[x], &[a]);
        graph.node(Op::Identity, &[a], &[b]);
        graph.node(ADD, &[b, x], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        let expected = run_graph(&graph, false);
        assert_eq!(apply(&mut graph, remove_identities), 1);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(run_graph(&graph, false), expected);
    }

    #[test]
    fn activations_are_attached_only_when_nothing_else_reads_the_value() {
        let build = |activation: Op, shared: bool| {
            let mut graph = Graph::new("시험");
            let x = graph.value("x", &[1, 4, 4, 2]);
            let w = constant(&mut graph, "w", &[3, 3, 3, 2], 1);
            let (h, a, y) = (
                graph.value("h", &[]),
                graph.value("a", &[]),
                graph.value("y", &[]),
            );
            graph.node(conv_op(Padding::Same, Activation::None), &[x, w], &[h]);
            graph.node(activation, &[h], &[a]);
            let other = if shared { h } else { a };
            graph.node(ADD, &[a, other], &[y]);
            graph.inputs = vec![x];
            graph.outputs = vec![y];
            graph
        };
        let relu6 = Op::Clip { min: 0.0, max: 6.0 };
        for (activation, expected) in [
            (Op::Relu, Activation::Relu),
            (relu6.clone(), Activation::Relu6),
            (
                Op::Clip {
                    min: -1.0,
                    max: 1.0,
                },
                Activation::ReluN1To1,
            ),
        ] {
            let mut graph = build(activation, false);
            let before = run_graph(&graph, false);
            assert_eq!(apply(&mut graph, fuse_activations), 1);
            assert_eq!(graph.nodes.len(), 2);
            assert!(
                matches!(graph.nodes[0].op, Op::Conv2d { activation, .. } if activation == expected)
            );
            assert_eq!(run_graph(&graph, false), before);
        }
        // 붙일 수 있는 종류가 아닌 끝값.
        let mut odd = build(Op::Clip { min: 0.0, max: 5.0 }, false);
        assert_eq!(apply(&mut odd, fuse_activations), 0);
        // 합성곱의 결과를 덧셈도 읽는다.
        let mut shared = build(relu6, true);
        assert_eq!(apply(&mut shared, fuse_activations), 0);
        // 이미 붙어 있는 연산에는 또 붙이지 않는다.
        let mut twice = build(Op::Relu, false);
        twice.nodes[0].op = conv_op(Padding::Same, Activation::Relu6);
        assert_eq!(apply(&mut twice, fuse_activations), 0);
    }

    #[test]
    fn transposes_sink_and_cancel() {
        // 두 갈래가 각각 축을 바꾼 뒤 ReLU를 거쳐 더해지고, 다시 되돌린다.
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[1, 2, 3, 4]);
        let names = ["a", "b", "c", "d", "e", "y"];
        let [a, b, c, d, e, y] = names.map(|name| graph.value(name, &[]));
        let to = |perm: [usize; 4]| Op::Transpose {
            perm: perm.to_vec(),
        };
        graph.node(to(crate::onnx::TO_NHWC), &[x], &[a]);
        graph.node(Op::Relu, &[a], &[b]);
        graph.node(to(crate::onnx::TO_NHWC), &[x], &[c]);
        graph.node(
            Op::Clip {
                min: -0.5,
                max: 0.5,
            },
            &[c],
            &[d],
        );
        graph.node(ADD, &[b, d], &[e]);
        graph.node(to(crate::onnx::TO_NCHW), &[e], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        let expected = run_graph(&graph, false);

        let registry = Registry::builtin();
        let applied = optimize(&mut graph, &registry).unwrap();
        assert_eq!(count(&graph, "TRANSPOSE"), 0, "{applied:?}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(run_graph(&graph, false), expected);
    }

    #[test]
    fn transposes_stay_when_the_two_sides_differ() {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[2, 2, 2]);
        let [a, b, y] = ["a", "b", "y"].map(|name| graph.value(name, &[]));
        graph.node(
            Op::Transpose {
                perm: vec![1, 0, 2],
            },
            &[x],
            &[a],
        );
        graph.node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            &[x],
            &[b],
        );
        graph.node(ADD, &[a, b], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        let expected = run_graph(&graph, false);
        optimize(&mut graph, &Registry::builtin()).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(run_graph(&graph, false), expected);

        // 잇달아 바꾼 둘은 하나가 된다. 되돌아오지 않아도 그렇다.
        let mut chain = Graph::new("시험");
        let x = chain.value("x", &[2, 3, 4]);
        let [a, y] = ["a", "y"].map(|name| chain.value(name, &[]));
        chain.node(
            Op::Transpose {
                perm: vec![1, 2, 0],
            },
            &[x],
            &[a],
        );
        chain.node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            &[a],
            &[y],
        );
        chain.inputs = vec![x];
        chain.outputs = vec![y];
        let expected = run_graph(&chain, false);
        assert_eq!(apply(&mut chain, cancel_transposes), 1);
        assert_eq!(chain.nodes.len(), 1);
        assert_eq!(
            chain.nodes[0].op,
            Op::Transpose {
                perm: vec![1, 0, 2]
            }
        );
        assert_eq!(run_graph(&chain, false), expected);
    }

    #[test]
    fn pads_are_merged_into_convolutions_only() {
        let build = |value: f32, pool: bool| {
            let mut graph = Graph::new("시험");
            let x = graph.value("x", &[1, 5, 6, 2]);
            let w = constant(&mut graph, "w", &[3, 3, 3, 2], 1);
            let (padded, y) = (graph.value("padded", &[]), graph.value("y", &[]));
            let amounts = vec![[0, 0], [0, 1], [2, 1], [0, 0]];
            graph.node(Op::Pad { amounts, value }, &[x], &[padded]);
            if pool {
                let op = Op::MaxPool2d {
                    padding: Padding::Valid,
                    stride: [2, 2],
                    filter: [3, 3],
                };
                graph.node(op, &[padded], &[y]);
            } else {
                let mut op = conv_op(Padding::Valid, Activation::Relu);
                if let Op::Conv2d { stride, .. } = &mut op {
                    *stride = [2, 2];
                }
                graph.node(op, &[padded, w], &[y]);
            }
            graph.inputs = vec![x];
            graph.outputs = vec![y];
            graph
        };
        let mut graph = build(0.0, false);
        let expected = run_graph(&graph, false);
        assert_eq!(apply(&mut graph, fuse_pads), 1);
        assert_eq!(graph.nodes.len(), 1);
        let Op::Conv2d { padding, .. } = graph.nodes[0].op else {
            panic!("합성곱이 남아야 한다");
        };
        assert_eq!(padding, Padding::Explicit([[0, 1], [2, 1]]));
        assert_close(&run_graph(&graph, false), &expected, 1e-6);

        // 0이 아닌 값을 패딩하는 PAD, 풀링 앞의 PAD는 그대로 둔다.
        assert_eq!(apply(&mut build(1.0, false), fuse_pads), 0);
        assert_eq!(apply(&mut build(0.0, true), fuse_pads), 0);
    }

    fn node(op_type: &str, inputs: &[&str], output: &str) -> onnx::Node {
        onnx::Node {
            op_type: op_type.into(),
            name: output.into(),
            inputs: inputs.iter().map(|name| name.to_string()).collect(),
            outputs: vec![output.into()],
            ..onnx::Node::default()
        }
    }

    fn ints(name: &str, values: &[i64]) -> (String, onnx::Attribute) {
        (name.into(), onnx::Attribute::Ints(values.to_vec()))
    }

    /// PyTorch가 내보낸 것과 같은 모양의 작은 모델: 합성곱·배치 정규화·
    /// ReLU6, 깊이별 합성곱·배치 정규화, 건너뛰는 덧셈, 평균, 선형 층.
    fn small_onnx() -> onnx::Model {
        let tensor = |shape: &[usize], seed: u32| {
            Tensor::from_vec(noise(shape.iter().product(), seed), shape).unwrap()
        };
        let variance = |count: usize, seed: u32| {
            let data = noise(count, seed).iter().map(|v| v.abs() + 0.1).collect();
            Tensor::from_vec(data, &[count]).unwrap()
        };
        let scalar = |value: f32| Tensor::from_vec(vec![value], &[]).unwrap();
        let mut graph = onnx::Graph {
            name: "small".into(),
            ..onnx::Graph::default()
        };
        graph.inputs.push(onnx::ValueInfo {
            name: "x".into(),
            elem_type: onnx::data_type::FLOAT,
            shape: vec![None, Some(3), Some(6), Some(5)],
        });
        graph.outputs.push(onnx::ValueInfo {
            name: "y".into(),
            ..onnx::ValueInfo::default()
        });
        let mut seed = 0;
        let mut add = |name: &str, tensor: Tensor| graph.initializers.push((name.into(), tensor));
        add("w1", tensor(&[4, 3, 3, 3], 1));
        add("w2", tensor(&[4, 1, 3, 3], 2));
        add("b2", tensor(&[4], 3));
        add("fc.w", tensor(&[2, 4], 4));
        add("fc.b", tensor(&[2], 5));
        for norm in ["n1", "n2"] {
            for part in ["scale", "bias", "mean"] {
                seed += 1;
                add(&format!("{norm}.{part}"), tensor(&[4], 10 + seed));
            }
            add(&format!("{norm}.var"), variance(4, 20 + seed));
        }

        let mut conv1 = node("Conv", &["x", "w1"], "c1");
        conv1.attributes = vec![ints("pads", &[1, 1, 1, 1]), ints("strides", &[2, 1])];
        let norm = |x: &str, norm: &str, output: &str| {
            let names = ["scale", "bias", "mean", "var"].map(|part| format!("{norm}.{part}"));
            let mut inputs = vec![x];
            inputs.extend(names.iter().map(String::as_str));
            node("BatchNormalization", &inputs, output)
        };
        let mut low = node("Constant", &[], "low");
        low.attributes = vec![("value".into(), onnx::Attribute::Tensor(scalar(0.0)))];
        let mut high = node("Constant", &[], "high");
        high.attributes = vec![("value".into(), onnx::Attribute::Tensor(scalar(6.0)))];
        let mut conv2 = node("Conv", &["a1", "w2", "b2"], "c2");
        conv2.attributes = vec![
            ints("pads", &[1, 1, 1, 1]),
            ("group".into(), onnx::Attribute::Int(4)),
        ];
        let mut gemm = node("Gemm", &["flat", "fc.w", "fc.b"], "y");
        gemm.attributes = vec![("transB".into(), onnx::Attribute::Int(1))];
        graph.nodes = vec![
            conv1,
            norm("c1", "n1", "bn1"),
            low,
            high,
            node("Clip", &["bn1", "low", "high"], "a1"),
            conv2,
            norm("c2", "n2", "bn2"),
            node("Add", &["bn2", "a1"], "sum"),
            node("Relu", &["sum"], "a2"),
            node("GlobalAveragePool", &["a2"], "pooled"),
            node("Flatten", &["pooled"], "flat"),
            gemm,
        ];
        onnx::Model::from_graph(graph, 18)
    }

    #[test]
    fn an_imported_model_shrinks_to_its_essentials() {
        let registry = Registry::builtin();
        let mut graph = Graph::from_onnx(&small_onnx()).unwrap();
        graph.infer_shapes(&registry).unwrap();
        // 입력은 NCHW 그대로다.
        assert_eq!(graph.values[graph.inputs[0]].shape, [1, 3, 6, 5]);
        assert_eq!(graph.values[graph.outputs[0]].shape, [1, 2]);
        // 마디 열둘(상수 둘은 값이 된다)이 스물둘이 된다. 축을 바꾸는
        // 마디가 열둘이다.
        assert_eq!(graph.nodes.len(), 22);
        assert_eq!(count(&graph, "TRANSPOSE"), 12);
        let expected = run_graph(&graph, false);

        let applied = optimize(&mut graph, &registry).unwrap();
        let changes = |name: &str| {
            applied
                .iter()
                .find(|pass| pass.name == name)
                .unwrap()
                .changes
        };
        assert_eq!(changes("상수 접기"), 2);
        assert_eq!(changes("배치 정규화 접기"), 2);
        assert_eq!(changes("활성화 함수 붙이기"), 2);
        // 남는 것: 입력의 축 바꾸기, 합성곱 둘, 덧셈, 평균, 선형 층.
        let names: Vec<&str> = graph.nodes.iter().map(|node| node.op.name()).collect();
        let order: Vec<&str> = graph
            .order()
            .unwrap()
            .iter()
            .map(|&index| names[index])
            .collect();
        assert_eq!(
            order,
            [
                "TRANSPOSE",
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
                "ADD",
                "MEAN",
                "FULLY_CONNECTED"
            ]
        );
        assert_close(&run_graph(&graph, false), &expected, 1e-5);
        // 한 번 더 해도 바뀌는 것이 없다.
        let again = optimize(&mut graph, &registry).unwrap();
        assert!(again.iter().all(|pass| pass.changes == 0));

        // 제자리 계산을 켜도 답이 같다.
        assert_eq!(run_graph(&graph, true), run_graph(&graph, false));
    }

    #[test]
    fn importer_rejects_what_it_does_not_understand() {
        let broken = |change: fn(&mut onnx::Model)| {
            let mut model = small_onnx();
            change(&mut model);
            Graph::from_onnx(&model)
        };
        assert!(broken(|model| model.opset = 9).is_err());
        assert!(broken(|model| model.graph.nodes[0].domain = "com.example".into()).is_err());
        assert!(broken(|model| model.graph.nodes[0].inputs[0] = "missing".into()).is_err());
        assert!(broken(|model| model.graph.nodes[0].outputs.push("extra".into())).is_err());
        assert!(broken(|model| model.graph.nodes[1].outputs[0] = "c1".into()).is_err());
        assert!(broken(|model| model.graph.outputs[0].name = "missing".into()).is_err());
        assert!(
            broken(|model| {
                let mode = onnx::Attribute::String("SAME_UPPER".into());
                model.graph.nodes[0]
                    .attributes
                    .push(("auto_pad".into(), mode));
            })
            .is_err()
        );
        assert!(
            broken(|model| {
                let training = onnx::Attribute::Int(1);
                model.graph.nodes[1]
                    .attributes
                    .push(("training_mode".into(), training));
            })
            .is_err()
        );
        // 묶음의 수가 채널의 수와 다른 합성곱. 옮길 때는 입력의 채널 수를
        // 모르므로 형상을 정할 때 드러난다.
        let mut model = small_onnx();
        model.graph.nodes[5].attributes[1] = ("group".into(), onnx::Attribute::Int(2));
        let mut graph = Graph::from_onnx(&model).unwrap();
        assert!(graph.infer_shapes(&Registry::builtin()).is_err());
        // 모르는 연산은 옮기되 실행할 때 거절한다.
        let mut model = small_onnx();
        model.graph.nodes[8].op_type = "Erf".into();
        let mut graph = Graph::from_onnx(&model).unwrap();
        assert!(graph.infer_shapes(&Registry::builtin()).is_err());
    }

    #[test]
    fn quantized_graphs_are_left_alone() {
        let mut graph = conv_then_norm(false, true, Activation::None);
        let x = graph.inputs[0];
        graph.values[x].quantization = Some(crate::graph::Quantization {
            scales: vec![0.5],
            zero_points: vec![0],
            axis: 0,
        });
        let before = graph.nodes.clone();
        let applied = optimize(&mut graph, &Registry::builtin()).unwrap();
        assert!(applied.iter().all(|pass| pass.changes == 0));
        assert_eq!(graph.nodes, before);
    }

    #[test]
    fn in_place_steps_reuse_the_input_slot() {
        // x → ReLU → Clip → (x와 더하기) → y. ReLU는 x를 덮어쓸 수 없다.
        // 덧셈이 x를 또 읽기 때문이다. Clip은 ReLU의 결과를, 덧셈은 Clip의
        // 결과를 덮어쓴다.
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[1000]);
        let [a, b, c, y] = ["a", "b", "c", "y"].map(|name| graph.value(name, &[]));
        graph.node(Op::Relu, &[x], &[a]);
        graph.node(Op::Clip { min: 0.1, max: 0.5 }, &[a], &[b]);
        graph.node(ADD, &[b, x], &[c]);
        // 같은 값을 두 번 읽는 덧셈은 제자리에서 하지 않는다.
        graph.node(ADD, &[c, c], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        let registry = Registry::builtin();
        graph.infer_shapes(&registry).unwrap();
        let plain = Interpreter::new(&graph, &registry).unwrap();
        let shared = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap();
        assert_eq!(plain.in_place_count(), 0);
        assert_eq!(shared.in_place_count(), 2);
        // 따로 두면 x, a, b가 함께 살아 있는 때가 있어 셋의 자리가 든다.
        // 물려받으면 a, b, c가 한 자리이고 y는 x가 쓰던 자리에 들어간다.
        assert_eq!(plain.arena_len(), 3000);
        assert_eq!(shared.arena_len(), 2000);
        assert_eq!(run_graph(&graph, true), run_graph(&graph, false));

        // 덧셈의 둘째 입력을 덮어쓰는 경우와, 형상만 바꾸는 마디.
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[2, 3, 4]);
        let [a, b, y] = ["a", "b", "y"].map(|name| graph.value(name, &[]));
        graph.node(Op::Relu, &[x], &[a]);
        graph.node(
            Op::Mul {
                activation: Activation::Relu6,
            },
            &[x, a],
            &[b],
        );
        graph.node(Op::Flatten { axis: 1 }, &[b], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        graph.infer_shapes(&registry).unwrap();
        let shared = Interpreter::in_place(&graph, &registry, &[], Strategy::BySize).unwrap();
        assert_eq!(shared.in_place_count(), 2);
        assert_eq!(shared.arena_len(), 48);
        assert_eq!(run_graph(&graph, true), run_graph(&graph, false));
    }
}
