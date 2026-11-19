//! 해석기: 그래프를 받아 실행 계획을 세우고, 계획대로 실행한다.

use std::ops::Range;
use std::time::{Duration, Instant};

use badak_core::Tensor;

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::kernels::{Kernel, View, ViewMut};
use crate::memory::{self, Layout, Lifetime, Strategy};
use crate::registry::Registry;

// ANCHOR: types
/// 값이 실행 중에 놓이는 곳.
enum Slot {
    /// 가중치. 모델 파일을 비춘 메모리에 있다.
    Constant(Tensor),
    /// 계산되는 값. 해석기의 자리 안의 한 구간이다.
    Arena(Range<usize>),
    /// 어떤 마디도 쓰지 않는 값.
    Unused,
}

/// 실행 계획의 한 단계.
struct Step {
    /// 연산의 이름. 사람이 읽으려고 둔다.
    name: String,
    kernel: Box<dyn Kernel>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    /// 결과를 이 번호의 입력이 있던 자리에 적는 단계이면 그 번호 (30장).
    in_place: Option<usize>,
}

/// 그래프 하나를 실행하는 해석기.
///
/// 만들 때 모든 것을 정한다. 실행할 순서, 값마다의 형상, 값마다의 자리.
/// [`run`](Self::run)은 정해진 대로 커널을 부르기만 하고, 메모리를 새로
/// 얻지 않는다.
pub struct Interpreter {
    steps: Vec<Step>,
    slots: Vec<Slot>,
    shapes: Vec<Vec<usize>>,
    names: Vec<String>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    /// 계산되는 값마다의 번호와 수명. 계획을 설명할 때 쓴다.
    lifetimes: Vec<(usize, Lifetime)>,
    layout: Layout,
    /// 계산되는 값이 모두 놓이는 자리.
    arena: Vec<f32>,
}
// ANCHOR_END: types

/// 마디 하나의 입력이나 출력이 이보다 많지 않으면 실행 중에 메모리를 얻지
/// 않는다.
const INLINE: usize = 4;

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

// ANCHOR: share
/// 결과를 입력의 자리에 적을 단계를 고르고, 수명을 그에 맞게 합친다.
///
/// 단계의 출력이 입력 하나의 자리를 물려받으려면 넷이 맞아야 한다. 커널이
/// 그 입력의 자리에 적어도 된다고 해야 하고, 그 입력이 이 단계에서 수명이
/// 끝나야 하고(뒤에서 또 읽을 값을 덮어쓸 수는 없다), 원소의 수가 같아야
/// 하고, 이 단계가 그 값을 한 번만 읽어야 한다(`x + x`의 한쪽을 덮어쓰면
/// 다른 쪽도 바뀐다).
///
/// 물려받은 값은 자리를 따로 얻지 않는다. 물려준 값의 수명이 물려받은 값의
/// 수명만큼 길어질 뿐이다. 그래서 물려받기는 줄줄이 이어질 수 있다.
fn share(steps: &mut [Step], lifetimes: &mut Vec<(usize, Lifetime)>, owner: &mut [usize]) {
    let mut spans: Vec<Option<Lifetime>> = vec![None; owner.len()];
    for &(value, lifetime) in lifetimes.iter() {
        spans[value] = Some(lifetime);
    }
    for (index, step) in steps.iter_mut().enumerate() {
        let &[output] = &step.outputs[..] else {
            continue;
        };
        let Some(produced) = spans[output] else {
            continue;
        };
        let target = step
            .inputs
            .iter()
            .enumerate()
            .position(|(position, &input)| {
                let root = owner[input];
                let fits = spans[root]
                    .is_some_and(|span| span.last == index && span.size == produced.size);
                let once = step
                    .inputs
                    .iter()
                    .filter(|&&other| owner[other] == root)
                    .count()
                    == 1;
                fits && once && step.kernel.overwrites(position)
            });
        if let Some(target) = target {
            let root = owner[step.inputs[target]];
            if let Some(span) = &mut spans[root] {
                span.last = produced.last;
            }
            spans[output] = None;
            owner[output] = root;
            step.in_place = Some(target);
        }
    }
    lifetimes.retain_mut(|(value, lifetime)| match spans[*value] {
        Some(span) => {
            *lifetime = span;
            true
        }
        None => false,
    });
}
// ANCHOR_END: share

impl Interpreter {
    /// 그래프에 적힌 입력 형상 그대로, LiteRT와 같은 방법으로 자리를 나누는
    /// 해석기를 만든다.
    pub fn new(graph: &Graph, registry: &Registry) -> Result<Self> {
        Self::with(graph, registry, &[], Strategy::BySize)
    }

    /// [`with`](Self::with)와 같되, 할 수 있는 단계는 결과를 입력의 자리에
    /// 적는다 (30장).
    pub fn in_place(
        graph: &Graph,
        registry: &Registry,
        input_shapes: &[&[usize]],
        strategy: Strategy,
    ) -> Result<Self> {
        Self::build(graph, registry, input_shapes, strategy, true)
    }

    // ANCHOR: build
    /// 해석기를 만든다. `input_shapes`를 주면 그래프의 입력 형상을 그것으로
    /// 바꾼다(한 번에 여러 장을 넣을 때).
    pub fn with(
        graph: &Graph,
        registry: &Registry,
        input_shapes: &[&[usize]],
        strategy: Strategy,
    ) -> Result<Self> {
        Self::build(graph, registry, input_shapes, strategy, false)
    }

    fn build(
        graph: &Graph,
        registry: &Registry,
        input_shapes: &[&[usize]],
        strategy: Strategy,
        in_place: bool,
    ) -> Result<Self> {
        // 1. 실행할 순서를 정하고, 마디마다 커널을 찾는다.
        let order = graph.order()?;
        let mut steps = Vec::with_capacity(order.len());
        for &index in &order {
            let node = &graph.nodes[index];
            steps.push(Step {
                name: node.op.name().to_string(),
                kernel: registry.resolve(&node.op)?,
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                in_place: None,
            });
        }

        // 2. 입력에서 출발해 값마다의 형상을 정한다.
        let mut shapes: Vec<Vec<usize>> = graph.values.iter().map(|v| v.shape.clone()).collect();
        let resized = !input_shapes.is_empty();
        if resized && input_shapes.len() != graph.inputs.len() {
            return Err(invalid(format!(
                "입력이 {}개인 그래프에 형상을 {}개 주었습니다",
                graph.inputs.len(),
                input_shapes.len()
            )));
        }
        for (&input, shape) in graph.inputs.iter().zip(input_shapes) {
            shapes[input] = shape.to_vec();
        }
        for step in &steps {
            let inputs: Vec<&[usize]> = step.inputs.iter().map(|&v| &*shapes[v]).collect();
            let outputs = step.kernel.shapes(&inputs)?;
            if outputs.len() != step.outputs.len() {
                return Err(invalid(format!(
                    "{}의 출력이 {}개여야 하는데 그래프에는 {}개입니다",
                    step.name,
                    outputs.len(),
                    step.outputs.len()
                )));
            }
            for (&value, shape) in step.outputs.iter().zip(outputs) {
                // 파일에 적힌 형상과 계산한 형상이 다르면 파일이나 커널
                // 가운데 하나가 틀린 것이다.
                if !resized && shape != shapes[value] {
                    return Err(invalid(format!(
                        "{}의 출력 {:?}의 형상이 {:?}로 적혀 있지만 계산하면 {shape:?}입니다",
                        step.name, graph.values[value].name, shapes[value]
                    )));
                }
                shapes[value] = shape;
            }
        }

        // 3. 값마다의 수명을 읽는다.
        let mut lifetimes = memory::lifetimes(graph, &order, &shapes)?;
        // 30장: 결과를 입력의 자리에 적을 단계를 고른다. 그런 단계의 출력은
        // 자리를 따로 얻지 않고 `owner`가 가리키는 값의 자리를 쓴다.
        let mut owner: Vec<usize> = (0..graph.values.len()).collect();
        if in_place {
            share(&mut steps, &mut lifetimes, &mut owner);
        }

        // 4. 자리를 나누고, 나눈 결과를 확인한다.
        let only: Vec<Lifetime> = lifetimes.iter().map(|&(_, lifetime)| lifetime).collect();
        let layout = memory::plan(&only, strategy);
        memory::check(&only, &layout)?;

        let mut slots: Vec<Slot> = graph
            .values
            .iter()
            .map(|value| match &value.constant {
                Some(tensor) => Slot::Constant(tensor.contiguous()),
                None => Slot::Unused,
            })
            .collect();
        for (&(value, lifetime), &offset) in lifetimes.iter().zip(&layout.offsets) {
            slots[value] = Slot::Arena(offset..offset + lifetime.size);
        }
        for (value, &owner) in owner.iter().enumerate() {
            if owner != value
                && let Slot::Arena(range) = &slots[owner]
            {
                slots[value] = Slot::Arena(range.clone());
            }
        }
        for step in &steps {
            for &input in &step.inputs {
                if let Slot::Constant(tensor) = &slots[input] {
                    // 커널은 f32 슬라이스만 받는다. 여기서 한 번 확인해 둔다.
                    tensor.as_slice::<f32>()?;
                }
            }
        }

        let interpreter = Self {
            steps,
            slots,
            shapes,
            names: graph.values.iter().map(|v| v.name.clone()).collect(),
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            lifetimes,
            arena: vec![0.0; layout.total],
            layout,
        };
        interpreter.check_steps()?;
        Ok(interpreter)
    }
    // ANCHOR_END: build

    // ANCHOR: check_steps
    /// 단계마다, 쓰는 자리가 그 단계에서 읽는 자리나 쓰는 다른 자리와 겹치지
    /// 않는지 확인한다. [`run`](Self::run)의 `unsafe`가 기대는 사실이다.
    fn check_steps(&self) -> Result<()> {
        let range = |value: usize| match &self.slots[value] {
            Slot::Arena(range) => Some(range.clone()),
            _ => None,
        };
        let overlap = |a: &Range<usize>, b: &Range<usize>| a.start < b.end && b.start < a.end;
        for step in &self.steps {
            for (index, &output) in step.outputs.iter().enumerate() {
                let written = range(output)
                    .filter(|range| range.end <= self.arena.len())
                    .ok_or_else(|| invalid(format!("{}의 출력에 자리가 없습니다", step.name)))?;
                let others = step.inputs.iter().chain(&step.outputs[..index]);
                for (position, &other) in others.enumerate() {
                    // 30장: 제자리 단계는 고른 입력 하나와 정확히 같은 자리에
                    // 쓴다. 그 입력은 읽는 창을 만들지 않으므로 겹쳐도 된다.
                    // 나머지와는 여전히 겹치면 안 된다.
                    if step.in_place == Some(position) && step.outputs.len() == 1 {
                        if range(other) != Some(written.clone()) {
                            return Err(invalid(format!(
                                "{}이 제자리에서 쓰는 자리가 입력의 자리와 다릅니다",
                                step.name
                            )));
                        }
                        continue;
                    }
                    if range(other).is_some_and(|range| overlap(&written, &range)) {
                        return Err(invalid(format!(
                            "{}이 읽는 자리와 쓰는 자리가 겹칩니다",
                            step.name
                        )));
                    }
                }
            }
            for &input in &step.inputs {
                let inside = match &self.slots[input] {
                    Slot::Constant(_) => true,
                    Slot::Arena(range) => range.end <= self.arena.len(),
                    Slot::Unused => false,
                };
                if !inside {
                    return Err(invalid(format!("{}의 입력에 자리가 없습니다", step.name)));
                }
            }
        }
        Ok(())
    }
    // ANCHOR_END: check_steps

    /// `index`번째 입력의 형상.
    pub fn input_shape(&self, index: usize) -> &[usize] {
        &self.shapes[self.inputs[index]]
    }

    /// `index`번째 출력의 형상.
    pub fn output_shape(&self, index: usize) -> &[usize] {
        &self.shapes[self.outputs[index]]
    }

    // ANCHOR: io
    /// `index`번째 입력이 놓일 자리. 여기에 값을 적고 [`run`](Self::run)을
    /// 부른다.
    ///
    /// 입력의 자리도 다른 값과 함께 쓴다. 실행이 끝나면 내용이 바뀌어 있을
    /// 수 있으므로 실행할 때마다 다시 적어야 한다.
    pub fn input_mut(&mut self, index: usize) -> &mut [f32] {
        match &self.slots[self.inputs[index]] {
            Slot::Arena(range) => &mut self.arena[range.clone()],
            _ => unreachable!("입력은 언제나 계산되는 값이다"),
        }
    }

    /// `index`번째 출력. [`run`](Self::run)이 끝난 뒤에 읽는다.
    pub fn output(&self, index: usize) -> &[f32] {
        match &self.slots[self.outputs[index]] {
            Slot::Arena(range) => &self.arena[range.clone()],
            Slot::Constant(tensor) => tensor.as_slice().expect("만들 때 확인했다"),
            Slot::Unused => unreachable!("출력에는 언제나 자리가 있다"),
        }
    }
    // ANCHOR_END: io

    // ANCHOR: run
    /// 계획대로 실행한다.
    pub fn run(&mut self) -> Result<()> {
        self.run_steps(|_, _| {})
    }

    /// 실행하면서 단계마다 걸린 시간을 `times`에 더한다.
    pub fn run_timed(&mut self, times: &mut [Duration]) -> Result<()> {
        let mut started = Instant::now();
        self.run_steps(|index, _| {
            let now = Instant::now();
            times[index] += now - started;
            started = now;
        })
    }

    fn run_steps(&mut self, mut after: impl FnMut(usize, &str)) -> Result<()> {
        let arena = self.arena.as_mut_ptr();
        // 입력과 출력이 몇 개 안 되는 보통의 마디는 스택의 배열에 담는다.
        // 그보다 많을 때만 벡터를 쓴다.
        let mut many_inputs: Vec<View<'_>> = Vec::new();
        let mut many_outputs: Vec<ViewMut<'_>> = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            let mut few_inputs = [View::EMPTY; INLINE];
            let mut few_outputs: [ViewMut<'_>; INLINE] = std::array::from_fn(|_| ViewMut::empty());
            // SAFETY: `check_steps`가 확인한 대로, 이 단계가 읽는 구간과 쓰는
            // 구간은 모두 `arena` 안에 있고, 쓰는 구간은 이 단계의 다른 어떤
            // 구간과도 겹치지 않는다. 읽는 구간끼리는 겹쳐도 된다. 여기서 만든
            // 참조들은 이 반복이 끝나면 사라지고, 그동안 `arena`를 가리키는
            // 다른 참조는 만들지 않는다.
            let read = |value: usize| unsafe { self.read(arena, value) };
            let write = |value: usize| unsafe { self.write(arena, value) };
            // 제자리 단계에서 덮어쓸 입력은 읽는 창을 만들지 않는다. 같은
            // 구간을 쓰는 창이 곧 생기기 때문이다.
            let read = |(position, &value): (usize, &usize)| {
                if step.in_place == Some(position) {
                    View::EMPTY
                } else {
                    read(value)
                }
            };
            let inputs: &[View<'_>] = if step.inputs.len() <= INLINE {
                for (slot, input) in few_inputs.iter_mut().zip(step.inputs.iter().enumerate()) {
                    *slot = read(input);
                }
                &few_inputs[..step.inputs.len()]
            } else {
                many_inputs.clear();
                many_inputs.extend(step.inputs.iter().enumerate().map(read));
                &many_inputs
            };
            let outputs: &mut [ViewMut<'_>] = if step.outputs.len() <= INLINE {
                for (slot, &value) in few_outputs.iter_mut().zip(&step.outputs) {
                    *slot = write(value);
                }
                &mut few_outputs[..step.outputs.len()]
            } else {
                many_outputs.clear();
                many_outputs.extend(step.outputs.iter().map(|&value| write(value)));
                &mut many_outputs
            };
            match step.in_place {
                Some(target) => step.kernel.run_in_place(inputs, target, &mut outputs[0])?,
                None => step.kernel.run(inputs, outputs)?,
            }
            after(index, &step.name);
        }
        Ok(())
    }

    /// 값을 읽는 창을 만든다.
    ///
    /// # Safety
    ///
    /// `arena`는 `self.arena`의 시작이어야 한다. 돌려준 창이 살아 있는 동안
    /// 같은 구간에 쓰는 참조가 있으면 안 된다.
    unsafe fn read(&self, arena: *mut f32, value: usize) -> View<'_> {
        let data = match &self.slots[value] {
            Slot::Constant(tensor) => tensor.as_slice().expect("만들 때 확인했다"),
            // SAFETY: 구간은 `arena` 안에 있다(`check_steps`). 쓰는 참조가
            // 없다는 것은 부르는 쪽이 지킨다.
            Slot::Arena(range) => unsafe {
                std::slice::from_raw_parts(arena.add(range.start), range.len())
            },
            Slot::Unused => unreachable!("check_steps가 확인했다"),
        };
        View {
            shape: &self.shapes[value],
            data,
        }
    }

    /// 값을 적을 창을 만든다.
    ///
    /// # Safety
    ///
    /// `arena`는 `self.arena`의 시작이어야 한다. 돌려준 창이 살아 있는 동안
    /// 같은 구간을 가리키는 다른 참조가 있으면 안 된다.
    unsafe fn write(&self, arena: *mut f32, value: usize) -> ViewMut<'_> {
        let Slot::Arena(range) = &self.slots[value] else {
            unreachable!("check_steps가 확인했다")
        };
        // SAFETY: 구간은 `arena` 안에 있다(`check_steps`). 다른 참조가 없다는
        // 것은 부르는 쪽이 지킨다.
        let data = unsafe { std::slice::from_raw_parts_mut(arena.add(range.start), range.len()) };
        ViewMut {
            shape: &self.shapes[value],
            data,
        }
    }
    // ANCHOR_END: run

    /// 단계의 수.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// 계획을 설명한다. 단계마다 연산의 이름, 입력들과 출력들의 이름과 형상.
    pub fn steps(&self) -> impl Iterator<Item = StepInfo<'_>> {
        let describe = |values: &[usize]| -> Vec<(&str, &[usize])> {
            values
                .iter()
                .map(|&value| (&*self.names[value], &*self.shapes[value]))
                .collect()
        };
        self.steps.iter().map(move |step| StepInfo {
            name: &step.name,
            inputs: describe(&step.inputs),
            outputs: describe(&step.outputs),
        })
    }

    /// 계산되는 값마다의 이름, 수명, 자리의 시작.
    pub fn placements(&self) -> impl Iterator<Item = (&str, Lifetime, usize)> {
        self.lifetimes
            .iter()
            .zip(&self.layout.offsets)
            .map(|(&(value, lifetime), &offset)| (&*self.names[value], lifetime, offset))
    }

    /// 계산되는 값들의 수명만.
    pub fn lifetimes(&self) -> Vec<Lifetime> {
        self.lifetimes
            .iter()
            .map(|&(_, lifetime)| lifetime)
            .collect()
    }

    /// 결과를 입력의 자리에 적는 단계의 수 (30장).
    pub fn in_place_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.in_place.is_some())
            .count()
    }

    /// 계산되는 값이 모두 놓이는 자리의 크기(원소 수).
    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    /// 자리 전체를 `value`로 채운다. 커널이 남아 있던 값에 기대지 않는지
    /// 시험할 때 쓴다.
    pub fn scribble(&mut self, value: f32) {
        self.arena.fill(value);
    }
}

/// [`Interpreter::steps`]가 돌려주는 단계 하나의 설명.
#[derive(Debug)]
pub struct StepInfo<'a> {
    /// 연산의 이름.
    pub name: &'a str,
    /// 입력들의 이름과 형상.
    pub inputs: Vec<(&'a str, &'a [usize])>,
    /// 출력들의 이름과 형상.
    pub outputs: Vec<(&'a str, &'a [usize])>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Activation, Op};

    /// `y = softmax(h + relu(W2 relu(W1 h)))`, `h = relu(W0 x + b0)`인 작은
    /// 잔차 모델.
    fn residual() -> Graph {
        let mut graph = Graph::new("잔차");
        let tensor = |values: Vec<f32>, shape: &[usize]| Tensor::from_vec(values, shape).unwrap();
        let fc = |activation| Op::FullyConnected {
            activation,
            keep_num_dims: false,
        };
        let x = graph.value("x", &[1, 2]);
        let w0 = graph.constant("w0", tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]));
        let b0 = graph.constant("b0", tensor(vec![0.0, 0.5, -1.0], &[3]));
        let h = graph.value("h", &[1, 3]);
        let w1 = graph.constant("w1", tensor(vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0], &[2, 3]));
        let a = graph.value("a", &[1, 2]);
        let w2 = graph.constant("w2", tensor(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 2]));
        let b = graph.value("b", &[1, 3]);
        let sum = graph.value("sum", &[1, 3]);
        let y = graph.value("y", &[1, 3]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        graph.node(fc(Activation::Relu), &[x, w0, b0], &[h]);
        graph.node(fc(Activation::Relu), &[h, w1], &[a]);
        graph.node(fc(Activation::None), &[a, w2], &[b]);
        graph.node(
            Op::Add {
                activation: Activation::Relu,
            },
            &[h, b],
            &[sum],
        );
        graph.node(Op::Softmax { beta: 1.0 }, &[sum], &[y]);
        graph
    }

    /// 같은 계산을 손으로 한 것.
    fn expected(x: [f32; 2]) -> Vec<f32> {
        let relu = |v: f32| v.max(0.0);
        let h = [relu(x[0]), relu(x[1] + 0.5), relu(x[0] + x[1] - 1.0)];
        let a = [relu(h[0] - h[1]), relu(h[1] + h[2])];
        let b = [a[0], a[1], 0.5 * a[0] + 0.5 * a[1]];
        let sum: Vec<f32> = h.iter().zip(b).map(|(h, b)| relu(h + b)).collect();
        badak_core::prob::softmax(&sum)
    }

    /// 두 결과가 같은지 본다. 양쪽 다 `exp`를 거치는데, Miri는 `exp`의
    /// 마지막 자리를 일부러 흔들므로 비트까지 같다고 할 수는 없다.
    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn runs_a_residual_graph() {
        let graph = residual();
        for strategy in [Strategy::Separate, Strategy::InOrder, Strategy::BySize] {
            let mut interpreter =
                Interpreter::with(&graph, &Registry::builtin(), &[], strategy).unwrap();
            for x in [[1.0, 2.0], [-3.0, 0.25], [0.5, 0.5]] {
                // 앞의 실행이 남긴 값에 기대면 여기서 드러난다.
                interpreter.scribble(f32::NAN);
                interpreter.input_mut(0).copy_from_slice(&x);
                interpreter.run().unwrap();
                assert_close(interpreter.output(0), &expected(x));
            }
        }
    }

    #[test]
    fn shares_memory_between_values_that_never_meet() {
        let graph = residual();
        let registry = Registry::builtin();
        let separate = Interpreter::with(&graph, &registry, &[], Strategy::Separate).unwrap();
        let planned = Interpreter::new(&graph, &registry).unwrap();
        // x 2, h 3, a 2, b 3, sum 3, y 3
        assert_eq!(separate.arena_len(), 16);
        assert_eq!(
            planned.arena_len(),
            memory::lower_bound(&planned.lifetimes())
        );
        assert!(planned.arena_len() < separate.arena_len());
        // h는 덧셈(3번 단계)까지 살아 있어야 한다.
        let (_, lifetime, _) = planned
            .placements()
            .find(|(name, ..)| *name == "h")
            .unwrap();
        assert_eq!((lifetime.first, lifetime.last), (0, 3));
    }

    #[test]
    fn resizes_the_batch() {
        let graph = residual();
        let mut interpreter =
            Interpreter::with(&graph, &Registry::builtin(), &[&[3, 2]], Strategy::BySize).unwrap();
        assert_eq!(interpreter.output_shape(0), [3, 3]);
        let batch = [[1.0, 2.0], [-3.0, 0.25], [0.5, 0.5]];
        interpreter
            .input_mut(0)
            .copy_from_slice(batch.as_flattened());
        interpreter.run().unwrap();
        let rows: Vec<f32> = batch.into_iter().flat_map(expected).collect();
        assert_close(interpreter.output(0), &rows);
        // 벡터의 길이를 바꾸면 가중치와 맞지 않는다.
        assert!(
            Interpreter::with(&graph, &Registry::builtin(), &[&[1, 3]], Strategy::BySize).is_err()
        );
    }

    #[test]
    fn follows_dependency_order_not_file_order() {
        let mut graph = residual();
        graph.nodes.reverse();
        let mut interpreter = Interpreter::new(&graph, &Registry::builtin()).unwrap();
        interpreter.input_mut(0).copy_from_slice(&[1.0, 2.0]);
        interpreter.run().unwrap();
        assert_close(interpreter.output(0), &expected([1.0, 2.0]));
        let names: Vec<&str> = interpreter.steps().map(|step| step.name).collect();
        assert_eq!(names.last(), Some(&"SOFTMAX"));
    }

    #[test]
    fn rejects_shapes_that_disagree_with_the_file() {
        let mut graph = residual();
        graph.values[3].shape = vec![1, 4];
        assert!(Interpreter::new(&graph, &Registry::builtin()).is_err());
    }

    #[test]
    fn reports_missing_kernels_before_running() {
        let mut graph = residual();
        graph.nodes[4].op = Op::Custom { name: "ERF".into() };
        let Err(Error::Unsupported(message)) =
            Interpreter::new(&graph, &Registry::builtin()).map(|_| ())
        else {
            panic!("모르는 연산은 만들 때 오류여야 한다");
        };
        assert!(message.contains("ERF"));
    }

    #[test]
    fn times_every_step() {
        let mut interpreter = Interpreter::new(&residual(), &Registry::builtin()).unwrap();
        let mut times = vec![Duration::ZERO; interpreter.step_count()];
        interpreter.input_mut(0).copy_from_slice(&[1.0, 2.0]);
        interpreter.run_timed(&mut times).unwrap();
        assert_eq!(times.len(), 5);
        assert_close(interpreter.output(0), &expected([1.0, 2.0]));
    }
}
