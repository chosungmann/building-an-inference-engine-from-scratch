//! 8비트 정수로 계산하는 해석기 (29장).
//!
//! 값이 `f32`가 아니라 `i8`이다. 값마다 눈금과 영점이 있고, 실수 값은
//! `눈금 × (정수 − 영점)`이다. 계산은 처음부터 끝까지 정수로 한다. 눈금을
//! 곱하는 일도 정수의 곱셈과 자리 옮기기로 한다.
//!
//! 그래프, 실행 순서, 형상, 수명, 메모리 계획은 26장의 것을 그대로 쓴다.
//! 달라지는 것은 자리의 원소와 커널이다. 이 모듈의 커널은 LiteRT의 기준
//! 구현과 **같은 비트**를 내도록 짰다.

use std::ops::Range;

use badak_core::{DType, Tensor, threads};

use crate::conv::Axis;
use crate::error::{Error, Result};
use crate::graph::{Activation, Graph, Op, Padding, Quantization};
use crate::memory::{self, Layout, Lifetime, Strategy};
use crate::registry::Registry;

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

// ANCHOR: multiplier
/// 실수 배율을 "31비트 정수 × 2의 거듭제곱"으로 바꾼다.
///
/// 배율 `m`을 `q × 2^shift`로 쓴다. `q`는 0.5 이상 1 미만이고, 그것에
/// `2^31`을 곱해 반올림한 정수가 돌려주는 첫 값이다. LiteRT의
/// `QuantizeMultiplier`와 같다.
pub fn quantize_multiplier(real: f64) -> (i32, i32) {
    if real == 0.0 {
        return (0, 0);
    }
    let (mut q, mut shift) = (real, 0);
    while q >= 1.0 {
        q /= 2.0;
        shift += 1;
    }
    while q < 0.5 {
        q *= 2.0;
        shift -= 1;
    }
    let mut fixed = (q * (1_i64 << 31) as f64).round() as i64;
    if fixed == 1 << 31 {
        fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        return (0, 0);
    }
    (fixed as i32, shift)
}

/// 두 정수를 곱해 위쪽 32비트를 반올림해서 얻는다. `a × b / 2^31`이다.
fn rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let product = i64::from(a) * i64::from(b);
    let nudge = if product >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    // 0을 향해 버리는 나눗셈이다. 자리 옮기기(>>)와 다르다.
    ((product + nudge) / (1_i64 << 31)) as i32
}

/// 2의 거듭제곱으로 나누고 반올림한다. 딱 절반이면 0에서 먼 쪽으로 간다.
fn rounding_divide_by_power_of_two(x: i32, exponent: i32) -> i32 {
    let mask = ((1_i64 << exponent) - 1) as i32;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// 정수에 [`quantize_multiplier`]가 만든 배율을 곱한다. 64비트로 곱하고,
/// 절반을 더한 뒤, 한 번에 자리를 옮긴다. 반올림이 한 번이다.
pub fn multiply(x: i32, multiplier: i32, shift: i32) -> i32 {
    let total = 31 - shift;
    let round = 1_i64 << (total - 1);
    ((i64::from(x) * i64::from(multiplier) + round) >> total) as i32
}

/// [`multiply`]와 같은 일을 두 단계로 한다. 먼저 위쪽 32비트를 반올림해
/// 얻고, 그것을 다시 2의 거듭제곱으로 나누며 반올림한다. 반올림이 두 번이라
/// 드물게 1이 다르다. 32비트 레지스터만으로 할 수 있어서 벡터 명령으로 짠
/// 커널들이 이렇게 한다.
pub fn multiply_twice_rounded(x: i32, multiplier: i32, shift: i32) -> i32 {
    let (left, right) = (shift.max(0), (-shift).max(0));
    rounding_divide_by_power_of_two(
        rounding_doubling_high_mul(x.wrapping_mul(1 << left), multiplier),
        right,
    )
}
// ANCHOR_END: multiplier

/// [`multiply_twice_rounded`]와 같되, 두 번의 반올림이 모두 "절반이면 큰
/// 쪽으로"다. ARM의 벡터 명령 둘(`sqrdmulh`, `srshl`)이 하는 그대로이고,
/// 음수가 딱 절반일 때만 위의 것과 1이 다르다.
pub fn multiply_half_up(x: i32, multiplier: i32, shift: i32) -> i32 {
    let (left, right) = (shift.max(0), (-shift).max(0));
    let high = (i64::from(x.wrapping_mul(1 << left)) * i64::from(multiplier) + (1 << 30)) >> 31;
    if right == 0 {
        high as i32
    } else {
        ((high + (1 << (right - 1))) >> right) as i32
    }
}

// ANCHOR: rounding
/// 배율을 곱하고 반올림하는 세 가지 방법. 수학으로는 같은 계산이고 결과가
/// 드물게 1씩 다르다. LiteRT는 연산마다 다른 것을 쓴다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// 64비트로 곱하고 한 번 반올림한다([`multiply`]).
    Once,
    /// 32비트 안에서 두 번 반올림한다([`multiply_twice_rounded`]).
    Twice,
    /// 두 번 반올림하되 절반을 언제나 큰 쪽으로 보낸다
    /// ([`multiply_half_up`]).
    TwiceHalfUp,
}

impl Rounding {
    /// 정수에 배율을 곱한다.
    #[inline]
    pub fn multiply(self, x: i32, multiplier: i32, shift: i32) -> i32 {
        match self {
            Self::Once => multiply(x, multiplier, shift),
            Self::Twice => multiply_twice_rounded(x, multiplier, shift),
            Self::TwiceHalfUp => multiply_half_up(x, multiplier, shift),
        }
    }
}

impl Weighted {
    /// LiteRT의 커널이 이 연산에 쓰는 반올림. 소스에 적힌 것이 아니라 층마다
    /// 출력을 견줘서 찾은 것이다(29.7절).
    fn rounding(self) -> Rounding {
        match self {
            Self::Conv => Rounding::Once,
            Self::Depthwise => Rounding::Twice,
            Self::Dense => Rounding::TwiceHalfUp,
        }
    }
}
// ANCHOR_END: rounding

/// 이어 붙인 활성화 함수가 정하는 정수의 범위.
fn activation_range(activation: Activation, output: &Quantization) -> Result<(i32, i32)> {
    let (scale, zero) = (output.scales[0], output.zero_points[0]);
    let level = |real: f32| zero + (real / scale).round() as i32;
    let (low, high) = match activation {
        Activation::None => (-128, 127),
        Activation::Relu => (level(0.0), 127),
        Activation::Relu6 => (level(0.0), level(6.0)),
        Activation::ReluN1To1 => (level(-1.0), level(1.0)),
        Activation::Tanh => return Err(Error::Unsupported("정수 연산에 이어 붙인 tanh".into())),
        Activation::Silu => return Err(Error::Unsupported("정수 연산에 이어 붙인 SiLU".into())),
    };
    Ok((low.max(-128), high.min(127)))
}

// ANCHOR: prepared
/// 마디마다 실행 전에 계산해 두는 것들.
enum Prepared {
    /// 합성곱, 깊이별 합성곱, 선형 층. 출력 채널마다 배율이 하나씩 있다.
    Weighted {
        kind: Weighted,
        rows: Axis,
        columns: Axis,
        /// 출력 채널마다 (배율, 자리 옮김).
        multipliers: Vec<(i32, i32)>,
        /// 출력 채널마다 `편향 − 입력의 영점 × Σ가중치`.
        bias: Vec<i32>,
        input_zero: i32,
        output_zero: i32,
        range: (i32, i32),
    },
    Add {
        offsets: [i32; 2],
        multipliers: [(i32, i32); 2],
        output: (i32, i32),
        output_zero: i32,
        range: (i32, i32),
    },
    Mean {
        /// `입력의 눈금 / (출력의 눈금 × 원소 수)`.
        multiplier: (i32, i32),
        /// 합에서 뺄 값. `입력의 영점 × 원소 수`.
        offset: i32,
        /// 출력의 영점.
        bias: i32,
    },
    Pad {
        amounts: Vec<[usize; 2]>,
        value: i8,
    },
    Transpose {
        perm: Vec<usize>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Weighted {
    Conv,
    Depthwise,
    Dense,
}
// ANCHOR_END: prepared

/// 값이 실행 중에 놓이는 곳.
enum Slot {
    Constant(Tensor),
    Arena(Range<usize>),
    Unused,
}

struct Step {
    name: String,
    prepared: Prepared,
    inputs: Vec<usize>,
    output: usize,
}

/// 8비트 정수 그래프를 실행하는 해석기.
pub struct QuantizedInterpreter {
    steps: Vec<Step>,
    slots: Vec<Slot>,
    shapes: Vec<Vec<usize>>,
    quantizations: Vec<Option<Quantization>>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    lifetimes: Vec<Lifetime>,
    arena: Vec<i8>,
}

/// 가중치 텐서의 바이트를 부호 있는 정수로 읽는다.
fn signed(tensor: &Tensor) -> Result<&[i8]> {
    let bytes = tensor.as_slice::<u8>()?;
    // SAFETY: `u8`과 `i8`은 크기와 정렬이 같고 모든 비트 무늬가 두 타입
    // 모두에서 유효하다. 수명은 `bytes`의 것을 물려받는다.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) })
}

fn single(quantization: &Quantization, what: &str) -> Result<(f32, i32)> {
    match (
        quantization.scales.as_slice(),
        quantization.zero_points.as_slice(),
    ) {
        (&[scale], &[zero]) => Ok((scale, zero)),
        _ => Err(invalid(format!("{what}의 눈금이 하나가 아닙니다"))),
    }
}

impl QuantizedInterpreter {
    // ANCHOR: build
    /// 해석기를 만든다. 순서, 형상, 수명, 자리는 26장과 같은 방법으로 정하고,
    /// 마디마다 정수 배율을 미리 계산한다.
    pub fn new(graph: &Graph) -> Result<Self> {
        let order = graph.order()?;
        // 형상은 자료형과 상관이 없다. 26~27장의 커널에게 묻는다.
        let registry = Registry::builtin();
        let mut shapes: Vec<Vec<usize>> = graph.values.iter().map(|v| v.shape.clone()).collect();
        let quantization = |value: usize| -> Result<&Quantization> {
            graph.values[value].quantization.as_ref().ok_or_else(|| {
                invalid(format!(
                    "값 {:?}에 눈금이 없습니다",
                    graph.values[value].name
                ))
            })
        };
        let mut steps = Vec::with_capacity(order.len());
        for &index in &order {
            let node = &graph.nodes[index];
            let &[output] = node.outputs.as_slice() else {
                return Err(Error::Unsupported("출력이 여럿인 정수 연산".into()));
            };
            let input_shapes: Vec<&[usize]> = node.inputs.iter().map(|&v| &*shapes[v]).collect();
            let computed = registry.resolve(&node.op)?.shapes(&input_shapes)?;
            if computed[0] != shapes[output] {
                return Err(invalid(format!(
                    "{}의 출력 형상이 파일과 다릅니다",
                    node.op.name()
                )));
            }
            shapes[output] = computed.into_iter().next().expect("출력이 하나다");
            let prepared = prepare(
                graph,
                node.op.clone(),
                &node.inputs,
                output,
                &shapes,
                &quantization,
            )?;
            steps.push(Step {
                name: node.op.name().to_string(),
                prepared,
                inputs: node.inputs.clone(),
                output,
            });
        }

        let placed = memory::lifetimes(graph, &order, &shapes)?;
        let lifetimes: Vec<Lifetime> = placed.iter().map(|&(_, lifetime)| lifetime).collect();
        let layout: Layout = memory::plan(&lifetimes, Strategy::BySize);
        memory::check(&lifetimes, &layout)?;
        let mut slots: Vec<Slot> = graph
            .values
            .iter()
            .map(|value| match &value.constant {
                Some(tensor) => Slot::Constant(tensor.contiguous()),
                None => Slot::Unused,
            })
            .collect();
        for (&(value, lifetime), &offset) in placed.iter().zip(&layout.offsets) {
            slots[value] = Slot::Arena(offset..offset + lifetime.size);
        }
        let interpreter = Self {
            steps,
            slots,
            shapes,
            quantizations: graph
                .values
                .iter()
                .map(|v| v.quantization.clone())
                .collect(),
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            lifetimes,
            arena: vec![0; layout.total],
        };
        interpreter.check_steps()?;
        Ok(interpreter)
    }
    // ANCHOR_END: build

    /// 단계마다 쓰는 자리가 읽는 자리와 겹치지 않는지 확인한다. `run`의
    /// `unsafe`가 기대는 사실이다(26.10절).
    fn check_steps(&self) -> Result<()> {
        for step in &self.steps {
            let Slot::Arena(written) = &self.slots[step.output] else {
                return Err(invalid(format!("{}의 출력에 자리가 없습니다", step.name)));
            };
            if written.end > self.arena.len() {
                return Err(invalid("출력의 자리가 전체를 벗어납니다"));
            }
            let Some(&first) = step.inputs.first() else {
                return Err(invalid(format!("{}에 입력이 없습니다", step.name)));
            };
            let operands = if matches!(step.prepared, Prepared::Add { .. }) {
                &step.inputs[..]
            } else {
                std::slice::from_ref(&first)
            };
            for &input in operands {
                let Slot::Arena(read) = &self.slots[input] else {
                    return Err(invalid(format!("{}의 입력에 자리가 없습니다", step.name)));
                };
                if read.end > self.arena.len()
                    || (read.start < written.end && written.start < read.end)
                {
                    return Err(invalid(format!(
                        "{}이 읽는 자리와 쓰는 자리가 겹칩니다",
                        step.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// `index`번째 입력의 자리와, 그 값의 눈금과 영점.
    pub fn input_mut(&mut self, index: usize) -> &mut [i8] {
        match &self.slots[self.inputs[index]] {
            Slot::Arena(range) => &mut self.arena[range.clone()],
            _ => unreachable!("입력은 언제나 계산되는 값이다"),
        }
    }

    /// `index`번째 입력의 눈금과 영점.
    pub fn input_quantization(&self, index: usize) -> (f32, i32) {
        let quantization = self.quantizations[self.inputs[index]].as_ref();
        quantization.map_or((1.0, 0), |q| (q.scales[0], q.zero_points[0]))
    }

    /// `index`번째 출력.
    pub fn output(&self, index: usize) -> &[i8] {
        match &self.slots[self.outputs[index]] {
            Slot::Arena(range) => &self.arena[range.clone()],
            _ => unreachable!("출력은 언제나 계산되는 값이다"),
        }
    }

    /// `index`번째 출력의 눈금과 영점.
    pub fn output_quantization(&self, index: usize) -> (f32, i32) {
        let quantization = self.quantizations[self.outputs[index]].as_ref();
        quantization.map_or((1.0, 0), |q| (q.scales[0], q.zero_points[0]))
    }

    /// 계산되는 값이 모두 놓이는 자리의 크기(바이트).
    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    /// 계산되는 값들의 수명.
    pub fn lifetimes(&self) -> &[Lifetime] {
        &self.lifetimes
    }

    /// 단계의 수.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// 단계마다, 출력 값의 번호(파일의 텐서 번호).
    pub fn step_outputs(&self) -> Vec<usize> {
        self.steps.iter().map(|step| step.output).collect()
    }

    /// 단계마다, 출력의 원소 수.
    pub fn step_sizes(&self) -> Vec<usize> {
        self.steps
            .iter()
            .map(|step| self.shapes[step.output].iter().product())
            .collect()
    }

    /// 단계마다의 연산 이름.
    pub fn step_names(&self) -> impl Iterator<Item = &str> {
        self.steps.iter().map(|step| &*step.name)
    }

    /// 계획대로 실행한다.
    pub fn run(&mut self) -> Result<()> {
        self.run_steps(|_| {})
    }

    /// 실행하면서 단계마다 걸린 시간을 `times`에 더한다.
    pub fn run_timed(&mut self, times: &mut [std::time::Duration]) -> Result<()> {
        let mut started = std::time::Instant::now();
        self.run_steps(|index| {
            let now = std::time::Instant::now();
            times[index] += now - started;
            started = now;
        })
    }

    /// 실행하면서 단계마다 `visit(단계 번호, 출력)`을 부른다. 층마다 정답과
    /// 견줄 때 쓴다.
    pub fn run_visiting(&mut self, mut visit: impl FnMut(usize, &[i8])) -> Result<()> {
        let outputs: Vec<Range<usize>> = self
            .steps
            .iter()
            .map(|step| match &self.slots[step.output] {
                Slot::Arena(range) => range.clone(),
                _ => unreachable!("check_steps가 확인했다"),
            })
            .collect();
        let arena = self.arena.as_ptr();
        self.run_steps(|index| {
            let range = &outputs[index];
            // SAFETY: 구간은 자리 안에 있고(`check_steps`), 이 순간 그 구간에
            // 쓰는 참조는 없다. 단계가 끝난 뒤에 불린다.
            visit(index, unsafe {
                std::slice::from_raw_parts(arena.add(range.start), range.len())
            });
        })
    }

    fn run_steps(&mut self, mut after: impl FnMut(usize)) -> Result<()> {
        let arena = self.arena.as_mut_ptr();
        for (index, step) in self.steps.iter().enumerate() {
            let read = |value: usize| -> &[i8] {
                match &self.slots[value] {
                    // SAFETY: `check_steps`가 확인한 대로 구간은 자리 안에 있고
                    // 이 단계가 쓰는 구간과 겹치지 않는다.
                    Slot::Arena(range) => unsafe {
                        std::slice::from_raw_parts(arena.add(range.start), range.len())
                    },
                    _ => &[],
                }
            };
            let constant = |value: usize| match &self.slots[value] {
                Slot::Constant(tensor) => Ok(tensor),
                _ => Err(invalid("가중치가 파일에 없는 정수 연산")),
            };
            let Slot::Arena(range) = &self.slots[step.output] else {
                unreachable!("check_steps가 확인했다")
            };
            // SAFETY: 위와 같다. 이 구간을 가리키는 다른 참조가 없다.
            let out =
                unsafe { std::slice::from_raw_parts_mut(arena.add(range.start), range.len()) };
            let x = read(step.inputs[0]);
            let (in_shape, out_shape) = (&self.shapes[step.inputs[0]], &self.shapes[step.output]);
            match &step.prepared {
                Prepared::Weighted {
                    kind,
                    rows,
                    columns,
                    multipliers,
                    bias,
                    input_zero,
                    output_zero,
                    range,
                } => {
                    let w = signed(constant(step.inputs[1])?)?;
                    let requantize = Requantize {
                        rounding: kind.rounding(),
                        multipliers,
                        bias,
                        output_zero: *output_zero,
                        range: *range,
                    };
                    match kind {
                        Weighted::Depthwise => {
                            depthwise(
                                x,
                                in_shape,
                                w,
                                rows,
                                columns,
                                *input_zero,
                                &requantize,
                                out,
                                out_shape,
                            );
                        }
                        Weighted::Conv | Weighted::Dense => {
                            convolve(
                                x,
                                in_shape,
                                w,
                                rows,
                                columns,
                                *input_zero,
                                &requantize,
                                out,
                                out_shape,
                            );
                        }
                    }
                }
                Prepared::Add {
                    offsets,
                    multipliers,
                    output,
                    output_zero,
                    range,
                } => {
                    let y = read(step.inputs[1]);
                    for ((value, &a), &b) in out.iter_mut().zip(x).zip(y) {
                        // 두 입력을 같은 눈금으로 옮겨 더하고, 출력의 눈금으로
                        // 옮긴다. 먼저 2^20을 곱해 두어 자리를 잃지 않는다.
                        let a = multiply(
                            (i32::from(a) + offsets[0]) << 20,
                            multipliers[0].0,
                            multipliers[0].1,
                        );
                        let b = multiply(
                            (i32::from(b) + offsets[1]) << 20,
                            multipliers[1].0,
                            multipliers[1].1,
                        );
                        let sum = multiply(a + b, output.0, output.1) + output_zero;
                        *value = sum.clamp(range.0, range.1) as i8;
                    }
                }
                Prepared::Mean {
                    multiplier,
                    offset,
                    bias,
                } => {
                    let channels = out.len();
                    for (channel, value) in out.iter_mut().enumerate() {
                        let sum: i32 = x
                            .iter()
                            .skip(channel)
                            .step_by(channels)
                            .map(|&v| i32::from(v))
                            .sum();
                        let scaled =
                            multiply_twice_rounded(sum - offset, multiplier.0, multiplier.1);
                        *value = (scaled + bias).clamp(-128, 127) as i8;
                    }
                }
                Prepared::Pad { amounts, value } => {
                    pad(x, in_shape, amounts, *value, out, out_shape)
                }
                Prepared::Transpose { perm } => transpose(x, in_shape, perm, out, out_shape),
            }
            after(index);
        }
        Ok(())
    }
}

/// 마디 하나의 정수 배율들을 계산한다.
fn prepare<'a>(
    graph: &'a Graph,
    op: Op,
    inputs: &[usize],
    output: usize,
    shapes: &[Vec<usize>],
    quantization: &impl Fn(usize) -> Result<&'a Quantization>,
) -> Result<Prepared> {
    let out = quantization(output)?;
    let (out_scale, out_zero) = single(out, "출력")?;
    // 합성곱류: 출력 채널마다 `입력의 눈금 × 가중치의 눈금 / 출력의 눈금`.
    let weighted =
        |kind: Weighted, rows: Axis, columns: Axis, activation: Activation| -> Result<Prepared> {
            let (in_scale, in_zero) = single(quantization(inputs[0])?, "입력")?;
            let filter = quantization(inputs[1])?;
            let weights = graph.values[inputs[1]]
                .constant
                .as_ref()
                .ok_or_else(|| invalid("가중치가 파일에 없습니다"))?;
            let (w, shape) = (signed(weights)?, &shapes[inputs[1]]);
            let channels = match kind {
                Weighted::Depthwise => shape[3],
                _ => shape[0],
            };
            let per_channel = w.len() / channels;
            let scale =
                |channel: usize| filter.scales[if filter.scales.len() == 1 { 0 } else { channel }];
            if filter.scales.len() != 1 && filter.scales.len() != channels {
                return Err(invalid("가중치의 눈금 수가 출력 채널 수와 다릅니다"));
            }
            let bias_values: Vec<i32> = match inputs.get(2) {
                Some(&value) => {
                    let tensor = graph.values[value].constant.as_ref();
                    let tensor = tensor.ok_or_else(|| invalid("편향이 파일에 없습니다"))?;
                    if tensor.dtype() != DType::I32 {
                        return Err(Error::Unsupported("32비트 정수가 아닌 편향".into()));
                    }
                    tensor.to_vec::<i32>()?
                }
                None => vec![0; channels],
            };
            let mut multipliers = Vec::with_capacity(channels);
            let mut bias = Vec::with_capacity(channels);
            for channel in 0..channels {
                let effective =
                    f64::from(in_scale) * f64::from(scale(channel)) / f64::from(out_scale);
                multipliers.push(quantize_multiplier(effective));
                // Σ(x − z)·w = Σx·w − z·Σw. 뒤의 항은 미리 계산해 편향에 넣는다.
                let sum: i32 = match kind {
                    Weighted::Depthwise => w
                        .iter()
                        .skip(channel)
                        .step_by(channels)
                        .map(|&v| i32::from(v))
                        .sum(),
                    _ => w[channel * per_channel..(channel + 1) * per_channel]
                        .iter()
                        .map(|&v| i32::from(v))
                        .sum(),
                };
                // 깊이별 합성곱은 패딩한 칸을 건너뛰므로 이렇게 묶을 수 없다. 칸마다
                // 영점을 뺀다.
                bias.push(match kind {
                    Weighted::Depthwise => bias_values[channel],
                    // 묶은 값은 32비트를 넘나들 수 있다. 합에 다시 더하면 제자리로
                    // 돌아오므로 넘침을 그대로 둔다.
                    _ => bias_values[channel].wrapping_sub(in_zero.wrapping_mul(sum)),
                });
            }
            Ok(Prepared::Weighted {
                kind,
                rows,
                columns,
                multipliers,
                bias,
                input_zero: in_zero,
                output_zero: out_zero,
                range: activation_range(activation, out)?,
            })
        };
    let (input, filter) = (&shapes[inputs[0]], inputs.get(1).map(|&v| &shapes[v]));
    Ok(match op {
        Op::Conv2d {
            padding,
            stride,
            dilation,
            activation,
        } => {
            let filter = filter.ok_or_else(|| invalid("CONV_2D에 필터가 없습니다"))?;
            weighted(
                Weighted::Conv,
                Axis::new(input[1], filter[1], stride[0], dilation[0], padding, 0)?,
                Axis::new(input[2], filter[2], stride[1], dilation[1], padding, 1)?,
                activation,
            )?
        }
        Op::DepthwiseConv2d {
            padding,
            stride,
            dilation,
            multiplier,
            activation,
        } => {
            if multiplier != 1 {
                return Err(Error::Unsupported(
                    "배수가 1이 아닌 정수 깊이별 합성곱".into(),
                ));
            }
            let filter = filter.ok_or_else(|| invalid("DEPTHWISE_CONV_2D에 필터가 없습니다"))?;
            weighted(
                Weighted::Depthwise,
                Axis::new(input[1], filter[1], stride[0], dilation[0], padding, 0)?,
                Axis::new(input[2], filter[2], stride[1], dilation[1], padding, 1)?,
                activation,
            )?
        }
        Op::FullyConnected { activation, .. } => {
            // 선형 층은 1 × 1 사진 한 장에 대한 1 × 1 합성곱이다.
            let unit = Axis::new(1, 1, 1, 1, Padding::Valid, 0)?;
            weighted(Weighted::Dense, unit, unit, activation)?
        }
        Op::Add { activation } => {
            let (first, first_zero) = single(quantization(inputs[0])?, "입력")?;
            let (second, second_zero) = single(quantization(inputs[1])?, "입력")?;
            let twice_max = 2.0 * f64::from(first).max(f64::from(second));
            Prepared::Add {
                offsets: [-first_zero, -second_zero],
                multipliers: [
                    quantize_multiplier(f64::from(first) / twice_max),
                    quantize_multiplier(f64::from(second) / twice_max),
                ],
                output: quantize_multiplier(
                    twice_max / (f64::from(1 << 20) * f64::from(out_scale)),
                ),
                output_zero: out_zero,
                range: activation_range(activation, out)?,
            }
        }
        Op::Mean { axes, .. } => {
            if input.len() != 4 || !(axes == [1, 2] || axes == [2, 1]) {
                return Err(Error::Unsupported(
                    "높이와 너비가 아닌 축의 정수 MEAN".into(),
                ));
            }
            let (in_scale, in_zero) = single(quantization(inputs[0])?, "입력")?;
            let count = (input[1] * input[2]) as i64;
            // 눈금의 비를 배율로 바꾼 뒤, 원소 수로 나누는 일을 그 배율에
            // 넣는다. 배율을 왼쪽으로 옮겨 놓고 정수로 나누어 자리를 덜
            // 잃는다. LiteRT의 `QuantizedMeanOrSum`이 하는 그대로다.
            let (fixed, exponent) = quantize_multiplier(f64::from(in_scale) / f64::from(out_scale));
            let shift = (63 - (count as u64).leading_zeros() as i32)
                .min(32)
                .min(31 + exponent);
            Prepared::Mean {
                multiplier: (
                    ((i64::from(fixed) << shift) / count) as i32,
                    exponent - shift,
                ),
                offset: in_zero * count as i32,
                bias: out_zero,
            }
        }
        Op::Pad { amounts, .. } => Prepared::Pad {
            amounts,
            // 패딩하는 값은 실수 0, 곧 영점이다.
            value: out_zero.clamp(-128, 127) as i8,
        },
        Op::Transpose { perm } => Prepared::Transpose { perm },
        other => {
            return Err(Error::Unsupported(format!("정수 연산 {}", other.name())));
        }
    })
}

/// 누산한 32비트 정수를 출력의 8비트 정수로 옮기는 데 필요한 것들.
struct Requantize<'a> {
    rounding: Rounding,
    multipliers: &'a [(i32, i32)],
    bias: &'a [i32],
    output_zero: i32,
    range: (i32, i32),
}

impl Requantize<'_> {
    // ANCHOR: requantize
    /// 출력 채널 `channel`의 누산값을 출력의 눈금으로 옮긴다.
    #[inline]
    fn apply(&self, channel: usize, sum: i32) -> i8 {
        let (multiplier, shift) = self.multipliers[channel];
        let sum = sum.wrapping_add(self.bias[channel]);
        let scaled = self.rounding.multiply(sum, multiplier, shift) + self.output_zero;
        scaled.clamp(self.range.0, self.range.1) as i8
    }
    // ANCHOR_END: requantize

    // ANCHOR: requantize_all
    /// 채널마다의 누산값 `sums`를 한꺼번에 출력으로 옮긴다 (31장).
    ///
    /// 한 번 반올림하는 연산(합성곱)은 넷씩 벡터 명령으로 한다. 32비트 둘을
    /// 곱해 64비트를 얻는 명령이 한 번에 둘을 계산하므로 명령 둘에 넷이다.
    /// 계산은 [`multiply`]와 같다. 64비트로 곱하고, 절반을 더하고, 오른쪽으로
    /// 옮긴다. 그래서 값도 비트까지 같다.
    fn apply_all(&self, sums: &[i32], out: &mut [i8]) {
        #[cfg_attr(not(all(target_arch = "aarch64", not(miri))), allow(unused_mut))]
        let mut done = 0;
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        if self.rounding == Rounding::Once {
            // SAFETY: NEON은 64비트 ARM의 모든 프로세서에 있다.
            done = unsafe { self.apply_once_neon(sums, out) };
        }
        for channel in done..sums.len() {
            out[channel] = self.apply(channel, sums[channel]);
        }
    }

    /// [`apply_all`](Self::apply_all)의 NEON 판. 처리한 채널의 수를 돌려준다.
    ///
    /// # Safety
    ///
    /// 프로세서가 NEON을 지원해야 한다.
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    #[target_feature(enable = "neon")]
    unsafe fn apply_once_neon(&self, sums: &[i32], out: &mut [i8]) -> usize {
        use std::arch::aarch64::*;
        let count = sums.len().min(out.len()).min(self.bias.len());
        let full = count - count % 4;
        let zero = vdupq_n_s32(self.output_zero);
        let (low, high) = (vdupq_n_s32(self.range.0), vdupq_n_s32(self.range.1));
        for channel in (0..full).step_by(4) {
            let quad: [(i32, i32); 4] =
                std::array::from_fn(|lane| self.multipliers[channel + lane]);
            let multipliers = quad.map(|(multiplier, _)| multiplier);
            // 오른쪽으로 옮길 칸 수와, 그 전에 더할 절반.
            let totals = quad.map(|(_, shift)| i64::from(31 - shift));
            let rounds = totals.map(|total| 1_i64 << (total - 1));
            // SAFETY: `channel + 4 <= full`이므로 읽고 쓰는 넷은 슬라이스
            // 안이다. 배열은 길이가 맞는 지역 변수다.
            unsafe {
                let sum = vaddq_s32(
                    vld1q_s32(sums.as_ptr().add(channel)),
                    vld1q_s32(self.bias.as_ptr().add(channel)),
                );
                let m = vld1q_s32(multipliers.as_ptr());
                let scale = |product, lane: usize| {
                    let rounded = vaddq_s64(product, vld1q_s64(rounds.as_ptr().add(lane)));
                    // 음수만큼 왼쪽으로 옮기면 오른쪽으로 옮기는 것이다.
                    vshlq_s64(rounded, vnegq_s64(vld1q_s64(totals.as_ptr().add(lane))))
                };
                let first = scale(vmull_s32(vget_low_s32(sum), vget_low_s32(m)), 0);
                let second = scale(vmull_high_s32(sum, m), 2);
                let scaled = vcombine_s32(vmovn_s64(first), vmovn_s64(second));
                let clamped = vminq_s32(vmaxq_s32(vaddq_s32(scaled, zero), low), high);
                let narrow = vmovn_s16(vcombine_s16(vmovn_s32(clamped), vmovn_s32(clamped)));
                let mut bytes = [0_i8; 8];
                vst1_s8(bytes.as_mut_ptr(), narrow);
                out[channel..channel + 4].copy_from_slice(&bytes[..4]);
            }
        }
        full
    }
    // ANCHOR_END: requantize_all
}

// ANCHOR: dot
/// 8비트 정수 둘의 내적. 32비트로 누산한다.
#[inline]
fn dot(x: &[i8], w: &[i8]) -> i32 {
    // Miri는 이 벡터 명령을 흉내 내지 못한다. 그때는 평범한 반복문을 쓴다.
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        // SAFETY: NEON은 64비트 ARM의 모든 프로세서에 있다.
        unsafe { dot_neon(x, w) }
    }
    #[cfg(not(all(target_arch = "aarch64", not(miri))))]
    {
        dot_portable(x, w)
    }
}

/// 벡터 명령을 직접 쓰지 않은 내적. 정수의 덧셈은 순서를 바꿔도 값이 같으므로
/// 컴파일러가 벡터 명령으로 바꿀 수 있지만, 8비트를 32비트로 넓히는 길을
/// 잘 고르지는 못한다.
#[cfg_attr(all(target_arch = "aarch64", not(miri)), allow(dead_code))]
fn dot_portable(x: &[i8], w: &[i8]) -> i32 {
    x.iter()
        .zip(w)
        .map(|(&a, &b)| i32::from(a) * i32::from(b))
        .sum()
}

/// NEON으로 짠 내적. 값 열여섯 개씩 읽어, 8비트끼리 곱해 16비트를 얻고
/// (`vmull_s8`), 이웃한 둘씩 더해 32비트 누산기에 쌓는다(`vpadalq_s16`).
/// 8비트 둘의 곱은 16비트에 들어가고, 그 둘의 합은 17비트이므로 32비트로
/// 넓히면서 더해야 넘치지 않는다.
///
/// # Safety
///
/// 프로세서가 NEON을 지원해야 한다.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(x: &[i8], w: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let len = x.len().min(w.len());
    let full = len - len % 16;
    let mut sums = [vdupq_n_s32(0); 2];
    for start in (0..full).step_by(16) {
        // SAFETY: `start + 16 <= full <= len`이므로 읽는 열여섯 값은 두
        // 슬라이스 안에 있다.
        let (a, b) = unsafe {
            (
                vld1q_s8(x.as_ptr().add(start)),
                vld1q_s8(w.as_ptr().add(start)),
            )
        };
        sums[0] = vpadalq_s16(sums[0], vmull_s8(vget_low_s8(a), vget_low_s8(b)));
        sums[1] = vpadalq_s16(sums[1], vmull_high_s8(a, b));
    }
    let mut sum = vaddvq_s32(vaddq_s32(sums[0], sums[1]));
    for index in full..len {
        sum += i32::from(x[index]) * i32::from(w[index]);
    }
    sum
}
// ANCHOR_END: dot

// ANCHOR: dots
/// 창 하나와 필터 모두의 내적을 `sums`에 적는다. `w`는 필터마다 `patch`개씩
/// 이어져 있다.
///
/// 31장: [`dot`]을 필터마다 부르면 창을 필터의 수만큼 다시 읽고, 부를
/// 때마다 누산기를 하나로 모은다. 여기서는 필터 넷을 함께 계산해 창을 한
/// 번만 읽는다. 프로세서에 8비트 내적 명령(`SDOT`)이 있으면 그것을 쓴다.
/// 정수의 덧셈은 순서를 바꿔도 값이 같으므로 결과는 [`dot`]과 비트까지
/// 같다.
fn dots(values: &[i8], w: &[i8], patch: usize, sums: &mut [i32]) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: 바로 위에서 이 프로세서에 명령이 있음을 확인했다.
        unsafe { dots_sdot(values, w, patch, sums) };
        return;
    }
    for (sum, filter) in sums.iter_mut().zip(w.chunks_exact(patch)) {
        *sum = dot(values, filter);
    }
}

/// `SDOT`으로 짠 [`dots`]. 명령 하나가 8비트 값 열여섯 쌍을 곱해 넷씩 더한
/// 32비트 값 넷을 누산기에 더한다.
///
/// # Safety
///
/// 프로세서가 NEON과 `dotprod`를 지원해야 한다.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dots_sdot(values: &[i8], w: &[i8], patch: usize, sums: &mut [i32]) {
    use std::arch::aarch64::*;
    let full = patch - patch % 16;
    let tail = |filter: &[i8]| -> i32 {
        (full..patch)
            .map(|index| i32::from(values[index]) * i32::from(filter[index]))
            .sum()
    };
    let (quads, rest) = sums.as_chunks_mut::<4>();
    for (quad, filters) in quads.iter_mut().zip(w.chunks_exact(4 * patch)) {
        let filters: [&[i8]; 4] = std::array::from_fn(|f| &filters[f * patch..][..patch]);
        let mut acc = [vdupq_n_s32(0); 4];
        for start in (0..full).step_by(16) {
            // SAFETY: `start + 16 <= full <= patch`이므로 읽는 열여섯 값은
            // 창과 필터 넷 안에 있다.
            unsafe {
                let x = vld1q_s8(values.as_ptr().add(start));
                for (acc, filter) in acc.iter_mut().zip(filters) {
                    *acc = vdotq_s32(*acc, x, vld1q_s8(filter.as_ptr().add(start)));
                }
            }
        }
        for ((sum, acc), filter) in quad.iter_mut().zip(acc).zip(filters) {
            *sum = vaddvq_s32(acc) + tail(filter);
        }
    }
    let done = quads.len() * 4;
    for (sum, filter) in rest.iter_mut().zip(w[done * patch..].chunks_exact(patch)) {
        let mut acc = vdupq_n_s32(0);
        for start in (0..full).step_by(16) {
            // SAFETY: 위와 같다.
            unsafe {
                let x = vld1q_s8(values.as_ptr().add(start));
                acc = vdotq_s32(acc, x, vld1q_s8(filter.as_ptr().add(start)));
            }
        }
        *sum = vaddvq_s32(acc) + tail(filter);
    }
}
// ANCHOR_END: dots

// ANCHOR: convolve
/// 정수 합성곱. 출력의 한 줄이 일의 단위다. 창을 펼쳐 필터마다 내적한다.
#[allow(clippy::too_many_arguments)]
fn convolve(
    x: &[i8],
    in_shape: &[usize],
    w: &[i8],
    rows: &Axis,
    columns: &Axis,
    input_zero: i32,
    requantize: &Requantize<'_>,
    out: &mut [i8],
    out_shape: &[usize],
) {
    let channels = *in_shape.last().expect("형상이 있다");
    let out_channels = *out_shape.last().expect("형상이 있다");
    let patch = rows.filter * columns.filter * channels;
    let pointwise = patch == channels && rows.stride == 1 && columns.stride == 1;
    let line = columns.output * out_channels;
    let image = rows.input * columns.input * channels;
    let pool = threads::global();
    let work = rows.output * columns.output * patch * out_channels;
    let wanted = if work < (1 << 16) {
        1
    } else {
        pool.for_vectors(rows.output)
    };
    let padding = input_zero.clamp(-128, 127) as i8;
    for (x, out) in x
        .chunks_exact(image)
        .zip(out.chunks_exact_mut(rows.output * line))
    {
        let task = |row: usize, out: &mut [i8]| {
            let mut window = vec![padding; if pointwise { 0 } else { patch }];
            let mut sums = vec![0_i32; out_channels];
            for (column, pixel) in out.chunks_exact_mut(out_channels).enumerate() {
                let values: &[i8] = if pointwise {
                    &x[(row * columns.input + column) * channels..][..channels]
                } else {
                    // 패딩한 자리에는 영점을 둔다. 실수 0이다.
                    for tap_row in 0..rows.filter {
                        for tap_column in 0..columns.filter {
                            let target = &mut window
                                [(tap_row * columns.filter + tap_column) * channels..][..channels];
                            match (
                                rows.source(row, tap_row),
                                columns.source(column, tap_column),
                            ) {
                                (Some(r), Some(c)) => {
                                    target.copy_from_slice(
                                        &x[(r * columns.input + c) * channels..][..channels],
                                    );
                                }
                                _ => target.fill(padding),
                            }
                        }
                    }
                    &window
                };
                // 31장: 필터 모두의 내적을 먼저 구하고, 그다음에 출력의
                // 눈금으로 옮긴다.
                dots(values, w, patch, &mut sums);
                requantize.apply_all(&sums, pixel);
            }
        };
        pool.pool().run_chunks(out, line, wanted, &task);
    }
}
// ANCHOR_END: convolve

/// 정수 깊이별 합성곱.
#[allow(clippy::too_many_arguments)]
fn depthwise(
    x: &[i8],
    in_shape: &[usize],
    w: &[i8],
    rows: &Axis,
    columns: &Axis,
    input_zero: i32,
    requantize: &Requantize<'_>,
    out: &mut [i8],
    out_shape: &[usize],
) {
    let channels = *in_shape.last().expect("형상이 있다");
    debug_assert_eq!(channels, *out_shape.last().expect("형상이 있다"));
    let line = columns.output * channels;
    let image = rows.input * columns.input * channels;
    let pool = threads::global();
    let work = rows.output * line * rows.filter * columns.filter;
    let wanted = if work < (1 << 16) {
        1
    } else {
        pool.for_vectors(rows.output)
    };
    // 패딩한 칸은 실수 0이므로 건너뛴다. 나머지 칸은 `(x − 영점) × w`다.
    for (x, out) in x
        .chunks_exact(image)
        .zip(out.chunks_exact_mut(rows.output * line))
    {
        let task = |row: usize, out: &mut [i8]| {
            let mut sums = vec![0_i32; channels];
            for (column, pixel) in out.chunks_exact_mut(channels).enumerate() {
                sums.fill(0);
                for tap_row in 0..rows.filter {
                    let Some(r) = rows.source(row, tap_row) else {
                        continue;
                    };
                    for tap_column in 0..columns.filter {
                        let Some(c) = columns.source(column, tap_column) else {
                            continue;
                        };
                        let values = &x[(r * columns.input + c) * channels..][..channels];
                        let weights =
                            &w[(tap_row * columns.filter + tap_column) * channels..][..channels];
                        for ((sum, &value), &weight) in sums.iter_mut().zip(values).zip(weights) {
                            *sum += (i32::from(value) - input_zero) * i32::from(weight);
                        }
                    }
                }
                requantize.apply_all(&sums, pixel);
            }
        };
        pool.pool().run_chunks(out, line, wanted, &task);
    }
}

/// 가장자리에 값을 패딩한다.
fn pad(
    x: &[i8],
    in_shape: &[usize],
    amounts: &[[usize; 2]],
    value: i8,
    out: &mut [i8],
    out_shape: &[usize],
) {
    out.fill(value);
    let last = in_shape.len() - 1;
    let width = in_shape[last];
    let mut steps = vec![1; out_shape.len()];
    for axis in (0..last).rev() {
        steps[axis] = steps[axis + 1] * out_shape[axis + 1];
    }
    let mut index = vec![0; last];
    for line in x.chunks_exact(width) {
        let mut start = amounts[last][0];
        for axis in 0..last {
            start += (index[axis] + amounts[axis][0]) * steps[axis];
        }
        out[start..start + width].copy_from_slice(line);
        for axis in (0..last).rev() {
            index[axis] += 1;
            if index[axis] < in_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// 축의 순서를 바꾼다.
fn transpose(x: &[i8], in_shape: &[usize], perm: &[usize], out: &mut [i8], out_shape: &[usize]) {
    let mut source = vec![1; in_shape.len()];
    for axis in (0..in_shape.len() - 1).rev() {
        source[axis] = source[axis + 1] * in_shape[axis + 1];
    }
    let steps: Vec<usize> = perm.iter().map(|&axis| source[axis]).collect();
    let last = out_shape.len() - 1;
    let mut index = vec![0; last];
    for line in out.chunks_exact_mut(out_shape[last]) {
        let start: usize = index.iter().zip(&steps).map(|(i, step)| i * step).sum();
        for (column, value) in line.iter_mut().enumerate() {
            *value = x[start + column * steps[last]];
        }
        for axis in (0..last).rev() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 손으로 만든 정수 그래프: `1 × 1` 합성곱 하나와, 그 결과를 입력에 더하는
    /// 덧셈. 실수로 계산한 값과 눈금 한 칸 안에서 같아야 한다.
    #[test]
    fn runs_a_hand_built_integer_graph() {
        let quantized = |scale: f32, zero: i32| {
            Some(Quantization {
                scales: vec![scale],
                zero_points: vec![zero],
                axis: 0,
            })
        };
        let mut graph = Graph::new("정수");
        let x = graph.value("x", &[1, 2, 2, 2]);
        graph.values[x].quantization = quantized(0.05, -3);
        let weights: Vec<i8> = vec![10, -20, 30, 40];
        let bytes: Vec<u8> = weights.iter().map(|&w| w as u8).collect();
        let w = graph.constant("w", Tensor::from_vec(bytes, &[2, 1, 1, 2]).unwrap());
        graph.values[w].quantization = Some(Quantization {
            scales: vec![0.01, 0.02],
            zero_points: vec![0, 0],
            axis: 0,
        });
        let b = graph.constant("b", Tensor::from_vec(vec![100_i32, -50], &[2]).unwrap());
        let h = graph.value("h", &[1, 2, 2, 2]);
        graph.values[h].quantization = quantized(0.02, 5);
        let y = graph.value("y", &[1, 2, 2, 2]);
        graph.values[y].quantization = quantized(0.04, -10);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        graph.node(
            Op::Conv2d {
                padding: Padding::Valid,
                stride: [1, 1],
                dilation: [1, 1],
                activation: Activation::None,
            },
            &[x, w, b],
            &[h],
        );
        graph.node(
            Op::Add {
                activation: Activation::Relu,
            },
            &[x, h],
            &[y],
        );
        assert!(graph.is_quantized());
        let mut interpreter = QuantizedInterpreter::new(&graph).unwrap();
        let input: [i8; 8] = [-128, 127, 0, -3, 17, -40, 64, 5];
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().unwrap();

        // 같은 계산을 실수로.
        let real_x: Vec<f32> = input.iter().map(|&q| 0.05 * (f32::from(q) + 3.0)).collect();
        let mut expected = Vec::new();
        for pixel in real_x.chunks(2) {
            let h0 = 0.01 * (10.0 * pixel[0] - 20.0 * pixel[1]) / 1.0 + 0.05 * 0.01 * 100.0;
            let h1 = 0.02 * (30.0 * pixel[0] + 40.0 * pixel[1]) / 1.0 - 0.05 * 0.02 * 50.0;
            // 중간 결과도 int8의 범위에 잘린다.
            let clip = |v: f32| v.clamp(0.02 * (-128.0 - 5.0), 0.02 * (127.0 - 5.0));
            expected.push((pixel[0] + clip(h0)).max(0.0));
            expected.push((pixel[1] + clip(h1)).max(0.0));
        }
        let (scale, zero) = interpreter.output_quantization(0);
        for (&q, wanted) in interpreter.output(0).iter().zip(expected) {
            let got = scale * (i32::from(q) - zero) as f32;
            let wanted = wanted.clamp(scale * (-128 - zero) as f32, scale * (127 - zero) as f32);
            assert!((got - wanted).abs() <= 1.5 * scale, "{got} != {wanted}");
        }
        // 눈금이 없는 값이 섞인 그래프는 받지 않는다.
        graph.values[h].quantization = None;
        assert!(QuantizedInterpreter::new(&graph).is_err());
    }

    #[test]
    fn multiplier_splits_into_mantissa_and_shift() {
        // 0.5는 가수 0.5(2^30), 지수 0.
        assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
        assert_eq!(quantize_multiplier(1.0), (1 << 30, 1));
        assert_eq!(quantize_multiplier(0.25), (1 << 30, -1));
        assert_eq!(quantize_multiplier(0.0), (0, 0));
        let (fixed, shift) = quantize_multiplier(0.0003);
        let restored = f64::from(fixed) / f64::from(1_u32 << 31) * 2.0_f64.powi(shift);
        assert!((restored - 0.0003).abs() < 1e-12);
        // 가수가 1로 반올림되면 지수를 하나 올린다.
        assert_eq!(quantize_multiplier(1.0 - 1e-12), (1 << 30, 1));
    }

    #[test]
    fn multiply_rounds_to_nearest() {
        let (fixed, shift) = quantize_multiplier(0.5);
        assert_eq!(multiply(10, fixed, shift), 5);
        // 딱 절반이면 큰 쪽으로 간다(5.5 → 6, −5.5 → −5). LiteRT가 쓰는
        // 곱셈의 성질이고, 같은 비트를 내려면 이것까지 같아야 한다.
        assert_eq!(multiply(11, fixed, shift), 6);
        assert_eq!(multiply(-11, fixed, shift), -5);
        assert_eq!(multiply_twice_rounded(11, fixed, shift), 6);
        assert_eq!(multiply_twice_rounded(-11, fixed, shift), -5);
        let (fixed, shift) = quantize_multiplier(0.001);
        for x in [-100_000, -1234, 0, 999, 123_456, 2_000_000] {
            let exact = (f64::from(x) * 0.001).round() as i32;
            assert!((multiply(x, fixed, shift) - exact).abs() <= 1, "{x}");
        }
        // 배율이 1보다 크면 먼저 왼쪽으로 옮긴다.
        let (fixed, shift) = quantize_multiplier(3.0);
        assert_eq!(multiply(7, fixed, shift), 21);
    }

    #[test]
    fn rounding_primitives_match_their_definitions() {
        assert_eq!(rounding_divide_by_power_of_two(5, 1), 3);
        assert_eq!(rounding_divide_by_power_of_two(-5, 1), -3);
        assert_eq!(rounding_divide_by_power_of_two(4, 2), 1);
        assert_eq!(rounding_divide_by_power_of_two(-6, 2), -2);
        assert_eq!(rounding_divide_by_power_of_two(7, 0), 7);
        assert_eq!(rounding_doubling_high_mul(1 << 30, 1 << 30), 1 << 29);
        assert_eq!(rounding_doubling_high_mul(i32::MIN, i32::MIN), i32::MAX);
    }

    #[test]
    fn dot_accumulates_in_32_bits() {
        let x = [127_i8; 1000];
        let w = [-128_i8; 1000];
        assert_eq!(dot(&x, &w), -16_256_000);
        let x = [-128_i8; 37];
        assert_eq!(dot(&x, &x), 37 * 16_384);
        // 벡터 명령으로 짠 것과 평범한 반복문이 같은 값을 낸다.
        let a: Vec<i8> = (0..1003).map(|i| ((i * 37) % 251) as u8 as i8).collect();
        let b: Vec<i8> = (0..1003).map(|i| ((i * 91) % 239) as u8 as i8).collect();
        for len in [0, 1, 15, 16, 17, 100, 1003] {
            assert_eq!(
                dot(&a[..len], &b[..len]),
                dot_portable(&a[..len], &b[..len])
            );
        }
    }
}
