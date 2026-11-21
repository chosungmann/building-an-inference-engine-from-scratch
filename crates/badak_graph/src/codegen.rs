//! 그래프를 Rust 소스로 옮긴다 (52장).
//!
//! 26장의 해석기는 그래프를 읽어 가며 마디마다 커널을 부른다. 이 모듈은
//! 같은 그래프를 받아 **그 그래프만 계산하는 프로그램의 소스**를 낸다.
//! 형상이 숫자로 박히고, 이어지는 연산들이 반복문 하나로 합쳐진다.
//!
//! 세 단계를 거친다. 실제 컴파일러들이 중간 표현을 여러 층으로 두는 것의
//! 가장 작은 꼴이다.
//!
//! 1. 그래프 ([`Graph`]): "이 값에 이 연산을".
//! 2. 반복문의 표현 ([`Stage`]): "이 길이만큼 돌면서 원소마다 이 식을".
//! 3. 소스: Rust의 글.
//!
//! 아는 연산은 여섯이다. 선형 층, 더하기, 곱하기, ReLU, 자르기, 시그모이드.
//! 형상은 모두 정해져 있어야 한다.

use std::collections::HashMap;
use std::fmt::Write;

use crate::{Activation, Error, Graph, Op, Result};

// ANCHOR: ir
/// 원소 하나를 계산하는 식.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// 이 단계의 `index`번째 입력의 원소.
    Input(usize),
    /// 상수 배열 `name`의 원소. 원소가 하나뿐인 배열이면 그 값이다.
    Constant(String, usize),
    /// 더하기.
    Add(Box<Expr>, Box<Expr>),
    /// 곱하기.
    Mul(Box<Expr>, Box<Expr>),
    /// 아래와 위를 자른다.
    Clamp(Box<Expr>, f32, f32),
    /// 시그모이드.
    Sigmoid(Box<Expr>),
}

/// 반복문 하나.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// 원소마다 `body`를 계산해 `output`에 적는다.
    Map {
        /// 읽는 값들의 번호.
        inputs: Vec<usize>,
        /// 쓰는 값의 번호.
        output: usize,
        /// 원소의 수.
        len: usize,
        /// 원소 하나의 식.
        body: Expr,
    },
    /// 선형 층. 행마다 내적을 하고 `post`를 적용한다. `post`의 `Input(0)`이
    /// 내적에 치우침을 더한 값이다.
    Linear {
        /// 읽는 값의 번호.
        input: usize,
        /// 쓰는 값의 번호.
        output: usize,
        /// 가중치 배열의 이름.
        weight: String,
        /// 치우침 배열의 이름.
        bias: Option<String>,
        /// 행의 수.
        rows: usize,
        /// 열의 수.
        cols: usize,
        /// 결과에 이어서 적용하는 식.
        post: Expr,
    },
}
// ANCHOR_END: ir

/// 소스를 내는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// 이어지는 연산을 반복문 하나로 합칠지.
    pub fuse: bool,
    /// 내적 하나의 합을 변수 몇 개에 나누어 담을지(16장).
    pub accumulators: usize,
    /// 가중치의 행을 몇 개씩 함께 읽을지(17장).
    pub rows_at_once: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            fuse: true,
            accumulators: 8,
            rows_at_once: 4,
        }
    }
}

/// 옮긴 결과.
#[derive(Debug, Clone)]
pub struct Program {
    /// 반복문들. 실행 순서대로다.
    pub stages: Vec<Stage>,
    /// 상수 배열들: 이름과 값.
    pub constants: Vec<(String, Vec<f32>)>,
    /// 값마다의 길이.
    pub lengths: HashMap<usize, usize>,
    /// 입력 값의 번호.
    pub input: usize,
    /// 출력 값의 번호.
    pub output: usize,
}

fn unsupported(what: impl Into<String>) -> Error {
    Error::Unsupported(what.into())
}

fn activation(expr: Expr, kind: Activation) -> Result<Expr> {
    Ok(match kind {
        Activation::None => expr,
        Activation::Relu => Expr::Clamp(Box::new(expr), 0.0, f32::INFINITY),
        Activation::Relu6 => Expr::Clamp(Box::new(expr), 0.0, 6.0),
        Activation::ReluN1To1 => Expr::Clamp(Box::new(expr), -1.0, 1.0),
        other => return Err(unsupported(format!("활성화 함수 {other:?}"))),
    })
}

// ANCHOR: lower
/// 1단계에서 2단계로: 그래프의 마디를 반복문으로 낮춘다.
pub fn lower(graph: &Graph) -> Result<Program> {
    let (&[input], &[output]) = (&graph.inputs[..], &graph.outputs[..]) else {
        return Err(unsupported("입력과 출력이 하나씩인 그래프만 옮깁니다"));
    };
    let mut constants = Vec::new();
    let mut lengths = HashMap::new();
    for (index, value) in graph.values.iter().enumerate() {
        match &value.constant {
            Some(tensor) => constants.push((format!("C{index}"), tensor.to_vec::<f32>()?)),
            None => {
                lengths.insert(index, value.shape.iter().product());
            }
        }
    }
    // 연산의 인자 하나를 식으로: 계산되는 값이면 입력, 상수이면 상수 배열.
    let operand = |value: usize, inputs: &mut Vec<usize>| -> Expr {
        if graph.values[value].constant.is_some() {
            let len = graph.values[value].shape.iter().product();
            return Expr::Constant(format!("C{value}"), len);
        }
        let slot = inputs
            .iter()
            .position(|&known| known == value)
            .unwrap_or_else(|| {
                inputs.push(value);
                inputs.len() - 1
            });
        Expr::Input(slot)
    };
    let mut stages = Vec::new();
    for index in graph.order()? {
        let node = &graph.nodes[index];
        let out = node.outputs[0];
        let len = lengths[&out];
        let mut inputs = Vec::new();
        let first = |inputs: &mut Vec<usize>| Box::new(operand(node.inputs[0], inputs));
        let body = match &node.op {
            Op::FullyConnected {
                activation: kind, ..
            } => {
                let (weight, bias) = (node.inputs[1], node.inputs.get(2).copied());
                let &[rows, cols] = &graph.values[weight].shape[..] else {
                    return Err(unsupported("가중치가 행렬이 아닌 선형 층"));
                };
                stages.push(Stage::Linear {
                    input: node.inputs[0],
                    output: out,
                    weight: format!("C{weight}"),
                    bias: bias.map(|bias| format!("C{bias}")),
                    rows,
                    cols,
                    post: activation(Expr::Input(0), *kind)?,
                });
                continue;
            }
            Op::Add { activation: kind } => {
                let sum = Expr::Add(
                    first(&mut inputs),
                    Box::new(operand(node.inputs[1], &mut inputs)),
                );
                activation(sum, *kind)?
            }
            Op::Mul { activation: kind } => {
                let product = Expr::Mul(
                    first(&mut inputs),
                    Box::new(operand(node.inputs[1], &mut inputs)),
                );
                activation(product, *kind)?
            }
            Op::Relu => Expr::Clamp(first(&mut inputs), 0.0, f32::INFINITY),
            Op::Clip { min, max } => Expr::Clamp(first(&mut inputs), *min, *max),
            Op::Sigmoid => Expr::Sigmoid(first(&mut inputs)),
            other => return Err(unsupported(format!("연산 {}", other.name()))),
        };
        stages.push(Stage::Map {
            inputs,
            output: out,
            len,
            body,
        });
    }
    Ok(Program {
        stages,
        constants,
        lengths,
        input,
        output,
    })
}
// ANCHOR_END: lower

/// `expr` 안의 `Input(slot)`을 `with`로 바꾼다. 나머지 입력의 번호는 `remap`으로 옮긴다.
fn substitute(expr: &Expr, slot: usize, with: &Expr, remap: &[usize]) -> Expr {
    let again = |inner: &Expr| Box::new(substitute(inner, slot, with, remap));
    match expr {
        Expr::Input(index) if *index == slot => with.clone(),
        Expr::Input(index) => Expr::Input(remap[*index]),
        Expr::Constant(name, len) => Expr::Constant(name.clone(), *len),
        Expr::Add(a, b) => Expr::Add(again(a), again(b)),
        Expr::Mul(a, b) => Expr::Mul(again(a), again(b)),
        Expr::Clamp(inner, low, high) => Expr::Clamp(again(inner), *low, *high),
        Expr::Sigmoid(inner) => Expr::Sigmoid(again(inner)),
    }
}

// ANCHOR: fuse
/// 2단계 안에서: 이어지는 반복문을 하나로 합친다.
///
/// 앞 반복문이 쓴 값을 뒤 반복문만 읽고 길이가 같으면, 뒤의 식에서 그 값을
/// 읽는 자리에 앞의 식을 통째로 넣는다. 가운데 값은 메모리에 적히지 않는다.
pub fn fuse(program: &mut Program) {
    let mut index = 0;
    while index + 1 < program.stages.len() {
        let (produced, readers) = {
            let produced = match &program.stages[index] {
                Stage::Map { output, .. } | Stage::Linear { output, .. } => *output,
            };
            let readers = program
                .stages
                .iter()
                .filter(|stage| reads(stage, produced))
                .count();
            (produced, readers)
        };
        let next = program.stages[index + 1].clone();
        let Stage::Map {
            inputs,
            output,
            len,
            body,
        } = next
        else {
            index += 1;
            continue;
        };
        let slot = inputs.iter().position(|&value| value == produced);
        let (Some(slot), 1, false) = (slot, readers, produced == program.output) else {
            index += 1;
            continue;
        };
        let merged = match &program.stages[index] {
            Stage::Map {
                inputs: before,
                len: before_len,
                body: before_body,
                ..
            } if *before_len == len => {
                // 뒤 반복문의 다른 입력들을 앞 반복문의 입력 목록 뒤에 붙인다.
                let mut all = before.clone();
                let mut remap = vec![0; inputs.len()];
                for (position, &value) in inputs.iter().enumerate() {
                    if position == slot {
                        continue;
                    }
                    remap[position] =
                        all.iter()
                            .position(|&known| known == value)
                            .unwrap_or_else(|| {
                                all.push(value);
                                all.len() - 1
                            });
                }
                let identity: Vec<usize> = (0..before.len()).collect();
                let inner = substitute(before_body, usize::MAX, &Expr::Input(0), &identity);
                Some(Stage::Map {
                    inputs: all,
                    output,
                    len,
                    body: substitute(&body, slot, &inner, &remap),
                })
            }
            Stage::Linear {
                input,
                weight,
                bias,
                rows,
                cols,
                post,
                ..
            } if inputs.len() == 1 && *rows == len => Some(Stage::Linear {
                input: *input,
                output,
                weight: weight.clone(),
                bias: bias.clone(),
                rows: *rows,
                cols: *cols,
                post: substitute(&body, 0, post, &[0]),
            }),
            _ => None,
        };
        match merged {
            Some(stage) => {
                program.stages[index] = stage;
                program.stages.remove(index + 1);
                program.lengths.remove(&produced);
            }
            None => index += 1,
        }
    }
}
// ANCHOR_END: fuse

fn reads(stage: &Stage, value: usize) -> bool {
    match stage {
        Stage::Map { inputs, .. } => inputs.contains(&value),
        Stage::Linear { input, .. } => *input == value,
    }
}

fn literal(value: f32) -> String {
    if value == f32::INFINITY {
        "f32::INFINITY".to_string()
    } else if value == f32::NEG_INFINITY {
        "f32::NEG_INFINITY".to_string()
    } else {
        // `{:?}`는 같은 비트로 되읽히는 가장 짧은 글을 낸다.
        format!("{value:?}_f32")
    }
}

/// 식을 Rust의 글로. `names[k]`가 `Input(k)`를 읽는 글이다.
fn write_expr(expr: &Expr, names: &[String]) -> String {
    match expr {
        Expr::Input(index) => names[*index].clone(),
        Expr::Constant(name, 1) => format!("{name}[0]"),
        Expr::Constant(name, _) => format!("{name}[i]"),
        Expr::Add(a, b) => format!("({} + {})", write_expr(a, names), write_expr(b, names)),
        Expr::Mul(a, b) => format!("({} * {})", write_expr(a, names), write_expr(b, names)),
        Expr::Clamp(inner, low, high) if *high == f32::INFINITY => {
            format!("{}.max({})", write_expr(inner, names), literal(*low))
        }
        Expr::Clamp(inner, low, high) => {
            format!(
                "{}.max({}).min({})",
                write_expr(inner, names),
                literal(*low),
                literal(*high)
            )
        }
        Expr::Sigmoid(inner) => format!("(1.0 / (1.0 + (-{}).exp()))", write_expr(inner, names)),
    }
}

// ANCHOR: dot
/// 길이가 `cols`인 벡터 하나와 가중치 `rows`행의 내적을 한꺼번에 계산하는
/// 함수의 소스. 합의 수와 길이가 모두 숫자로 박힌다.
pub fn write_dot(out: &mut String, name: &str, cols: usize, accumulators: usize, rows: usize) {
    let (a, full) = (accumulators, cols - cols % accumulators);
    let _ = writeln!(out, "#[inline(always)]");
    let _ = writeln!(
        out,
        "fn {name}(x: &[f32; {cols}], w: [&[f32; {cols}]; {rows}]) -> [f32; {rows}] {{"
    );
    let _ = writeln!(out, "    let mut sums = [[0.0_f32; {a}]; {rows}];");
    let _ = writeln!(out, "    let mut i = 0;");
    let _ = writeln!(out, "    while i < {full} {{");
    let _ = writeln!(out, "        for r in 0..{rows} {{");
    let _ = writeln!(out, "            for k in 0..{a} {{");
    let _ = writeln!(out, "                sums[r][k] += x[i + k] * w[r][i + k];");
    let _ = writeln!(out, "            }}");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "        i += {a};");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    let mut total = [0.0_f32; {rows}];");
    let _ = writeln!(out, "    for r in 0..{rows} {{");
    let _ = writeln!(out, "        for k in 0..{a} {{");
    let _ = writeln!(out, "            total[r] += sums[r][k];");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "        for j in {full}..{cols} {{");
    let _ = writeln!(out, "            total[r] += x[j] * w[r][j];");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    total");
    let _ = writeln!(out, "}}");
}
// ANCHOR_END: dot

// ANCHOR: emit
/// 2단계에서 3단계로: 반복문들을 Rust의 함수 `run`으로 적는다.
///
/// 내는 소스에는 상수 배열, 값마다의 자리를 쥔 `State`, 그리고
/// `run(state, input, output)`이 들어 있다.
pub fn emit(program: &Program, options: &Options) -> String {
    let mut out = String::from("// badak_graph::codegen 이 만든 파일이다. 고치지 않는다.\n");
    for (name, values) in &program.constants {
        let _ = write!(out, "static {name}: [f32; {}] = [", values.len());
        for value in values {
            let _ = write!(out, "{value:?},");
        }
        out.push_str("];\n");
    }
    // 가운데 값들의 자리. 입력과 출력은 부르는 쪽이 준다.
    let mut middle: Vec<(usize, usize)> = program
        .lengths
        .iter()
        .filter(|(value, _)| **value != program.input && **value != program.output)
        .filter(|(value, _)| program.stages.iter().any(|stage| reads(stage, **value)))
        .map(|(&value, &len)| (value, len))
        .collect();
    middle.sort_unstable();
    out.push_str("pub struct State {\n");
    for (value, len) in &middle {
        let _ = writeln!(out, "    v{value}: Box<[f32; {len}]>,");
    }
    out.push_str("}\nimpl State {\n    pub fn new() -> Self {\n        Self {\n");
    for (value, len) in &middle {
        let _ = writeln!(out, "            v{value}: Box::new([0.0; {len}]),");
    }
    out.push_str("        }\n    }\n}\n");

    let mut dots = Vec::new();
    let (input_len, output_len) = (
        program.lengths[&program.input],
        program.lengths[&program.output],
    );
    let mut body = String::new();
    let read = |value: usize| {
        if value == program.input {
            "input".to_string()
        } else {
            format!("state.v{value}")
        }
    };
    for stage in &program.stages {
        match stage {
            Stage::Map {
                inputs,
                output,
                len,
                body: expr,
            } => {
                let names: Vec<String> = inputs
                    .iter()
                    .map(|&value| format!("{}[i]", read(value)))
                    .collect();
                let target = if *output == program.output {
                    "output".to_string()
                } else {
                    format!("state.v{output}")
                };
                let _ = writeln!(body, "    for i in 0..{len} {{");
                let _ = writeln!(body, "        let value = {};", write_expr(expr, &names));
                let _ = writeln!(body, "        {target}[i] = value;");
                let _ = writeln!(body, "    }}");
            }
            Stage::Linear {
                input,
                output,
                weight,
                bias,
                rows,
                cols,
                post,
            } => {
                let together = options.rows_at_once.clamp(1, *rows);
                let kernel = format!("dot_{cols}_{}_{together}", options.accumulators);
                if !dots.contains(&kernel) {
                    write_dot(&mut out, &kernel, *cols, options.accumulators, together);
                    if together > 1 {
                        write_dot(
                            &mut out,
                            &format!("{kernel}_one"),
                            *cols,
                            options.accumulators,
                            1,
                        );
                    }
                    dots.push(kernel.clone());
                }
                let target = if *output == program.output {
                    "output".to_string()
                } else {
                    format!("state.v{output}")
                };
                let source = read(*input);
                let row = |index: &str| {
                    format!("{weight}[{index} * {cols}..][..{cols}].try_into().unwrap()")
                };
                let finish = |value: &str, index: &str| {
                    let with_bias = match bias {
                        Some(bias) => format!("({value} + {bias}[{index}])"),
                        None => value.to_string(),
                    };
                    write_expr(post, &[with_bias])
                };
                let _ = writeln!(
                    body,
                    "    let x: &[f32; {cols}] = (&{source}[..]).try_into().unwrap();"
                );
                let whole = rows - rows % together;
                let _ = writeln!(body, "    let mut r = 0;");
                let _ = writeln!(body, "    while r < {whole} {{");
                let rows_text: Vec<String> = (0..together)
                    .map(|offset| row(&format!("(r + {offset})")))
                    .collect();
                let _ = writeln!(
                    body,
                    "        let dots = {kernel}(x, [{}]);",
                    rows_text.join(", ")
                );
                for offset in 0..together {
                    let _ = writeln!(
                        body,
                        "        {target}[r + {offset}] = {};",
                        finish(&format!("dots[{offset}]"), &format!("r + {offset}"))
                    );
                }
                let _ = writeln!(body, "        r += {together};");
                let _ = writeln!(body, "    }}");
                if whole < *rows {
                    let _ = writeln!(body, "    for r in {whole}..{rows} {{");
                    let _ = writeln!(body, "        let [dot] = {kernel}_one(x, [{}]);", row("r"));
                    let _ = writeln!(body, "        {target}[r] = {};", finish("dot", "r"));
                    let _ = writeln!(body, "    }}");
                }
            }
        }
    }
    let _ = writeln!(out, "pub const INPUT: usize = {input_len};");
    let _ = writeln!(out, "pub const OUTPUT: usize = {output_len};");
    let _ = writeln!(out, "#[allow(unused_variables, clippy::all)]");
    let _ = writeln!(
        out,
        "pub fn run(state: &mut State, input: &[f32; {input_len}], output: &mut [f32; {output_len}]) {{"
    );
    out.push_str(&body);
    out.push_str("}\n");
    out
}
// ANCHOR_END: emit

/// 그래프를 소스로: 낮추고, 합치고(`options.fuse`), 적는다.
pub fn compile(graph: &Graph, options: &Options) -> Result<(Program, String)> {
    let mut program = lower(graph)?;
    if options.fuse {
        fuse(&mut program);
    }
    let source = emit(&program, options);
    Ok((program, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use badak_core::Tensor;

    /// x → (x + 1) × 2 → ReLU → 자르기(0, 3).
    fn chain() -> Graph {
        let mut graph = Graph::new("chain");
        let x = graph.value("x", &[4]);
        let one = graph.constant("one", Tensor::from_vec(vec![1.0_f32], &[1]).unwrap());
        let two = graph.constant("two", Tensor::from_vec(vec![2.0_f32; 4], &[4]).unwrap());
        let (a, b, c, d) = (
            graph.value("a", &[4]),
            graph.value("b", &[4]),
            graph.value("c", &[4]),
            graph.value("d", &[4]),
        );
        graph.node(
            Op::Add {
                activation: Activation::None,
            },
            &[x, one],
            &[a],
        );
        graph.node(
            Op::Mul {
                activation: Activation::None,
            },
            &[a, two],
            &[b],
        );
        graph.node(Op::Relu, &[b], &[c]);
        graph.node(Op::Clip { min: 0.0, max: 3.0 }, &[c], &[d]);
        graph.inputs = vec![x];
        graph.outputs = vec![d];
        graph
    }

    #[test]
    fn a_chain_lowers_to_one_loop_per_node_and_fuses_into_one() {
        let graph = chain();
        let mut program = lower(&graph).unwrap();
        assert_eq!(program.stages.len(), 4);
        fuse(&mut program);
        assert_eq!(program.stages.len(), 1);
        let Stage::Map { inputs, len, .. } = &program.stages[0] else {
            panic!("반복문이어야 한다");
        };
        assert_eq!((inputs.as_slice(), *len), (&[0_usize][..], 4));
        let source = emit(&program, &Options::default());
        // 가운데 값의 자리가 하나도 남지 않는다.
        assert!(source.contains("pub struct State {\n}"), "{source}");
        assert_eq!(source.matches("for i in 0..4").count(), 1);
    }

    #[test]
    fn a_value_read_twice_is_not_fused_away() {
        // a = x + 1, b = ReLU(a), c = a + b.
        let mut graph = Graph::new("square");
        let x = graph.value("x", &[2]);
        let one = graph.constant("one", Tensor::from_vec(vec![1.0_f32], &[1]).unwrap());
        let (a, b, c) = (
            graph.value("a", &[2]),
            graph.value("b", &[2]),
            graph.value("c", &[2]),
        );
        graph.node(
            Op::Add {
                activation: Activation::None,
            },
            &[x, one],
            &[a],
        );
        graph.node(Op::Relu, &[a], &[b]);
        // c = a + b: a를 읽는 마디가 둘이 된다. a는 합쳐 없애면 안 된다.
        graph.node(
            Op::Add {
                activation: Activation::None,
            },
            &[a, b],
            &[c],
        );
        graph.inputs = vec![x];
        graph.outputs = vec![c];
        let mut program = lower(&graph).unwrap();
        fuse(&mut program);
        assert_eq!(program.stages.len(), 2, "{:?}", program.stages);
        assert!(program.lengths.contains_key(&a));
    }

    #[test]
    fn unknown_operators_are_refused() {
        let mut graph = Graph::new("unknown");
        let x = graph.value("x", &[2]);
        let y = graph.value("y", &[2]);
        graph.node(Op::Identity, &[x], &[y]);
        graph.inputs = vec![x];
        graph.outputs = vec![y];
        assert!(matches!(lower(&graph), Err(Error::Unsupported(_))));
    }
}
