//! ONNX 그래프를 이 크레이트의 그래프로 옮긴다 (30장).
//!
//! 옮기는 일은 일부러 **기계적으로** 한다. 마디 하나를 보고 그 마디만
//! 옮기고, 이웃을 보지 않는다. 그래서 결과에는 군더더기가 많다. 그것을
//! 걷어 내는 일은 [`optimize`](mod@crate::optimize)가 한다.
//!
//! 군더더기의 대부분은 **축의 순서**에서 온다. ONNX의 사진은 `(장 수, 채널,
//! 높이, 너비)`(NCHW)이고 이 크레이트의 커널은 `(장 수, 높이, 너비, 채널)`
//! (NHWC)만 안다(27장). 그래서 합성곱처럼 축의 순서를 따지는 마디는 앞뒤에
//! 축을 바꾸는 마디를 하나씩 끼워 옮긴다. 필터도 축을 바꾸는 마디를 거쳐
//! 들어간다.

use std::collections::BTreeMap;

use badak_formats::onnx::{self, Attribute};

use crate::error::{Error, Result};
use crate::graph::{Activation, Graph, Op, Padding};
use crate::registry::Registry;

/// `(장 수, 채널, 높이, 너비)`를 `(장 수, 높이, 너비, 채널)`로 바꾸는 순서.
pub const TO_NHWC: [usize; 4] = [0, 2, 3, 1];
/// 그 반대.
pub const TO_NCHW: [usize; 4] = [0, 3, 1, 2];

/// 이보다 낮은 판(opset)의 파일은 받지 않는다. 그 아래에서는 `Clip`의 두
/// 끝이 입력이 아니라 속성이었다.
const OLDEST_OPSET: i64 = 11;

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

fn unsupported(node: &onnx::Node, what: &str) -> Error {
    Error::Unsupported(format!("{} 마디 {:?}: {what}", node.op_type, node.name))
}

/// 옮기는 동안 쥐고 있는 것들.
struct Importer {
    graph: Graph,
    /// ONNX의 값 이름에서 우리 값의 번호로.
    names: BTreeMap<String, usize>,
    /// 마디를 더할 때마다 출력의 형상을 정하는 데 쓴다 (32장). 축의 수에
    /// 따라 옮기는 방법이 달라지는 연산이 있어서 형상을 알아야 한다.
    registry: Registry,
}

impl Importer {
    fn lookup(&self, node: &onnx::Node, index: usize) -> Result<usize> {
        let name = node
            .inputs
            .get(index)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| unsupported(node, &format!("입력 {index}번이 없습니다")))?;
        self.names
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("값 {name:?}를 만드는 마디가 앞에 없습니다")))
    }

    /// 있을 수도 없을 수도 있는 입력.
    fn optional(&self, node: &onnx::Node, index: usize) -> Result<Option<usize>> {
        match node.inputs.get(index) {
            Some(name) if !name.is_empty() => self.lookup(node, index).map(Some),
            _ => Ok(None),
        }
    }

    /// 마디 하나를 더하고, 새로 만든 출력 값의 번호를 돌려준다. 커널이
    /// 있는 연산이면 출력의 형상도 정해 둔다.
    fn emit(&mut self, op: Op, inputs: &[usize], name: &str) -> usize {
        let shapes: Vec<&[usize]> = inputs
            .iter()
            .map(|&value| &*self.graph.values[value].shape)
            .collect();
        let shape = self
            .registry
            .resolve(&op)
            .and_then(|kernel| kernel.shapes(&shapes))
            .ok()
            .and_then(|mut shapes| shapes.pop())
            .unwrap_or_default();
        let output = self.graph.value(name, &shape);
        self.graph.node(op, inputs, &[output]);
        output
    }

    /// 값의 축의 수.
    fn rank(&self, value: usize) -> usize {
        self.graph.values[value].shape.len()
    }

    /// 값이 정해진 정수 입력을 읽는다. 없으면 `None`이다.
    fn integers(&self, node: &onnx::Node, index: usize) -> Result<Option<Vec<i32>>> {
        let Some(value) = self.optional(node, index)? else {
            return Ok(None);
        };
        let tensor = self.graph.values[value].constant.as_ref();
        match tensor.map(|tensor| tensor.to_vec::<i32>()) {
            Some(Ok(values)) => Ok(Some(values)),
            _ => Err(unsupported(
                node,
                &format!("입력 {index}번이 정수 상수가 아닙니다"),
            )),
        }
    }

    // ANCHOR: wrapped
    /// 축의 순서를 따지는 연산 하나를 옮긴다. 입력을 NHWC로 바꾸고, 연산을
    /// 하고, 결과를 NCHW로 되돌린다.
    fn wrapped(&mut self, op: Op, x: usize, rest: &[usize], name: &str) -> usize {
        let perm = TO_NHWC.to_vec();
        let x = self.emit(Op::Transpose { perm }, &[x], &format!("{name}/nhwc"));
        let mut inputs = vec![x];
        inputs.extend_from_slice(rest);
        let y = self.emit(op, &inputs, &format!("{name}/y"));
        let perm = TO_NCHW.to_vec();
        self.emit(Op::Transpose { perm }, &[y], name)
    }
    // ANCHOR_END: wrapped

    /// 값이 정해진 스칼라 입력을 읽는다. `Clip`의 두 끝이 이렇게 온다.
    fn scalar(&self, node: &onnx::Node, index: usize, default: f32) -> Result<f32> {
        let Some(value) = self.optional(node, index)? else {
            return Ok(default);
        };
        let tensor = self.graph.values[value].constant.as_ref();
        match tensor
            .map(|tensor| tensor.to_vec::<f32>())
            .transpose()?
            .as_deref()
        {
            Some(&[value]) => Ok(value),
            _ => Err(unsupported(node, "끝값이 상수 하나가 아닙니다")),
        }
    }

    // ANCHOR: node
    /// 마디 하나를 옮기고 출력 값의 번호를 돌려준다.
    fn node(&mut self, node: &onnx::Node) -> Result<usize> {
        let name = node.outputs.first().map_or("", String::as_str);
        let plain = Activation::None;
        Ok(match node.op_type.as_str() {
            // ANCHOR: conv
            "Conv" => {
                let x = self.lookup(node, 0)?;
                let weights = self.lookup(node, 1)?;
                let bias = self.optional(node, 2)?;
                let shape = self.graph.values[weights].shape.clone();
                let &[out_channels, per_group, _, _] = &shape[..] else {
                    return Err(unsupported(node, "필터의 축이 넷이 아닙니다"));
                };
                let group = usize::try_from(node.int("group", 1)?).unwrap_or(0);
                let (padding, stride) = window(node)?;
                let dilation = pair(node, "dilations", 1)?;
                // 필터: ONNX는 (출력 채널, 입력 채널, 높이, 너비)다. 축을
                // 바꾸는 마디를 거쳐 우리 순서로 넣는다. 값이 정해진 텐서를
                // 읽는 마디이므로 실행 전에 미리 계산할 수 있다(상수 접기).
                let (op, perm) = if group == 1 {
                    // (출력 채널, 높이, 너비, 입력 채널)
                    let op = Op::Conv2d {
                        padding,
                        stride,
                        dilation,
                        activation: plain,
                    };
                    (op, vec![0, 2, 3, 1])
                } else if per_group == 1 && out_channels % group == 0 {
                    // 묶음의 수가 입력 채널의 수와 같은 합성곱이 깊이별
                    // 합성곱이다. 우리 필터는 (1, 높이, 너비, 출력 채널)이다.
                    let op = Op::DepthwiseConv2d {
                        padding,
                        stride,
                        dilation,
                        multiplier: out_channels / group,
                        activation: plain,
                    };
                    (op, vec![1, 2, 3, 0])
                } else {
                    return Err(unsupported(node, "채널을 묶음으로 나눈 합성곱"));
                };
                let filter = self.emit(
                    Op::Transpose { perm },
                    &[weights],
                    &format!("{name}/filter"),
                );
                let mut rest = vec![filter];
                rest.extend(bias);
                self.wrapped(op, x, &rest, name)
            }
            // ANCHOR_END: conv
            "BatchNormalization" => {
                if node.int("training_mode", 0)? != 0 {
                    return Err(unsupported(node, "학습 상태의 배치 정규화"));
                }
                let x = self.lookup(node, 0)?;
                let rest = [
                    self.lookup(node, 1)?,
                    self.lookup(node, 2)?,
                    self.lookup(node, 3)?,
                    self.lookup(node, 4)?,
                ];
                let epsilon = node.float("epsilon", 1e-5)?;
                self.wrapped(Op::BatchNorm { epsilon }, x, &rest, name)
            }
            pool @ ("MaxPool" | "AveragePool") => {
                let x = self.lookup(node, 0)?;
                let (padding, stride) = window(node)?;
                let filter = pair(node, "kernel_shape", 0)?;
                if node.int("ceil_mode", 0)? != 0 || pair(node, "dilations", 1)? != [1, 1] {
                    return Err(unsupported(node, "올림하거나 칸을 띄운 풀링"));
                }
                let op = if pool == "MaxPool" {
                    Op::MaxPool2d {
                        padding,
                        stride,
                        filter,
                    }
                } else {
                    // 우리 평균 풀링은 덧댄 자리를 세지 않는다.
                    if node.int("count_include_pad", 0)? != 0 {
                        return Err(unsupported(node, "덧댄 자리까지 세는 평균"));
                    }
                    Op::AveragePool2d {
                        padding,
                        stride,
                        filter,
                    }
                };
                self.wrapped(op, x, &[], name)
            }
            "GlobalAveragePool" => {
                let x = self.lookup(node, 0)?;
                let op = Op::Mean {
                    axes: vec![1, 2],
                    keep_dims: true,
                };
                self.wrapped(op, x, &[], name)
            }
            // 원소마다 하는 연산은 축의 순서를 따지지 않는다.
            "Relu" => {
                let x = self.lookup(node, 0)?;
                self.emit(Op::Relu, &[x], name)
            }
            "Clip" => {
                let x = self.lookup(node, 0)?;
                let min = self.scalar(node, 1, f32::NEG_INFINITY)?;
                let max = self.scalar(node, 2, f32::INFINITY)?;
                self.emit(Op::Clip { min, max }, &[x], name)
            }
            binary @ ("Add" | "Mul") => {
                let inputs = [self.lookup(node, 0)?, self.lookup(node, 1)?];
                let op = if binary == "Add" {
                    Op::Add { activation: plain }
                } else {
                    Op::Mul { activation: plain }
                };
                self.emit(op, &inputs, name)
            }
            // 32장: 객체 검출 모델의 연산들.
            "Sigmoid" => {
                let x = self.lookup(node, 0)?;
                self.emit(Op::Sigmoid, &[x], name)
            }
            // ANCHOR: concat
            "Concat" => {
                let inputs = (0..node.inputs.len())
                    .map(|index| self.lookup(node, index))
                    .collect::<Result<Vec<_>>>()?;
                let rank = self.rank(inputs[0]);
                let axis = node.int("axis", 0)?;
                let axis = usize::try_from(if axis < 0 { axis + rank as i64 } else { axis })
                    .ok()
                    .filter(|&axis| axis < rank)
                    .ok_or_else(|| unsupported(node, "축이 형상을 벗어납니다"))?;
                if rank == 4 {
                    // 사진들이다. 하나씩 NHWC로 바꿔 이어 붙이고 되돌린다.
                    // 채널 축(1)은 NHWC에서 마지막 축(3)이다.
                    let sources: Vec<usize> = inputs
                        .iter()
                        .enumerate()
                        .map(|(index, &x)| {
                            let perm = TO_NHWC.to_vec();
                            self.emit(Op::Transpose { perm }, &[x], &format!("{name}/nhwc{index}"))
                        })
                        .collect();
                    let axis = TO_NCHW[axis];
                    let joined = self.emit(Op::Concat { axis }, &sources, &format!("{name}/y"));
                    let perm = TO_NCHW.to_vec();
                    self.emit(Op::Transpose { perm }, &[joined], name)
                } else {
                    self.emit(Op::Concat { axis }, &inputs, name)
                }
            }
            // ANCHOR_END: concat
            "Slice" => {
                let x = self.lookup(node, 0)?;
                let shape = self.graph.values[x].shape.clone();
                let starts = self.integers(node, 1)?.unwrap_or_default();
                let ends = self.integers(node, 2)?.unwrap_or_default();
                let axes = self.integers(node, 3)?;
                let steps = self.integers(node, 4)?;
                let wrap = shape.len() == 4;
                // 4차원이면 NHWC로 바꾼 뒤에 고른다. 범위도 그 순서로 적는다.
                let mut ranges: Vec<[usize; 3]> = (0..shape.len())
                    .map(|axis| [0, shape[if wrap { TO_NHWC[axis] } else { axis }], 1])
                    .collect();
                for (index, (&start, &end)) in starts.iter().zip(&ends).enumerate() {
                    let axis = axes.as_ref().map_or(index as i32, |axes| axes[index]);
                    let step = steps.as_ref().map_or(1, |steps| steps[index]);
                    let axis = usize::try_from(axis)
                        .ok()
                        .filter(|&axis| axis < shape.len())
                        .ok_or_else(|| unsupported(node, "뒤에서부터 센 축"))?;
                    let extent = shape[axis] as i64;
                    // 음수는 끝에서부터 센 것이고, 범위를 넘으면 끝으로 자른다.
                    let place = |value: i32| {
                        let value = i64::from(value);
                        (if value < 0 { value + extent } else { value }).clamp(0, extent) as usize
                    };
                    if step <= 0 {
                        return Err(unsupported(node, "거꾸로 고르는 Slice"));
                    }
                    let target = if wrap { TO_NCHW[axis] } else { axis };
                    ranges[target] = [place(start), place(end), step as usize];
                }
                if wrap {
                    self.wrapped(Op::Slice { ranges }, x, &[], name)
                } else {
                    self.emit(Op::Slice { ranges }, &[x], name)
                }
            }
            "Resize" => {
                let x = self.lookup(node, 0)?;
                let text = |key: &str| match node.attribute(key) {
                    Some(Attribute::String(value)) => value.as_str(),
                    _ => "",
                };
                // 정수 배로 키우는, 가장 가까운 픽셀을 쓰는 경우만 받는다.
                let plain = text("mode") == "nearest"
                    && text("coordinate_transformation_mode") == "asymmetric"
                    && text("nearest_mode") == "floor";
                let scales = self
                    .optional(node, 2)?
                    .and_then(|value| self.graph.values[value].constant.as_ref())
                    .and_then(|tensor| tensor.to_vec::<f32>().ok());
                let scale = match scales.as_deref() {
                    Some(&[1.0, 1.0, rows, columns])
                        if plain && rows.fract() == 0.0 && columns.fract() == 0.0 =>
                    {
                        [rows as usize, columns as usize]
                    }
                    _ => return Err(unsupported(node, "정수 배의 최근접 확대가 아닙니다")),
                };
                self.wrapped(Op::Resize { scale }, x, &[], name)
            }
            "Reshape" => {
                let x = self.lookup(node, 0)?;
                let source = self.graph.values[x].shape.clone();
                let wanted = self
                    .integers(node, 1)?
                    .ok_or_else(|| unsupported(node, "형상이 없습니다"))?;
                let total: usize = source.iter().product();
                // 0은 "입력의 그 축 그대로", -1은 "나머지 전부"다.
                let mut shape: Vec<usize> = wanted
                    .iter()
                    .enumerate()
                    .map(|(axis, &dim)| match dim {
                        0 => source.get(axis).copied().unwrap_or(1),
                        dim if dim > 0 => dim as usize,
                        _ => 1,
                    })
                    .collect();
                let known: usize = shape.iter().product();
                if let Some(free) = wanted.iter().position(|&dim| dim == -1)
                    && known > 0
                {
                    shape[free] = total / known;
                }
                self.emit(Op::Reshape { shape }, &[x], name)
            }
            "Identity" => {
                let x = self.lookup(node, 0)?;
                self.emit(Op::Identity, &[x], name)
            }
            "Flatten" => {
                let x = self.lookup(node, 0)?;
                let axis = usize::try_from(node.int("axis", 1)?)
                    .map_err(|_| unsupported(node, "뒤에서부터 센 축"))?;
                self.emit(Op::Flatten { axis }, &[x], name)
            }
            "Transpose" => {
                let x = self.lookup(node, 0)?;
                let perm = node
                    .sizes("perm")?
                    .ok_or_else(|| unsupported(node, "축의 순서가 없습니다"))?;
                self.emit(Op::Transpose { perm }, &[x], name)
            }
            // `Gemm`은 `alpha·A·B + beta·C`다. 선형 층이 이것으로 온다.
            // PyTorch의 가중치는 (출력, 입력)이어서 `B`를 뒤집으라는 표시
            // (`transB`)가 붙어 있고, 그것이 우리 선형 층의 순서다.
            "Gemm" => {
                let standard = node.float("alpha", 1.0)? == 1.0
                    && node.float("beta", 1.0)? == 1.0
                    && node.int("transA", 0)? == 0
                    && node.int("transB", 0)? == 1;
                if !standard {
                    return Err(unsupported(node, "선형 층의 모양이 아닌 Gemm"));
                }
                let mut inputs = vec![self.lookup(node, 0)?, self.lookup(node, 1)?];
                inputs.extend(self.optional(node, 2)?);
                let op = Op::FullyConnected {
                    activation: plain,
                    keep_num_dims: false,
                };
                self.emit(op, &inputs, name)
            }
            // 값이 정해진 텐서를 내놓는 마디. 마디로 두지 않고 값으로 옮긴다.
            "Constant" => match node.attribute("value") {
                Some(Attribute::Tensor(tensor)) => self.graph.constant(name, tensor.clone()),
                _ => return Err(unsupported(node, "텐서가 아닌 상수")),
            },
            // 모르는 연산도 그래프에는 담는다(26장).
            other => {
                let inputs = (0..node.inputs.len())
                    .map(|index| self.lookup(node, index))
                    .collect::<Result<Vec<_>>>()?;
                let op = Op::Custom {
                    name: other.to_string(),
                };
                self.emit(op, &inputs, name)
            }
        })
    }
    // ANCHOR_END: node
}

/// 길이가 2인 정수 속성. 없으면 둘 다 `default`다. `default`가 0이면 꼭
/// 있어야 한다.
fn pair(node: &onnx::Node, name: &str, default: usize) -> Result<[usize; 2]> {
    match node.sizes(name)?.as_deref() {
        Some(&[rows, columns]) => Ok([rows, columns]),
        None if default != 0 => Ok([default; 2]),
        _ => Err(unsupported(node, &format!("속성 {name}이 둘이 아닙니다"))),
    }
}

// ANCHOR: window
/// 합성곱과 풀링이 함께 쓰는 속성: 덧댈 길이와 보폭.
///
/// ONNX는 덧댈 길이를 `[위, 왼쪽, 아래, 오른쪽]`으로 적는다. 시작 쪽을 모두
/// 적은 다음 끝 쪽을 적는 순서다.
fn window(node: &onnx::Node) -> Result<(Padding, [usize; 2])> {
    if let Some(Attribute::String(rule)) = node.attribute("auto_pad")
        && rule != "NOTSET"
    {
        return Err(unsupported(node, "덧대는 규칙을 이름으로 적은 마디"));
    }
    let padding = match node.sizes("pads")?.as_deref() {
        None | Some(&[0, 0, 0, 0]) => Padding::Valid,
        Some(&[top, left, bottom, right]) => Padding::Explicit([[top, bottom], [left, right]]),
        Some(_) => return Err(unsupported(node, "덧댈 길이가 넷이 아닙니다")),
    };
    Ok((padding, pair(node, "strides", 1)?))
}
// ANCHOR_END: window

impl Graph {
    // ANCHOR: from_onnx
    /// ONNX 파일의 그래프를 옮긴다. 계산되는 값의 형상은 비어 있다.
    /// [`infer_shapes`](Self::infer_shapes)로 채운다.
    pub fn from_onnx(model: &onnx::Model) -> Result<Self> {
        if model.opset < OLDEST_OPSET {
            return Err(Error::Unsupported(format!(
                "ONNX 연산 {}판 ({OLDEST_OPSET}판부터 읽습니다)",
                model.opset
            )));
        }
        let source = &model.graph;
        let mut importer = Importer {
            graph: Self::new(&source.name),
            names: BTreeMap::new(),
            registry: Registry::builtin(),
        };
        for (name, tensor) in &source.initializers {
            let value = importer.graph.constant(name, tensor.clone());
            importer.names.insert(name.clone(), value);
        }
        for input in &source.inputs {
            // 예전 파일은 가중치도 입력으로 적었다. 그런 것은 입력이 아니다.
            if importer.names.contains_key(&input.name) {
                continue;
            }
            if input.elem_type != onnx::data_type::FLOAT {
                return Err(Error::Unsupported(format!(
                    "자료형이 {}번인 입력 {:?}",
                    input.elem_type, input.name
                )));
            }
            // 실행할 때 정해지는 축(보통 장 수)은 1로 둔다. 해석기를 만들 때
            // 바꿀 수 있다.
            let shape: Vec<usize> = input.shape.iter().map(|dim| dim.unwrap_or(1)).collect();
            let value = importer.graph.value(&input.name, &shape);
            importer.graph.inputs.push(value);
            importer.names.insert(input.name.clone(), value);
        }
        for node in &source.nodes {
            if !node.domain.is_empty() && node.domain != "ai.onnx" {
                return Err(unsupported(node, "표준 연산이 아닙니다"));
            }
            let [output] = &node.outputs[..] else {
                return Err(unsupported(node, "출력이 하나가 아닙니다"));
            };
            let value = importer.node(node)?;
            if importer.names.insert(output.clone(), value).is_some() {
                return Err(invalid(format!("값 {output:?}를 만드는 마디가 둘입니다")));
            }
        }
        for output in &source.outputs {
            let value = importer.names.get(&output.name).ok_or_else(|| {
                invalid(format!(
                    "그래프의 출력 {:?}를 만드는 마디가 없습니다",
                    output.name
                ))
            })?;
            importer.graph.outputs.push(*value);
        }
        Ok(importer.graph)
    }
    // ANCHOR_END: from_onnx
}
