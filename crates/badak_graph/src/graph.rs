//! 계산 그래프: 값들과, 값을 받아 값을 내는 마디들.

use std::collections::BTreeSet;

use badak_core::{DType, Tensor};
use badak_formats::tflite::{self, builtin};

pub use badak_formats::tflite::{Padding, Quantization};

use crate::error::{Error, Result};

// ANCHOR: types
/// 연산의 결과에 곧바로 이어 붙이는 활성화 함수.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// 없음.
    None,
    /// `max(x, 0)`
    Relu,
    /// `-1`과 `1` 사이로 자른다.
    ReluN1To1,
    /// `0`과 `6` 사이로 자른다.
    Relu6,
    /// 쌍곡 탄젠트.
    Tanh,
}

/// 마디가 하는 연산.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// 선형 층. 입력은 `(x, 가중치)` 또는 `(x, 가중치, 치우침)`이다.
    FullyConnected {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
        /// 입력의 앞쪽 차원들을 결과에 그대로 둘지.
        keep_num_dims: bool,
    },
    /// 원소끼리 더한다.
    Add {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 원소끼리 곱한다.
    Mul {
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// ReLU.
    Relu,
    /// 합성곱 (27장). 입력은 `(x, 필터)` 또는 `(x, 필터, 치우침)`이다. `x`는
    /// `(장 수, 높이, 너비, 채널)`, 필터는 `(출력 채널, 높이, 너비, 입력
    /// 채널)`이다.
    Conv2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 보폭. (세로, 가로).
        stride: [usize; 2],
        /// 창의 칸 사이의 간격. (세로, 가로).
        dilation: [usize; 2],
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 채널마다 따로 하는 합성곱 (27장). 필터는 `(1, 높이, 너비, 출력
    /// 채널)`이다.
    DepthwiseConv2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 보폭. (세로, 가로).
        stride: [usize; 2],
        /// 창의 칸 사이의 간격. (세로, 가로).
        dilation: [usize; 2],
        /// 입력 채널 하나가 내는 출력 채널의 수.
        multiplier: usize,
        /// 결과에 이어 붙이는 활성화 함수.
        activation: Activation,
    },
    /// 창 안의 최댓값 (27장).
    MaxPool2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 보폭. (세로, 가로).
        stride: [usize; 2],
        /// 창의 크기. (세로, 가로).
        filter: [usize; 2],
    },
    /// 창 안의 평균 (27장).
    AveragePool2d {
        /// 가장자리를 다루는 방법.
        padding: Padding,
        /// 보폭. (세로, 가로).
        stride: [usize; 2],
        /// 창의 크기. (세로, 가로).
        filter: [usize; 2],
    },
    /// 가장자리에 값을 덧댄다 (27장).
    Pad {
        /// 축마다 앞과 뒤에 덧댈 길이.
        amounts: Vec<[usize; 2]>,
        /// 덧대는 값.
        value: f32,
    },
    /// 축들을 따라 평균을 낸다 (27장).
    Mean {
        /// 평균을 낼 축들.
        axes: Vec<usize>,
        /// 줄인 축을 길이 1로 남길지.
        keep_dims: bool,
    },
    /// 축의 순서를 바꾼다 (27장). 결과의 `i`번째 축이 입력의 `perm[i]`번째
    /// 축이다.
    Transpose {
        /// 축의 새 순서.
        perm: Vec<usize>,
    },
    /// 마지막 축을 따라 소프트맥스.
    Softmax {
        /// 점수에 곱하는 값.
        beta: f32,
    },
    /// 이 크레이트가 모르는 연산. 이름만 안다.
    Custom {
        /// 연산의 이름.
        name: String,
    },
}

impl Op {
    /// 연산의 이름. 레지스트리가 이 이름으로 커널을 찾는다.
    pub fn name(&self) -> &str {
        match self {
            Self::FullyConnected { .. } => "FULLY_CONNECTED",
            Self::Add { .. } => "ADD",
            Self::Mul { .. } => "MUL",
            Self::Relu => "RELU",
            Self::Conv2d { .. } => "CONV_2D",
            Self::DepthwiseConv2d { .. } => "DEPTHWISE_CONV_2D",
            Self::MaxPool2d { .. } => "MAX_POOL_2D",
            Self::AveragePool2d { .. } => "AVERAGE_POOL_2D",
            Self::Pad { .. } => "PAD",
            Self::Mean { .. } => "MEAN",
            Self::Transpose { .. } => "TRANSPOSE",
            Self::Softmax { .. } => "SOFTMAX",
            Self::Custom { name } => name,
        }
    }
}

/// 값 하나. 텐서가 놓일 자리다.
#[derive(Debug, Clone)]
pub struct Value {
    /// 이름.
    pub name: String,
    /// 형상. 실행 전에 입력의 형상을 바꾸면 계산되는 값들의 형상은 다시
    /// 정해진다.
    pub shape: Vec<usize>,
    /// 가중치처럼 처음부터 값이 정해져 있으면 그 값.
    pub constant: Option<Tensor>,
    /// 8비트 정수로 계산되는 값이면 실수로 되돌리는 방법 (29장). `f32` 값은
    /// `None`이다.
    pub quantization: Option<Quantization>,
}

/// 마디 하나. 값들을 받아 값들을 낸다.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// 연산.
    pub op: Op,
    /// 입력 값들의 번호.
    pub inputs: Vec<usize>,
    /// 출력 값들의 번호.
    pub outputs: Vec<usize>,
}

/// 계산 그래프.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// 이름.
    pub name: String,
    /// 값들.
    pub values: Vec<Value>,
    /// 마디들.
    pub nodes: Vec<Node>,
    /// 밖에서 넣어 주는 값들의 번호.
    pub inputs: Vec<usize>,
    /// 결과로 내놓는 값들의 번호.
    pub outputs: Vec<usize>,
}
// ANCHOR_END: types

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

impl Graph {
    /// 빈 그래프를 만든다.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Self::default()
        }
    }

    /// 계산될 값을 하나 더하고 번호를 돌려준다.
    pub fn value(&mut self, name: &str, shape: &[usize]) -> usize {
        self.values.push(Value {
            name: name.to_string(),
            shape: shape.to_vec(),
            constant: None,
            quantization: None,
        });
        self.values.len() - 1
    }

    /// 값이 정해진 텐서를 하나 더하고 번호를 돌려준다.
    pub fn constant(&mut self, name: &str, tensor: Tensor) -> usize {
        self.values.push(Value {
            name: name.to_string(),
            shape: tensor.shape().to_vec(),
            constant: Some(tensor),
            quantization: None,
        });
        self.values.len() - 1
    }

    /// 마디를 하나 더한다.
    pub fn node(&mut self, op: Op, inputs: &[usize], outputs: &[usize]) {
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        });
    }

    // ANCHOR: from_tflite
    /// `.tflite` 파일의 첫 부분 그래프를 옮긴다.
    pub fn from_tflite(model: &tflite::Model) -> Result<Self> {
        let source = &model.subgraphs[0];
        let mut graph = Self::new(&source.name);
        for info in &source.tensors {
            let constant = model.constant(info)?;
            // 계산되는 값은 f32만 받는다. 값이 정해진 텐서는 정수여도 된다.
            // 덧댈 길이나 축의 번호처럼 연산의 설정을 담은 것들이다(27장).
            // 29장: 8비트 정수로 양자화한 값도 받는다. 눈금과 영점이 있어야
            // 한다.
            let quantized = info.dtype == tflite::TensorType::Int8 && info.quantization.is_some();
            let supported = match &constant {
                Some(tensor) => matches!(tensor.dtype(), DType::F32 | DType::I32) || quantized,
                None => info.dtype == tflite::TensorType::Float32 || quantized,
            };
            if !supported {
                return Err(Error::Unsupported(format!(
                    "자료형이 {:?}인 텐서 {:?}",
                    info.dtype, info.name
                )));
            }
            graph.values.push(Value {
                name: info.name.clone(),
                shape: info.shape.clone(),
                constant,
                quantization: info.quantization.clone(),
            });
        }
        graph.inputs = source.inputs.clone();
        graph.outputs = source.outputs.clone();

        for operator in &source.operators {
            let code = &model.operator_codes[operator.code];
            // 생략한 입력은 맨 뒤에만 올 수 있다고 본다.
            let mut inputs = operator.inputs.clone();
            while inputs.last() == Some(&None) {
                inputs.pop();
            }
            let mut inputs = inputs
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "연산 {}번의 가운데 입력을 생략한 그래프",
                        code.builtin
                    ))
                })?;
            let op = convert(code, operator.options, &graph, &mut inputs)?;
            graph.node(op, &inputs, &operator.outputs);
        }
        Ok(graph)
    }
    // ANCHOR_END: from_tflite

    /// 8비트 정수로 계산하는 그래프인지. 입력이 양자화되어 있으면 그렇다.
    pub fn is_quantized(&self) -> bool {
        self.inputs
            .iter()
            .any(|&input| self.values[input].quantization.is_some())
    }

    // ANCHOR: order
    /// 마디들을 실행할 순서를 정한다.
    ///
    /// 마디는 자기 입력을 만드는 마디들이 모두 끝난 뒤에 실행할 수 있다.
    /// 실행할 수 있게 된 마디 가운데 번호가 가장 작은 것을 고르는 일을
    /// 되풀이한다(위상 정렬). 그래프가 올바른지도 여기서 확인한다.
    pub fn order(&self) -> Result<Vec<usize>> {
        let count = self.values.len();
        let check = |value: usize, what: &str| {
            (value < count)
                .then_some(())
                .ok_or_else(|| invalid(format!("{what}이 없는 값 {value}를 가리킵니다")))
        };
        // 값마다 그 값을 만드는 마디.
        let mut producer = vec![None; count];
        for (index, node) in self.nodes.iter().enumerate() {
            for &output in &node.outputs {
                check(output, "출력")?;
                let name = &self.values[output].name;
                if self.values[output].constant.is_some() || self.inputs.contains(&output) {
                    return Err(invalid(format!(
                        "마디 {index}가 이미 값이 정해진 {name:?}에 쓰려 합니다"
                    )));
                }
                if producer[output].replace(index).is_some() {
                    return Err(invalid(format!("값 {name:?}를 만드는 마디가 둘입니다")));
                }
            }
        }
        for &value in self.inputs.iter().chain(&self.outputs) {
            check(value, "그래프의 입출력")?;
        }

        // 마디마다 아직 끝나지 않은 앞 마디의 수, 그리고 뒤에 오는 마디들.
        let mut waiting = vec![0_usize; self.nodes.len()];
        let mut followers = vec![Vec::new(); self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for &input in &node.inputs {
                check(input, "입력")?;
                let value = &self.values[input];
                match producer[input] {
                    Some(before) => {
                        waiting[index] += 1;
                        followers[before].push(index);
                    }
                    None if value.constant.is_some() || self.inputs.contains(&input) => {}
                    None => {
                        return Err(invalid(format!(
                            "값 {:?}를 만드는 마디가 없습니다",
                            value.name
                        )));
                    }
                }
            }
        }
        let mut ready: BTreeSet<usize> = (0..self.nodes.len())
            .filter(|&index| waiting[index] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(index) = ready.pop_first() {
            order.push(index);
            for &follower in &followers[index] {
                waiting[follower] -= 1;
                if waiting[follower] == 0 {
                    ready.insert(follower);
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(invalid("마디들이 서로를 기다리며 돌고 있습니다"));
        }
        Ok(order)
    }
    // ANCHOR_END: order
}

// ANCHOR: parameters
/// 연산의 설정을 담은 정수 텐서를 읽는다. `.tflite`는 덧댈 길이나 축의
/// 번호를 설정 표가 아니라 입력 텐서로 준다. 값이 파일에 정해져 있을 때만
/// 받는다. 그래야 실행 전에 형상을 정할 수 있다.
fn parameter(graph: &Graph, value: usize, what: &str) -> Result<Vec<i32>> {
    let tensor = graph.values[value]
        .constant
        .as_ref()
        .ok_or_else(|| Error::Unsupported(format!("{what}가 실행 중에 정해지는 그래프")))?;
    Ok(tensor.to_vec::<i32>()?)
}

/// 정수들을 축의 번호나 길이로 바꾼다. 음수는 받지 않는다.
fn indices(values: &[i32], what: &str) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| invalid(format!("{what}에 음수가 있습니다")))
        })
        .collect()
}
// ANCHOR_END: parameters

/// 파일의 연산 하나를 `Op`로 바꾼다. 설정을 담은 입력 텐서는 `Op` 안으로
/// 옮기고 `inputs`에서 뺀다.
fn convert(
    code: &tflite::OperatorCode,
    options: tflite::Options,
    graph: &Graph,
    inputs: &mut Vec<usize>,
) -> Result<Op> {
    let fused = |activation: tflite::Activation| match activation {
        tflite::Activation::None => Ok(Activation::None),
        tflite::Activation::Relu => Ok(Activation::Relu),
        tflite::Activation::ReluN1To1 => Ok(Activation::ReluN1To1),
        tflite::Activation::Relu6 => Ok(Activation::Relu6),
        tflite::Activation::Tanh => Ok(Activation::Tanh),
        tflite::Activation::Other(code) => Err(Error::Unsupported(format!("활성화 함수 {code}번"))),
    };
    let arity = |inputs: &Vec<usize>, count: usize, name: &str| {
        (inputs.len() == count)
            .then_some(())
            .ok_or_else(|| invalid(format!("{name}의 입력이 {}개입니다", inputs.len())))
    };
    Ok(match (code.builtin, options) {
        (
            builtin::FULLY_CONNECTED,
            tflite::Options::FullyConnected {
                activation,
                keep_num_dims,
            },
        ) => Op::FullyConnected {
            activation: fused(activation)?,
            keep_num_dims,
        },
        (builtin::FULLY_CONNECTED, tflite::Options::None) => Op::FullyConnected {
            activation: Activation::None,
            keep_num_dims: false,
        },
        (builtin::ADD, tflite::Options::Add { activation }) => Op::Add {
            activation: fused(activation)?,
        },
        (builtin::ADD, tflite::Options::None) => Op::Add {
            activation: Activation::None,
        },
        (builtin::MUL, tflite::Options::Mul { activation }) => Op::Mul {
            activation: fused(activation)?,
        },
        (builtin::MUL, tflite::Options::None) => Op::Mul {
            activation: Activation::None,
        },
        (builtin::RELU, _) => Op::Relu,
        (builtin::SOFTMAX, tflite::Options::Softmax { beta }) => Op::Softmax { beta },
        (
            builtin::CONV_2D,
            tflite::Options::Conv2d {
                padding,
                stride,
                dilation,
                activation,
            },
        ) => Op::Conv2d {
            padding,
            stride,
            dilation,
            activation: fused(activation)?,
        },
        (
            builtin::DEPTHWISE_CONV_2D,
            tflite::Options::DepthwiseConv2d {
                padding,
                stride,
                dilation,
                multiplier,
                activation,
            },
        ) => Op::DepthwiseConv2d {
            padding,
            stride,
            dilation,
            multiplier,
            activation: fused(activation)?,
        },
        (
            pool @ (builtin::MAX_POOL_2D | builtin::AVERAGE_POOL_2D),
            tflite::Options::Pool2d {
                padding,
                stride,
                filter,
                activation,
            },
        ) => {
            if fused(activation)? != Activation::None {
                return Err(Error::Unsupported("활성화 함수가 붙은 풀링".into()));
            }
            if pool == builtin::MAX_POOL_2D {
                Op::MaxPool2d {
                    padding,
                    stride,
                    filter,
                }
            } else {
                Op::AveragePool2d {
                    padding,
                    stride,
                    filter,
                }
            }
        }
        (pad @ (builtin::PAD | builtin::PADV2), _) => {
            arity(inputs, if pad == builtin::PAD { 2 } else { 3 }, "PAD")?;
            // PADV2는 덧댈 값을 셋째 입력으로 받는다. PAD는 0을 덧댄다.
            let value = if pad == builtin::PADV2 {
                let tensor = graph.values[inputs[2]].constant.as_ref();
                let values = tensor.map(|tensor| tensor.to_vec::<f32>()).transpose()?;
                match values.as_deref() {
                    Some(&[value]) => value,
                    _ => {
                        return Err(Error::Unsupported(
                            "덧댈 값이 상수 하나가 아닌 PADV2".into(),
                        ));
                    }
                }
            } else {
                0.0
            };
            let amounts = indices(&parameter(graph, inputs[1], "덧댈 길이")?, "덧댈 길이")?;
            let (pairs, rest) = amounts.as_chunks::<2>();
            if !rest.is_empty() {
                return Err(invalid("덧댈 길이가 앞뒤의 쌍이 아닙니다"));
            }
            inputs.truncate(1);
            Op::Pad {
                amounts: pairs.to_vec(),
                value,
            }
        }
        (builtin::MEAN, tflite::Options::Reducer { keep_dims }) => {
            arity(inputs, 2, "MEAN")?;
            let axes = indices(&parameter(graph, inputs[1], "평균을 낼 축")?, "축")?;
            inputs.truncate(1);
            Op::Mean { axes, keep_dims }
        }
        (builtin::TRANSPOSE, _) => {
            arity(inputs, 2, "TRANSPOSE")?;
            let perm = indices(&parameter(graph, inputs[1], "축의 순서")?, "축의 순서")?;
            inputs.truncate(1);
            Op::Transpose { perm }
        }
        (builtin::CUSTOM, _) => Op::Custom {
            name: code.custom.clone().unwrap_or_default(),
        },
        // 모르는 연산도 그래프에는 담는다. 실행할 수 있는지는 레지스트리가
        // 정한다.
        (other, _) => Op::Custom {
            name: format!("BUILTIN_{other}"),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relu_chain() -> Graph {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[4]);
        let a = graph.value("a", &[4]);
        let b = graph.value("b", &[4]);
        graph.inputs = vec![x];
        graph.outputs = vec![b];
        // 일부러 뒤의 마디를 먼저 적는다.
        graph.node(Op::Relu, &[a], &[b]);
        graph.node(Op::Relu, &[x], &[a]);
        graph
    }

    #[test]
    fn orders_nodes_by_dependency() {
        assert_eq!(relu_chain().order().unwrap(), [1, 0]);
    }

    #[test]
    fn keeps_file_order_when_it_is_already_valid() {
        let mut graph = Graph::new("시험");
        let x = graph.value("x", &[4]);
        let a = graph.value("a", &[4]);
        let b = graph.value("b", &[4]);
        let c = graph.value("c", &[4]);
        graph.inputs = vec![x];
        graph.outputs = vec![c];
        graph.node(Op::Relu, &[x], &[a]);
        graph.node(Op::Relu, &[x], &[b]);
        graph.node(
            Op::Add {
                activation: Activation::None,
            },
            &[a, b],
            &[c],
        );
        assert_eq!(graph.order().unwrap(), [0, 1, 2]);
    }

    #[test]
    fn rejects_broken_graphs() {
        // 돌고 도는 길.
        let mut cycle = relu_chain();
        cycle.nodes[1].inputs = vec![2];
        assert!(cycle.order().is_err());
        // 자기 출력을 입력으로 받는 마디.
        let mut own = relu_chain();
        own.nodes[1].inputs = vec![1];
        assert!(own.order().is_err());
        // 아무도 만들지 않는 값.
        let mut orphan = relu_chain();
        orphan.inputs.clear();
        assert!(orphan.order().is_err());
        // 두 마디가 같은 값을 만든다.
        let mut twice = relu_chain();
        twice.nodes[0].outputs = vec![1];
        assert!(twice.order().is_err());
        // 없는 값.
        let mut missing = relu_chain();
        missing.nodes[0].inputs = vec![9];
        assert!(missing.order().is_err());
        // 입력에 쓰려는 마디.
        let mut overwrite = relu_chain();
        overwrite.nodes[0].outputs = vec![0];
        assert!(overwrite.order().is_err());
    }
}
