//! 계산 그래프: 값들과, 값을 받아 값을 내는 마디들.

use std::collections::BTreeSet;

use badak_core::{DType, Tensor};
use badak_formats::tflite::{self, builtin};

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
        });
        self.values.len() - 1
    }

    /// 값이 정해진 텐서를 하나 더하고 번호를 돌려준다.
    pub fn constant(&mut self, name: &str, tensor: Tensor) -> usize {
        self.values.push(Value {
            name: name.to_string(),
            shape: tensor.shape().to_vec(),
            constant: Some(tensor),
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
            let is_float = info.dtype == tflite::TensorType::Float32
                && constant.as_ref().is_none_or(|t| t.dtype() == DType::F32);
            if !is_float {
                return Err(Error::Unsupported(format!(
                    "f32가 아닌 텐서 {:?} ({:?})",
                    info.name, info.dtype
                )));
            }
            graph.values.push(Value {
                name: info.name.clone(),
                shape: info.shape.clone(),
                constant,
            });
        }
        graph.inputs = source.inputs.clone();
        graph.outputs = source.outputs.clone();

        for operator in &source.operators {
            let code = &model.operator_codes[operator.code];
            let op = convert(code, operator.options)?;
            // 생략한 입력은 맨 뒤에만 올 수 있다고 본다.
            let mut inputs = operator.inputs.clone();
            while inputs.last() == Some(&None) {
                inputs.pop();
            }
            let inputs = inputs
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    Error::Unsupported(format!("{}의 가운데 입력을 생략한 그래프", op.name()))
                })?;
            graph.node(op, &inputs, &operator.outputs);
        }
        Ok(graph)
    }
    // ANCHOR_END: from_tflite

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

fn convert(code: &tflite::OperatorCode, options: tflite::Options) -> Result<Op> {
    let fused = |activation: tflite::Activation| match activation {
        tflite::Activation::None => Ok(Activation::None),
        tflite::Activation::Relu => Ok(Activation::Relu),
        tflite::Activation::ReluN1To1 => Ok(Activation::ReluN1To1),
        tflite::Activation::Relu6 => Ok(Activation::Relu6),
        tflite::Activation::Tanh => Ok(Activation::Tanh),
        tflite::Activation::Other(code) => Err(Error::Unsupported(format!("활성화 함수 {code}번"))),
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
