//! 연산을 실제로 계산하는 커널들.
//!
//! 커널은 텐서를 만들지 않는다. 해석기가 미리 정해 둔 자리(슬라이스)를 받아
//! 거기에 결과를 적는다. 받은 자리에는 앞서 다른 값이 쓰던 내용이 남아
//! 있을 수 있으므로, 커널은 출력의 모든 원소를 덮어써야 한다.

use badak_core::{prob, simd};

use crate::error::{Error, Result};
use crate::graph::{Activation, Op};

// ANCHOR: kernel
/// 읽기만 하는 값 하나. 형상과 원소들이다.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    /// 형상.
    pub shape: &'a [usize],
    /// 원소들. 행 우선으로 놓여 있다.
    pub data: &'a [f32],
}

impl View<'_> {
    /// 빈 값. 자리를 채워 둘 때 쓴다.
    pub const EMPTY: Self = Self {
        shape: &[],
        data: &[],
    };
}

/// 결과를 적을 자리 하나.
#[derive(Debug)]
pub struct ViewMut<'a> {
    /// 형상.
    pub shape: &'a [usize],
    /// 원소들이 놓일 자리.
    pub data: &'a mut [f32],
}

impl ViewMut<'_> {
    /// 빈 자리.
    pub fn empty() -> Self {
        Self {
            shape: &[],
            data: &mut [],
        }
    }
}

/// 연산 하나를 계산하는 것.
///
/// 일을 두 단계로 나눈다. [`shapes`](Self::shapes)는 실행 전에 한 번
/// 불리고, 입력의 형상만 보고 출력의 형상을 정한다. 형상이 맞지 않으면
/// 여기서 오류를 낸다. [`run`](Self::run)은 추론할 때마다 불린다.
pub trait Kernel: Send + Sync {
    /// 입력들의 형상에서 출력들의 형상을 정한다.
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>>;

    /// 입력을 읽어 출력에 결과를 적는다.
    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()>;
}
// ANCHOR_END: kernel

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

/// 입력의 수가 맞는지 본다.
fn expect_inputs(
    name: &str,
    inputs: usize,
    allowed: std::ops::RangeInclusive<usize>,
) -> Result<()> {
    if allowed.contains(&inputs) {
        Ok(())
    } else {
        Err(invalid(format!("{name}의 입력이 {inputs}개입니다")))
    }
}

// ANCHOR: activation
/// 활성화 함수를 제자리에서 적용한다.
fn activate(activation: Activation, values: &mut [f32]) {
    match activation {
        Activation::None => {}
        Activation::Relu => values.iter_mut().for_each(|v| *v = v.max(0.0)),
        Activation::ReluN1To1 => values.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0)),
        Activation::Relu6 => values.iter_mut().for_each(|v| *v = v.clamp(0.0, 6.0)),
        Activation::Tanh => values.iter_mut().for_each(|v| *v = v.tanh()),
    }
}
// ANCHOR_END: activation

// ANCHOR: fully_connected
/// 선형 층 `y = act(x Wᵀ + b)`.
pub struct FullyConnected {
    activation: Activation,
    keep_num_dims: bool,
}

impl Kernel for FullyConnected {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        expect_inputs("FULLY_CONNECTED", inputs.len(), 2..=3)?;
        let &[rows, cols] = inputs[1] else {
            return Err(invalid("FULLY_CONNECTED의 가중치가 2차원이 아닙니다"));
        };
        let len: usize = inputs[0].iter().product();
        if cols == 0 || !len.is_multiple_of(cols) {
            return Err(invalid(format!(
                "FULLY_CONNECTED의 입력 {:?}를 길이 {cols}인 벡터들로 나눌 수 없습니다",
                inputs[0]
            )));
        }
        if let Some(bias) = inputs.get(2)
            && *bias != [rows]
        {
            return Err(invalid(format!(
                "FULLY_CONNECTED의 치우침 {bias:?}가 ({rows})이 아닙니다"
            )));
        }
        // 입력은 길이 cols인 벡터들의 묶음으로 본다. 앞쪽 차원들을 그대로
        // 둘 수도 있고, 하나로 합칠 수도 있다.
        let shape = match (self.keep_num_dims, inputs[0].split_last()) {
            (true, Some((&last, leading))) if last == cols => {
                let mut shape = leading.to_vec();
                shape.push(rows);
                shape
            }
            (true, _) => {
                return Err(invalid(
                    "FULLY_CONNECTED가 차원을 그대로 두려면 입력의 마지막 차원이 가중치와 맞아야 합니다",
                ));
            }
            (false, _) => vec![len / cols, rows],
        };
        Ok(vec![shape])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (x, w) = (inputs[0].data, inputs[1].data);
        let &[rows, cols] = inputs[1].shape else {
            unreachable!("shapes가 확인했다")
        };
        let y = &mut *outputs[0].data;
        // 17~18장의 커널을 그대로 쓴다.
        simd::linear(x, x.len() / cols, w, rows, cols, y);
        if let Some(bias) = inputs.get(2) {
            for row in y.chunks_exact_mut(rows) {
                for (value, &b) in row.iter_mut().zip(bias.data) {
                    *value += b;
                }
            }
        }
        activate(self.activation, y);
        Ok(())
    }
}
// ANCHOR_END: fully_connected

// ANCHOR: binary
/// 원소끼리 하는 계산의 종류.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arithmetic {
    Add,
    Mul,
}

/// 형상이 같은 두 텐서의 원소끼리 계산한다.
pub struct Binary {
    arithmetic: Arithmetic,
    activation: Activation,
}

impl Binary {
    fn name(&self) -> &'static str {
        match self.arithmetic {
            Arithmetic::Add => "ADD",
            Arithmetic::Mul => "MUL",
        }
    }
}

/// `y[i] = f(a[i], b[i])`. `f`가 무엇인지 컴파일할 때 정해지므로 반복문이
/// 벡터 명령으로 바뀐다(17.2절).
fn zip(y: &mut [f32], a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) {
    for ((value, &a), &b) in y.iter_mut().zip(a).zip(b) {
        *value = f(a, b);
    }
}

impl Kernel for Binary {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        expect_inputs(self.name(), inputs.len(), 2..=2)?;
        if inputs[0] != inputs[1] {
            return Err(Error::Unsupported(format!(
                "{}에서 형상이 다른 입력 {:?}와 {:?} (브로드캐스팅)",
                self.name(),
                inputs[0],
                inputs[1]
            )));
        }
        Ok(vec![inputs[0].to_vec()])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let y = &mut *outputs[0].data;
        let (a, b) = (inputs[0].data, inputs[1].data);
        // 어떤 계산인지는 반복문 밖에서 한 번만 고른다.
        match self.arithmetic {
            Arithmetic::Add => zip(y, a, b, |a, b| a + b),
            Arithmetic::Mul => zip(y, a, b, |a, b| a * b),
        }
        activate(self.activation, y);
        Ok(())
    }
}
// ANCHOR_END: binary

/// ReLU.
pub struct Relu;

impl Kernel for Relu {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        expect_inputs("RELU", inputs.len(), 1..=1)?;
        Ok(vec![inputs[0].to_vec()])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        for (value, &x) in outputs[0].data.iter_mut().zip(inputs[0].data) {
            *value = x.max(0.0);
        }
        Ok(())
    }
}

// ANCHOR: softmax
/// 마지막 축을 따라 소프트맥스.
pub struct Softmax {
    beta: f32,
}

impl Kernel for Softmax {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        expect_inputs("SOFTMAX", inputs.len(), 1..=1)?;
        if inputs[0].is_empty() {
            return Err(invalid("SOFTMAX의 입력에 축이 없습니다"));
        }
        Ok(vec![inputs[0].to_vec()])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let width = *inputs[0].shape.last().expect("shapes가 확인했다");
        if width == 0 {
            return Ok(());
        }
        let y = &mut *outputs[0].data;
        for (value, &x) in y.iter_mut().zip(inputs[0].data) {
            *value = self.beta * x;
        }
        // 5장의 소프트맥스를 줄마다 제자리에서 적용한다.
        for row in y.chunks_exact_mut(width) {
            prob::softmax_in_place(row);
        }
        Ok(())
    }
}
// ANCHOR_END: softmax

// ANCHOR: factories
/// `FULLY_CONNECTED`의 커널을 만든다.
pub fn fully_connected(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::FullyConnected {
        activation,
        keep_num_dims,
    } = op
    else {
        return Err(mismatch("FULLY_CONNECTED", op));
    };
    Ok(Box::new(FullyConnected {
        activation,
        keep_num_dims,
    }))
}

/// `ADD`의 커널을 만든다.
pub fn add(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Add { activation } = op else {
        return Err(mismatch("ADD", op));
    };
    Ok(Box::new(Binary {
        arithmetic: Arithmetic::Add,
        activation,
    }))
}

/// `MUL`의 커널을 만든다.
pub fn mul(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Mul { activation } = op else {
        return Err(mismatch("MUL", op));
    };
    Ok(Box::new(Binary {
        arithmetic: Arithmetic::Mul,
        activation,
    }))
}

/// `RELU`의 커널을 만든다.
pub fn relu(_: &Op) -> Result<Box<dyn Kernel>> {
    Ok(Box::new(Relu))
}

/// `SOFTMAX`의 커널을 만든다.
pub fn softmax(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Softmax { beta } = op else {
        return Err(mismatch("SOFTMAX", op));
    };
    Ok(Box::new(Softmax { beta }))
}
// ANCHOR_END: factories

fn mismatch(expected: &str, op: &Op) -> Error {
    invalid(format!("{expected}의 커널에 {}을 주었습니다", op.name()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kernel: &dyn Kernel, inputs: &[(&[usize], &[f32])]) -> (Vec<usize>, Vec<f32>) {
        let shapes: Vec<&[usize]> = inputs.iter().map(|(shape, _)| *shape).collect();
        let shape = kernel.shapes(&shapes).unwrap().remove(0);
        // 자리에 남아 있던 값이 결과에 섞이면 안 된다.
        let mut data = vec![f32::NAN; shape.iter().product()];
        let views: Vec<View<'_>> = inputs
            .iter()
            .map(|&(shape, data)| View { shape, data })
            .collect();
        kernel
            .run(
                &views,
                &mut [ViewMut {
                    shape: &shape,
                    data: &mut data,
                }],
            )
            .unwrap();
        (shape, data)
    }

    #[test]
    fn fully_connected_matches_the_definition() {
        let kernel = FullyConnected {
            activation: Activation::Relu,
            keep_num_dims: false,
        };
        let x: &[f32] = &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let w: &[f32] = &[1.0, 0.0, -1.0, 0.5, 0.5, 0.5];
        let b: &[f32] = &[0.25, -0.25];
        let (shape, y) = run(&kernel, &[(&[2, 3], x), (&[2, 3], w), (&[2], b)]);
        assert_eq!(shape, [2, 2]);
        // 첫 벡터: (1-3+0.25, 3-0.25) → relu
        assert_eq!(y, [0.0, 2.75, 0.0, 0.0]);
        // 치우침이 없어도 된다.
        let (_, y) = run(&kernel, &[(&[2, 3], x), (&[2, 3], w)]);
        assert_eq!(y, [0.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn fully_connected_flattens_or_keeps_dimensions() {
        let flat = FullyConnected {
            activation: Activation::None,
            keep_num_dims: false,
        };
        let keep = FullyConnected {
            activation: Activation::None,
            keep_num_dims: true,
        };
        let input: &[usize] = &[2, 5, 3];
        assert_eq!(flat.shapes(&[input, &[4, 3]]).unwrap(), [[10, 4]]);
        assert_eq!(keep.shapes(&[input, &[4, 3]]).unwrap(), [[2, 5, 4]]);
        // 벡터의 길이가 가중치와 맞지 않는다.
        assert!(flat.shapes(&[input, &[4, 7]]).is_err());
        assert!(flat.shapes(&[input, &[4, 3], &[5]]).is_err());
        assert!(flat.shapes(&[input, &[12]]).is_err());
    }

    #[test]
    fn binary_and_activations() {
        let add = add(&Op::Add {
            activation: Activation::Relu6,
        })
        .unwrap();
        let (_, y) = run(
            &*add,
            &[(&[3], &[1.0, -4.0, 5.0]), (&[3], &[2.0, 1.0, 5.0])],
        );
        assert_eq!(y, [3.0, 0.0, 6.0]);
        let mul = mul(&Op::Mul {
            activation: Activation::ReluN1To1,
        })
        .unwrap();
        let (_, y) = run(
            &*mul,
            &[(&[3], &[1.0, -4.0, 0.5]), (&[3], &[2.0, 1.0, 0.5])],
        );
        assert_eq!(y, [1.0, -1.0, 0.25]);
        assert!(add.shapes(&[&[3], &[1]]).is_err());
        assert!(super::add(&Op::Relu).is_err());
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let kernel = Softmax { beta: 2.0 };
        let x: &[f32] = &[0.0, 1.0, 2.0, 5.0, 5.0, 5.0];
        let (shape, y) = run(&kernel, &[(&[2, 3], x)]);
        assert_eq!(shape, [2, 3]);
        let expected = prob::softmax(&[0.0, 2.0, 4.0]);
        for (value, wanted) in y.iter().zip(&expected) {
            assert!((value - wanted).abs() < 1e-6);
        }
        for row in y.chunks(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}
