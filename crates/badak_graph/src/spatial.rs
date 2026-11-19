//! 합성곱 모델이 쓰는 나머지 연산들 (27장): 풀링, 덧대기, 평균, 축 바꾸기.

use crate::conv::Axis;
use crate::error::{Error, Result};
use crate::graph::{Op, Padding};
use crate::kernels::{Kernel, View, ViewMut};

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

fn single(name: &str, inputs: &[&[usize]]) -> Result<()> {
    (inputs.len() == 1)
        .then_some(())
        .ok_or_else(|| invalid(format!("{name}의 입력이 {}개입니다", inputs.len())))
}

// ANCHOR: pool
/// 창 안의 값들을 하나로 줄이는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reduce {
    Max,
    Average,
}

/// 풀링. 채널마다 따로, 창 안의 최댓값이나 평균을 낸다.
pub struct Pool2d {
    reduce: Reduce,
    padding: Padding,
    stride: [usize; 2],
    filter: [usize; 2],
}

impl Pool2d {
    fn axes(&self, input: &[usize]) -> Result<(usize, Axis, Axis, usize)> {
        let &[batch, height, width, channels] = input else {
            return Err(invalid(format!(
                "풀링의 입력 {input:?}는 4차원이어야 합니다"
            )));
        };
        let rows = Axis::new(height, self.filter[0], self.stride[0], 1, self.padding, 0)?;
        let columns = Axis::new(width, self.filter[1], self.stride[1], 1, self.padding, 1)?;
        Ok((batch, rows, columns, channels))
    }
}

impl Kernel for Pool2d {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("풀링", inputs)?;
        let (batch, rows, columns, channels) = self.axes(inputs[0])?;
        Ok(vec![vec![batch, rows.output, columns.output, channels]])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (_, rows, columns, channels) = self.axes(inputs[0].shape)?;
        let image = rows.input * columns.input * channels;
        let positions = rows.output * columns.output;
        let y = &mut *outputs[0].data;
        for (x, y) in inputs[0]
            .data
            .chunks_exact(image)
            .zip(y.chunks_exact_mut(positions * channels))
        {
            for (position, out) in y.chunks_exact_mut(channels).enumerate() {
                let (row, column) = (position / columns.output, position % columns.output);
                out.fill(match self.reduce {
                    Reduce::Max => f32::NEG_INFINITY,
                    Reduce::Average => 0.0,
                });
                // 덧댄 자리는 세지 않는다. 최댓값에서는 없는 값이고, 평균에서는
                // 나누는 수에도 넣지 않는다.
                let mut count = 0;
                for tap_row in 0..rows.filter {
                    let Some(source_row) = rows.source(row, tap_row) else {
                        continue;
                    };
                    for tap_column in 0..columns.filter {
                        let Some(source_column) = columns.source(column, tap_column) else {
                            continue;
                        };
                        let pixel = (source_row * columns.input + source_column) * channels;
                        let pixel = &x[pixel..pixel + channels];
                        count += 1;
                        match self.reduce {
                            Reduce::Max => {
                                for (value, &candidate) in out.iter_mut().zip(pixel) {
                                    *value = value.max(candidate);
                                }
                            }
                            Reduce::Average => {
                                for (value, &term) in out.iter_mut().zip(pixel) {
                                    *value += term;
                                }
                            }
                        }
                    }
                }
                if self.reduce == Reduce::Average {
                    let scale = 1.0 / count.max(1) as f32;
                    out.iter_mut().for_each(|value| *value *= scale);
                }
            }
        }
        Ok(())
    }
}
// ANCHOR_END: pool

/// 여러 축의 번호를 하나씩 올리며 도는 데 쓴다. 맨 뒤의 축부터 올리고,
/// 끝까지 돌았으면 `false`를 돌려준다.
fn advance(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

/// 축마다 한 칸 움직일 때 건너뛰는 원소의 수.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

// ANCHOR: pad
/// 가장자리에 값을 덧댄다.
pub struct Pad {
    amounts: Vec<[usize; 2]>,
    value: f32,
}

impl Kernel for Pad {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("PAD", inputs)?;
        if inputs[0].len() != self.amounts.len() || inputs[0].is_empty() {
            return Err(invalid(format!(
                "축이 {}개인 입력에 축 {}개의 덧댈 길이를 주었습니다",
                inputs[0].len(),
                self.amounts.len()
            )));
        }
        Ok(vec![
            inputs[0]
                .iter()
                .zip(&self.amounts)
                .map(|(&extent, &[before, after])| before + extent + after)
                .collect(),
        ])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (input, output) = (inputs[0].shape, outputs[0].shape);
        let y = &mut *outputs[0].data;
        y.fill(self.value);
        let last = input.len() - 1;
        let width = input[last];
        if inputs[0].data.is_empty() {
            return Ok(());
        }
        // 입력을 마지막 축의 줄 단위로 옮긴다. 줄마다 출력에서의 자리를
        // 계산한다.
        let steps = strides(output);
        let mut index = vec![0; last];
        for line in inputs[0].data.chunks_exact(width) {
            let mut start = self.amounts[last][0];
            for axis in 0..last {
                start += (index[axis] + self.amounts[axis][0]) * steps[axis];
            }
            y[start..start + width].copy_from_slice(line);
            advance(&mut index, &input[..last]);
        }
        Ok(())
    }
}
// ANCHOR_END: pad

// ANCHOR: mean
/// 축들을 따라 평균을 낸다.
pub struct Mean {
    axes: Vec<usize>,
    keep_dims: bool,
}

impl Mean {
    fn check(&self, input: &[usize]) -> Result<()> {
        let mut seen = vec![false; input.len()];
        for &axis in &self.axes {
            if axis >= input.len() || std::mem::replace(&mut seen[axis], true) {
                return Err(invalid(format!(
                    "축이 {}개인 입력에서 축 {:?}의 평균을 낼 수 없습니다",
                    input.len(),
                    self.axes
                )));
            }
        }
        Ok(())
    }
}

impl Kernel for Mean {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("MEAN", inputs)?;
        self.check(inputs[0])?;
        let shape = inputs[0]
            .iter()
            .enumerate()
            .filter_map(
                |(axis, &extent)| match (self.axes.contains(&axis), self.keep_dims) {
                    (true, true) => Some(1),
                    (true, false) => None,
                    (false, _) => Some(extent),
                },
            )
            .collect();
        Ok(vec![shape])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let input = inputs[0].shape;
        let (x, y) = (inputs[0].data, &mut *outputs[0].data);
        let count: usize = self.axes.iter().map(|&axis| input[axis]).product();
        y.fill(0.0);
        if x.is_empty() {
            return Ok(());
        }
        let last = input.len() - 1;
        if self.axes.contains(&last) {
            return Err(Error::Unsupported("마지막 축을 줄이는 MEAN".into()));
        }
        // 줄이지 않는 축들만으로 출력에서의 자리를 정한다. 마지막 축은 남으므로
        // 입력의 줄 하나를 출력의 줄 하나에 통째로 더한다. 사진의 높이와
        // 너비를 줄일 때(전역 평균 풀링) 채널들이 한 번에 더해진다.
        let width = input[last];
        let kept: Vec<usize> = (0..last).filter(|axis| !self.axes.contains(axis)).collect();
        let mut index = vec![0; last];
        for line in x.chunks_exact(width) {
            let mut row = 0;
            for &axis in &kept {
                row = row * input[axis] + index[axis];
            }
            for (sum, &value) in y[row * width..(row + 1) * width].iter_mut().zip(line) {
                *sum += value;
            }
            advance(&mut index, &input[..last]);
        }
        let scale = 1.0 / count.max(1) as f32;
        y.iter_mut().for_each(|value| *value *= scale);
        Ok(())
    }
}
// ANCHOR_END: mean

// ANCHOR: transpose
/// 축의 순서를 바꾼다. 값은 그대로이고 놓이는 순서만 달라진다.
pub struct Transpose {
    perm: Vec<usize>,
}

impl Kernel for Transpose {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("TRANSPOSE", inputs)?;
        let mut seen = vec![false; inputs[0].len()];
        let valid = self.perm.len() == inputs[0].len()
            && self
                .perm
                .iter()
                .all(|&axis| axis < seen.len() && !std::mem::replace(&mut seen[axis], true));
        if !valid || inputs[0].is_empty() {
            return Err(invalid(format!(
                "축의 순서 {:?}가 축이 {}개인 입력과 맞지 않습니다",
                self.perm,
                inputs[0].len()
            )));
        }
        Ok(vec![
            self.perm.iter().map(|&axis| inputs[0][axis]).collect(),
        ])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (x, y) = (inputs[0].data, &mut *outputs[0].data);
        if y.is_empty() {
            return Ok(());
        }
        let output = outputs[0].shape;
        // 출력의 축을 한 칸 움직이면 입력에서는 몇 칸을 건너뛰는지.
        let source = strides(inputs[0].shape);
        let steps: Vec<usize> = self.perm.iter().map(|&axis| source[axis]).collect();
        let last = output.len() - 1;
        let mut index = vec![0; last];
        for line in y.chunks_exact_mut(output[last]) {
            let start: usize = index.iter().zip(&steps).map(|(i, step)| i * step).sum();
            for (column, value) in line.iter_mut().enumerate() {
                *value = x[start + column * steps[last]];
            }
            advance(&mut index, &output[..last]);
        }
        Ok(())
    }
}
// ANCHOR_END: transpose

fn mismatch(expected: &str, op: &Op) -> Error {
    invalid(format!("{expected}의 커널에 {}을 주었습니다", op.name()))
}

/// `MAX_POOL_2D`의 커널을 만든다.
pub fn max_pool_2d(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::MaxPool2d {
        padding,
        stride,
        filter,
    } = op
    else {
        return Err(mismatch("MAX_POOL_2D", op));
    };
    Ok(Box::new(Pool2d {
        reduce: Reduce::Max,
        padding,
        stride,
        filter,
    }))
}

/// `AVERAGE_POOL_2D`의 커널을 만든다.
pub fn average_pool_2d(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::AveragePool2d {
        padding,
        stride,
        filter,
    } = op
    else {
        return Err(mismatch("AVERAGE_POOL_2D", op));
    };
    Ok(Box::new(Pool2d {
        reduce: Reduce::Average,
        padding,
        stride,
        filter,
    }))
}

/// `PAD`의 커널을 만든다.
pub fn pad(op: &Op) -> Result<Box<dyn Kernel>> {
    let Op::Pad { amounts, value } = op else {
        return Err(mismatch("PAD", op));
    };
    Ok(Box::new(Pad {
        amounts: amounts.clone(),
        value: *value,
    }))
}

/// `MEAN`의 커널을 만든다.
pub fn mean(op: &Op) -> Result<Box<dyn Kernel>> {
    let Op::Mean { axes, keep_dims } = op else {
        return Err(mismatch("MEAN", op));
    };
    Ok(Box::new(Mean {
        axes: axes.clone(),
        keep_dims: *keep_dims,
    }))
}

/// `TRANSPOSE`의 커널을 만든다.
pub fn transpose(op: &Op) -> Result<Box<dyn Kernel>> {
    let Op::Transpose { perm } = op else {
        return Err(mismatch("TRANSPOSE", op));
    };
    Ok(Box::new(Transpose { perm: perm.clone() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kernel: &dyn Kernel, shape: &[usize], data: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let out_shape = kernel.shapes(&[shape]).unwrap().remove(0);
        let mut out = vec![f32::NAN; out_shape.iter().product()];
        let mut outputs = [ViewMut {
            shape: &out_shape,
            data: &mut out,
        }];
        kernel.run(&[View { shape, data }], &mut outputs).unwrap();
        (out_shape, out)
    }

    fn range(count: usize) -> Vec<f32> {
        (0..count).map(|value| value as f32).collect()
    }

    #[test]
    fn max_pool_takes_the_largest_in_each_window() {
        let pool = Pool2d {
            reduce: Reduce::Max,
            padding: Padding::Valid,
            stride: [2, 2],
            filter: [2, 2],
        };
        // 4 × 4, 채널 하나.
        let (shape, y) = run(&pool, &[1, 4, 4, 1], &range(16));
        assert_eq!(shape, [1, 2, 2, 1]);
        assert_eq!(y, [5.0, 7.0, 13.0, 15.0]);
        // 채널이 둘이면 채널마다 따로.
        let x: Vec<f32> = (0..8).flat_map(|v| [v as f32, -(v as f32)]).collect();
        let (shape, y) = run(&pool, &[1, 2, 4, 2], &x);
        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(y, [5.0, 0.0, 7.0, -2.0]);
    }

    #[test]
    fn average_pool_ignores_padding() {
        let pool = Pool2d {
            reduce: Reduce::Average,
            padding: Padding::Same,
            stride: [2, 2],
            filter: [2, 2],
        };
        // 3 × 3: 마지막 행과 열의 창은 덧댄 자리를 빼고 평균을 낸다.
        let (shape, y) = run(&pool, &[1, 3, 3, 1], &range(9));
        assert_eq!(shape, [1, 2, 2, 1]);
        assert_eq!(y, [2.0, 3.5, 6.5, 8.0]);
    }

    #[test]
    fn pad_surrounds_the_input() {
        let pad = Pad {
            amounts: vec![[0, 0], [1, 2], [2, 1]],
            value: -1.0,
        };
        let (shape, y) = run(&pad, &[1, 1, 2], &[5.0, 6.0]);
        assert_eq!(shape, [1, 4, 5]);
        let mut expected = vec![-1.0; 20];
        expected[7] = 5.0;
        expected[8] = 6.0;
        assert_eq!(y, expected);
        assert!(pad.shapes(&[&[1, 2]]).is_err());
    }

    #[test]
    fn mean_over_height_and_width_keeps_channels() {
        let mean = Mean {
            axes: vec![1, 2],
            keep_dims: false,
        };
        // 두 장, 2 × 2, 채널 둘.
        let x: Vec<f32> = range(16);
        let (shape, y) = run(&mean, &[2, 2, 2, 2], &x);
        assert_eq!(shape, [2, 2]);
        assert_eq!(y, [3.0, 4.0, 11.0, 12.0]);
        let kept = Mean {
            axes: vec![1, 2],
            keep_dims: true,
        };
        assert_eq!(kept.shapes(&[&[2, 2, 2, 2]]).unwrap(), [[2, 1, 1, 2]]);
        // 가운데 축 하나만 줄이기.
        let one = Mean {
            axes: vec![1],
            keep_dims: false,
        };
        let (shape, y) = run(&one, &[2, 2, 2, 2], &x);
        assert_eq!(shape, [2, 2, 2]);
        assert_eq!(y, [2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0]);
        let bad = Mean {
            axes: vec![1, 1],
            keep_dims: false,
        };
        assert!(bad.shapes(&[&[2, 2, 2, 2]]).is_err());
    }

    #[test]
    fn transpose_moves_channels_last() {
        // (1, 채널 2, 높이 2, 너비 3) → (1, 높이, 너비, 채널).
        let transpose = Transpose {
            perm: vec![0, 2, 3, 1],
        };
        let (shape, y) = run(&transpose, &[1, 2, 2, 3], &range(12));
        assert_eq!(shape, [1, 2, 3, 2]);
        assert_eq!(
            y,
            [0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0]
        );
        // 되돌리면 원래대로.
        let back = Transpose {
            perm: vec![0, 3, 1, 2],
        };
        let (_, again) = run(&back, &shape, &y);
        assert_eq!(again, range(12));
        let bad = Transpose {
            perm: vec![0, 0, 1, 2],
        };
        assert!(bad.shapes(&[&[1, 2, 2, 3]]).is_err());
    }
}
