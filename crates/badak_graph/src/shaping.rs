//! 값을 계산하지 않고 옮기기만 하는 연산들과 시그모이드 (32장).
//!
//! 객체 검출 모델은 분류 모델에 없던 일을 한다. 해상도가 다른 특징 지도를
//! 이어 붙이고(`CONCAT`), 작은 지도를 키우고(`RESIZE`), 지도에서 일부를
//! 골라내고(`SLICE`), 지도를 "칸의 목록"으로 다시 묶는다(`RESHAPE`).

use crate::error::{Error, Result};
use crate::graph::Op;
use crate::kernels::{Kernel, View, ViewMut};

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

fn single(name: &str, inputs: &[&[usize]]) -> Result<()> {
    (inputs.len() == 1)
        .then_some(())
        .ok_or_else(|| invalid(format!("{name}의 입력이 {}개입니다", inputs.len())))
}

// ANCHOR: concat
/// 텐서들을 한 축을 따라 이어 붙인다. 그 축을 뺀 형상은 모두 같아야 한다.
pub struct Concat {
    axis: usize,
}

impl Kernel for Concat {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        let Some(first) = inputs.first().filter(|shape| self.axis < shape.len()) else {
            return Err(invalid("CONCAT의 입력이 없거나 축이 형상을 벗어납니다"));
        };
        let mut shape = first.to_vec();
        shape[self.axis] = 0;
        for input in inputs {
            let same = input.len() == first.len()
                && (0..first.len()).all(|axis| axis == self.axis || input[axis] == first[axis]);
            if !same {
                return Err(invalid(format!(
                    "CONCAT의 입력 {input:?}와 {first:?}는 축 {} 밖에서 형상이 다릅니다",
                    self.axis
                )));
            }
            shape[self.axis] += input[self.axis];
        }
        Ok(vec![shape])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        // 이어 붙이는 축부터 끝까지를 한 덩어리로 보면, 출력은 입력들의
        // 덩어리를 차례로 돌아가며 놓은 것이다.
        let chunks: Vec<usize> = inputs
            .iter()
            .map(|input| input.shape[self.axis..].iter().product())
            .collect();
        let total: usize = chunks.iter().sum();
        if total == 0 {
            return Ok(());
        }
        for (index, row) in outputs[0].data.chunks_exact_mut(total).enumerate() {
            let mut start = 0;
            for (input, &chunk) in inputs.iter().zip(&chunks) {
                row[start..start + chunk].copy_from_slice(&input.data[index * chunk..][..chunk]);
                start += chunk;
            }
        }
        Ok(())
    }
}
// ANCHOR_END: concat

// ANCHOR: slice
/// 축마다 `[시작, 끝, 스텝]`으로 일부를 골라낸다. 끝은 포함하지 않는다.
pub struct Slice {
    ranges: Vec<[usize; 3]>,
}

impl Slice {
    fn lengths(&self, input: &[usize]) -> Result<Vec<usize>> {
        if input.len() != self.ranges.len() {
            return Err(invalid(format!(
                "축이 {}개인 입력에 축 {}개의 범위를 주었습니다",
                input.len(),
                self.ranges.len()
            )));
        }
        input
            .iter()
            .zip(&self.ranges)
            .map(|(&extent, &[start, end, step])| {
                if step == 0 || start > end || end > extent {
                    return Err(invalid(format!(
                        "길이 {extent}인 축에서 {start}부터 {end}까지 {step}씩 고를 수 없습니다"
                    )));
                }
                Ok((end - start).div_ceil(step))
            })
            .collect()
    }
}

impl Kernel for Slice {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("SLICE", inputs)?;
        Ok(vec![self.lengths(inputs[0])?])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (x, shape) = (inputs[0].data, inputs[0].shape);
        let output = outputs[0].shape;
        let y = &mut *outputs[0].data;
        if y.is_empty() {
            return Ok(());
        }
        let rank = shape.len();
        // 축마다 한 칸 움직일 때 건너뛰는 원소의 수.
        let mut strides = vec![1; rank];
        for axis in (0..rank - 1).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        // 뒤쪽 축들이 통째로 골라졌으면 그 축들은 한 덩어리로 복사한다.
        // 사진에서 줄이나 픽셀을 골라내는 경우가 그렇다(채널은 그대로다).
        let whole = |axis: usize| self.ranges[axis] == [0, shape[axis], 1];
        let cut = (0..rank)
            .rev()
            .find(|&axis| !whole(axis))
            .map_or(0, |axis| axis + 1);
        let block: usize = shape[cut..].iter().product();
        let mut index = vec![0; cut];
        for target in y.chunks_exact_mut(block.max(1)) {
            let source: usize = (0..cut)
                .map(|axis| {
                    (self.ranges[axis][0] + index[axis] * self.ranges[axis][2]) * strides[axis]
                })
                .sum();
            target.copy_from_slice(&x[source..source + target.len()]);
            for axis in (0..cut).rev() {
                index[axis] += 1;
                if index[axis] < output[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        Ok(())
    }
}
// ANCHOR_END: slice

// ANCHOR: resize
/// 사진을 정수 배로 키운다. 가장 가까운 픽셀의 값을 그대로 쓴다. 입력은
/// `(장 수, 높이, 너비, 채널)`이다.
pub struct Resize {
    scale: [usize; 2],
}

impl Kernel for Resize {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("RESIZE", inputs)?;
        let &[batch, height, width, channels] = inputs[0] else {
            return Err(invalid("RESIZE의 입력은 4차원이어야 합니다"));
        };
        if self.scale.contains(&0) {
            return Err(invalid("RESIZE의 배율이 0입니다"));
        }
        Ok(vec![vec![
            batch,
            height * self.scale[0],
            width * self.scale[1],
            channels,
        ]])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let &[_, _, width, channels] = inputs[0].shape else {
            unreachable!("shapes가 확인했다")
        };
        let line = width * channels;
        let wide = line * self.scale[1];
        if line == 0 {
            return Ok(());
        }
        // 입력의 한 줄이 출력의 `scale[0]`줄이 된다. 줄 안에서는 픽셀
        // 하나가 `scale[1]`번 되풀이된다.
        let rows = outputs[0].data.chunks_exact_mut(wide * self.scale[0]);
        for (source, rows) in inputs[0].data.chunks_exact(line).zip(rows) {
            let (first, rest) = rows.split_at_mut(wide);
            for (pixel, copies) in source
                .chunks_exact(channels)
                .zip(first.chunks_exact_mut(channels * self.scale[1]))
            {
                for copy in copies.chunks_exact_mut(channels) {
                    copy.copy_from_slice(pixel);
                }
            }
            for row in rest.chunks_exact_mut(wide) {
                row.copy_from_slice(first);
            }
        }
        Ok(())
    }
}
// ANCHOR_END: resize

/// 형상을 바꾼다. 값과 그 순서는 그대로다.
pub struct Reshape {
    shape: Vec<usize>,
}

impl Kernel for Reshape {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("RESHAPE", inputs)?;
        let (from, to): (usize, usize) = (inputs[0].iter().product(), self.shape.iter().product());
        if from != to {
            return Err(invalid(format!(
                "원소가 {from}개인 {:?}를 {:?}로 바꿀 수 없습니다",
                inputs[0], self.shape
            )));
        }
        Ok(vec![self.shape.clone()])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        outputs[0].data.copy_from_slice(inputs[0].data);
        Ok(())
    }

    // 제자리에서는 할 일이 없다(30.11절).
    fn overwrites(&self, _: usize) -> bool {
        true
    }

    fn run_in_place(&self, _: &[View<'_>], _: usize, _: &mut ViewMut<'_>) -> Result<()> {
        Ok(())
    }
}

/// 원소마다 시그모이드.
pub struct Sigmoid;

impl Kernel for Sigmoid {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        single("SIGMOID", inputs)?;
        Ok(vec![inputs[0].to_vec()])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        for (value, &x) in outputs[0].data.iter_mut().zip(inputs[0].data) {
            *value = badak_core::nn::sigmoid(x);
        }
        Ok(())
    }

    fn overwrites(&self, _: usize) -> bool {
        true
    }

    fn run_in_place(&self, _: &[View<'_>], _: usize, output: &mut ViewMut<'_>) -> Result<()> {
        for value in output.data.iter_mut() {
            *value = badak_core::nn::sigmoid(*value);
        }
        Ok(())
    }
}

fn mismatch(expected: &str, op: &Op) -> Error {
    invalid(format!("{expected}의 커널에 {}을 주었습니다", op.name()))
}

/// `CONCAT`의 커널을 만든다.
pub fn concat(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Concat { axis } = op else {
        return Err(mismatch("CONCAT", op));
    };
    Ok(Box::new(Concat { axis }))
}

/// `SLICE`의 커널을 만든다.
pub fn slice(op: &Op) -> Result<Box<dyn Kernel>> {
    let Op::Slice { ranges } = op else {
        return Err(mismatch("SLICE", op));
    };
    Ok(Box::new(Slice {
        ranges: ranges.clone(),
    }))
}

/// `RESIZE`의 커널을 만든다.
pub fn resize(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Resize { scale } = op else {
        return Err(mismatch("RESIZE", op));
    };
    Ok(Box::new(Resize { scale }))
}

/// `RESHAPE`의 커널을 만든다.
pub fn reshape(op: &Op) -> Result<Box<dyn Kernel>> {
    let Op::Reshape { shape } = op else {
        return Err(mismatch("RESHAPE", op));
    };
    Ok(Box::new(Reshape {
        shape: shape.clone(),
    }))
}

/// `SIGMOID`의 커널을 만든다.
pub fn sigmoid(_: &Op) -> Result<Box<dyn Kernel>> {
    Ok(Box::new(Sigmoid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kernel: &dyn Kernel, inputs: &[(&[usize], &[f32])]) -> (Vec<usize>, Vec<f32>) {
        let shapes: Vec<&[usize]> = inputs.iter().map(|&(shape, _)| shape).collect();
        let shape = kernel.shapes(&shapes).unwrap().remove(0);
        let views: Vec<View<'_>> = inputs
            .iter()
            .map(|&(shape, data)| View { shape, data })
            .collect();
        let mut y = vec![f32::NAN; shape.iter().product()];
        let mut outputs = [ViewMut {
            shape: &shape,
            data: &mut y,
        }];
        kernel.run(&views, &mut outputs).unwrap();
        (shape, y)
    }

    #[test]
    fn concat_joins_along_any_axis() {
        let a: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let b: &[f32] = &[5.0, 6.0];
        // 마지막 축: 줄마다 a의 둘 뒤에 b의 하나.
        let (shape, y) = run(&Concat { axis: 1 }, &[(&[2, 2], a), (&[2, 1], b)]);
        assert_eq!(shape, [2, 3]);
        assert_eq!(y, [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
        // 첫 축: a 전체 뒤에 b 전체.
        let (shape, y) = run(&Concat { axis: 0 }, &[(&[2, 2], a), (&[1, 2], b)]);
        assert_eq!(shape, [3, 2]);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // 다른 축의 길이가 다르면 거절한다.
        assert!(Concat { axis: 0 }.shapes(&[&[2, 2], &[1, 3]]).is_err());
        assert!(Concat { axis: 2 }.shapes(&[&[2, 2]]).is_err());
    }

    #[test]
    fn slice_picks_every_other_pixel() {
        // 4 × 4에서 홀수 줄, 짝수 칸.
        let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let kernel = Slice {
            ranges: vec![[1, 4, 2], [0, 4, 2]],
        };
        let (shape, y) = run(&kernel, &[(&[4, 4], &x)]);
        assert_eq!(shape, [2, 2]);
        assert_eq!(y, [4.0, 6.0, 12.0, 14.0]);
        let bad = Slice {
            ranges: vec![[0, 5, 1], [0, 4, 1]],
        };
        assert!(bad.shapes(&[&[4, 4]]).is_err());
        let zero = Slice {
            ranges: vec![[0, 4, 0], [0, 4, 1]],
        };
        assert!(zero.shapes(&[&[4, 4]]).is_err());
    }

    #[test]
    fn resize_repeats_pixels() {
        // 1 × 2 픽셀, 채널 2. 세로 2배, 가로 3배.
        let x: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let (shape, y) = run(&Resize { scale: [2, 3] }, &[(&[1, 1, 2, 2], x)]);
        assert_eq!(shape, [1, 2, 6, 2]);
        let row = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
        assert_eq!(y[..12], row);
        assert_eq!(y[12..], row);
    }

    #[test]
    fn reshape_keeps_values_and_checks_the_count() {
        let x: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let kernel = Reshape { shape: vec![3, 2] };
        let (shape, y) = run(&kernel, &[(&[1, 2, 3], x)]);
        assert_eq!((shape.as_slice(), y.as_slice()), (&[3, 2][..], x));
        assert!(kernel.shapes(&[&[2, 2]]).is_err());
    }

    #[test]
    fn sigmoid_squashes() {
        let (_, y) = run(&Sigmoid, &[(&[3], &[0.0, 100.0, -100.0])]);
        assert_eq!(y[0], 0.5);
        assert!(y[1] > 0.999 && y[2] < 0.001 && y[2] >= 0.0);
    }
}
