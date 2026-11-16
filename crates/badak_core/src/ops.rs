//! 텐서 위의 연산자 (5장).
//!
//! 연산자는 텐서를 받아 새 텐서를 돌려주는 함수다. 여기 있는 연산자는
//! 모두 `f32` 텐서를 다루며, 세 가지 일을 순서대로 한다.
//!
//! 1. 형상과 자료형을 검사하고 출력의 형상을 정한다.
//! 2. 입력이 연속이면 슬라이스로, 아니면 반복자로 원소를 읽는다.
//! 3. 2장과 3장의 단순한 함수로 계산해 새 텐서에 담는다.

use crate::error::{Error, Result};
use crate::tensor::{Tensor, broadcast_shapes};
use crate::{linalg, nn, prob};

// ANCHOR: map
/// 모든 원소에 함수를 적용한다. 형상은 변하지 않는다.
pub fn map(x: &Tensor, f: impl Fn(f32) -> f32) -> Result<Tensor> {
    let data: Vec<f32> = match x.as_slice::<f32>() {
        // 빠른 길: 연속인 메모리를 단순한 반복문으로 읽는다.
        Ok(values) => values.iter().map(|&value| f(value)).collect(),
        // 일반적인 길: 스트라이드를 따라가며 읽는다.
        Err(Error::NotContiguous) => x.iter::<f32>()?.map(f).collect(),
        Err(error) => return Err(error),
    };
    Tensor::from_vec(data, x.shape())
}
// ANCHOR_END: map

// ANCHOR: unary
/// 원소별 ReLU.
pub fn relu(x: &Tensor) -> Result<Tensor> {
    map(x, nn::relu)
}

/// 원소별 시그모이드.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    map(x, nn::sigmoid)
}

/// 원소별 tanh.
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    map(x, nn::tanh)
}

/// 원소별 SiLU.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    map(x, nn::silu)
}
// ANCHOR_END: unary

// ANCHOR: zip
/// 두 텐서의 같은 자리 원소끼리 함수를 적용한다. 형상이 다르면
/// 브로드캐스팅한다.
pub fn zip(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    let shape = broadcast_shapes(a.shape(), b.shape())?;

    // 빠른 길: 형상이 같고 둘 다 연속이면 슬라이스 두 개를 나란히 읽는다.
    if a.shape() == b.shape()
        && let (Ok(left), Ok(right)) = (a.as_slice::<f32>(), b.as_slice::<f32>())
    {
        let data = left.iter().zip(right).map(|(&x, &y)| f(x, y)).collect();
        return Tensor::from_vec(data, &shape);
    }

    // 일반적인 길: 둘을 결과 형상의 뷰로 늘린 뒤 반복자로 읽는다.
    let left = a.broadcast_to(&shape)?;
    let right = b.broadcast_to(&shape)?;
    let data = left
        .iter::<f32>()?
        .zip(right.iter::<f32>()?)
        .map(|(x, y)| f(x, y))
        .collect();
    Tensor::from_vec(data, &shape)
}
// ANCHOR_END: zip

// ANCHOR: binary
/// 원소별 덧셈.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    zip(a, b, |x, y| x + y)
}

/// 원소별 뺄셈.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    zip(a, b, |x, y| x - y)
}

/// 원소별 곱셈. 행렬 곱이 아니다.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    zip(a, b, |x, y| x * y)
}

/// 원소별 나눗셈.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    zip(a, b, |x, y| x / y)
}
// ANCHOR_END: binary

// ANCHOR: matmul
/// 행렬 곱. `a`는 `(m, k)`, `b`는 `(k, n)`, 결과는 `(m, n)`이다.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (&[m, k], &[k2, n]) = (a.shape(), b.shape()) else {
        return Err(shape_error("matmul", a, b));
    };
    if k != k2 {
        return Err(shape_error("matmul", a, b));
    }
    // 커널은 연속인 메모리를 요구한다. 전치 뷰가 들어오면 여기서 복사된다.
    let (a, b) = (a.contiguous(), b.contiguous());
    let data = linalg::matmul(a.as_slice()?, b.as_slice()?, m, k, n);
    Tensor::from_vec(data, &[m, n])
}
// ANCHOR_END: matmul

// ANCHOR: linear
/// 선형 층 `y = x Wᵀ + b`를 마지막 축에 적용한다.
///
/// `x`는 `(..., in)`, `weight`는 `(out, in)`, `bias`는 `(out)`이고 결과는
/// `(..., out)`이다. 앞쪽 축이 몇 개든 벡터 하나하나에 같은 층을 적용한다.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (&[out_features, in_features], Some(&last)) = (weight.shape(), x.shape().last()) else {
        return Err(shape_error("linear", x, weight));
    };
    if last != in_features {
        return Err(shape_error("linear", x, weight));
    }
    if let Some(bias) = bias
        && bias.shape() != [out_features]
    {
        return Err(shape_error("linear", weight, bias));
    }

    let (x_data, weight) = (x.contiguous(), weight.contiguous());
    let bias = bias.map(Tensor::contiguous);
    let weight = weight.as_slice::<f32>()?;
    let bias = bias.as_ref().map(Tensor::as_slice::<f32>).transpose()?;

    let mut data = Vec::with_capacity(x.len() / in_features.max(1) * out_features);
    if in_features > 0 {
        for row in x_data.as_slice::<f32>()?.chunks_exact(in_features) {
            let start = data.len();
            data.extend(linalg::matvec(weight, out_features, in_features, row));
            if let Some(bias) = bias {
                for (value, b) in data[start..].iter_mut().zip(bias) {
                    *value += b;
                }
            }
        }
    }

    let mut shape = x.shape().to_vec();
    *shape.last_mut().expect("마지막 축이 있음을 확인했다") = out_features;
    Tensor::from_vec(data, &shape)
}
// ANCHOR_END: linear

// ANCHOR: lanes
/// 한 축을 따라 놓인 원소들의 줄(lane)마다 함수를 적용한다.
///
/// `axis`를 맨 뒤로 보낸 연속인 텐서를 만들어, 그 축의 원소들이 메모리에
/// 한 줄로 이어지게 한다. 그런 다음 줄마다 `f`를 부른다. 돌려주는 값은
/// 맨 뒤로 보낸 텐서의 형상과 줄별 결과다.
fn for_each_lane<R>(
    x: &Tensor,
    axis: usize,
    mut f: impl FnMut(&[f32]) -> R,
) -> Result<(Vec<usize>, Vec<R>)> {
    if axis >= x.ndim() {
        return Err(Error::Axis {
            axis,
            ndim: x.ndim(),
        });
    }
    let mut axes: Vec<usize> = (0..x.ndim()).filter(|&a| a != axis).collect();
    axes.push(axis);
    let moved = x.permute(&axes)?.contiguous();
    let lane_len = x.shape()[axis];
    let lane_count = x.len().checked_div(lane_len).unwrap_or(0);

    let values = moved.as_slice::<f32>()?;
    let results = match lane_len {
        // 길이 0인 축: 줄마다 빈 슬라이스를 넘긴다.
        0 => {
            let empty_lanes: usize = moved.shape()[..x.ndim() - 1].iter().product();
            (0..empty_lanes).map(|_| f(&[])).collect()
        }
        _ => {
            let mut results = Vec::with_capacity(lane_count);
            results.extend(values.chunks_exact(lane_len).map(&mut f));
            results
        }
    };
    Ok((moved.shape().to_vec(), results))
}

/// 축소 연산의 결과 형상. `keepdim`이면 줄어든 축을 길이 1로 남긴다.
fn reduced_shape(x: &Tensor, axis: usize, keepdim: bool) -> Vec<usize> {
    let mut shape = x.shape().to_vec();
    if keepdim {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    shape
}
// ANCHOR_END: lanes

// ANCHOR: reduce
/// 한 축을 따라 더한다. 그 축이 사라지거나(`keepdim`이 거짓) 길이 1이
/// 된다.
pub fn sum(x: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    let (_, sums) = for_each_lane(x, axis, linalg::sum_pairwise)?;
    Tensor::from_vec(sums, &reduced_shape(x, axis, keepdim))
}

/// 한 축을 따라 평균을 구한다.
pub fn mean(x: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    let (_, means) = for_each_lane(x, axis, |lane| {
        linalg::sum_pairwise(lane) / lane.len() as f32
    })?;
    Tensor::from_vec(means, &reduced_shape(x, axis, keepdim))
}

/// 한 축을 따라 최댓값을 구한다. 그 축의 길이가 0이면 음의 무한대다.
pub fn max(x: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    let (_, maxima) = for_each_lane(x, axis, |lane| {
        lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    })?;
    Tensor::from_vec(maxima, &reduced_shape(x, axis, keepdim))
}

/// 한 축을 따라 가장 큰 값의 위치를 구한다. 결과는 `i32` 텐서다.
///
/// 그 축에 유효한 값이 없으면(길이가 0이거나 모두 `NaN`) -1을 넣는다.
pub fn argmax(x: &Tensor, axis: usize) -> Result<Tensor> {
    let (_, positions) = for_each_lane(x, axis, |lane| {
        prob::argmax(lane).map_or(-1, |position| position as i32)
    })?;
    Tensor::from_vec(positions, &reduced_shape(x, axis, false))
}

/// 모든 원소의 합.
pub fn sum_all(x: &Tensor) -> Result<f32> {
    let x = x.contiguous();
    Ok(linalg::sum_pairwise(x.as_slice()?))
}
// ANCHOR_END: reduce

// ANCHOR: softmax
/// 한 축을 따라 소프트맥스를 적용한다. 형상은 변하지 않는다.
pub fn softmax(x: &Tensor, axis: usize) -> Result<Tensor> {
    along_axis(x, axis, prob::softmax)
}

/// 한 축을 따라 로그 소프트맥스를 적용한다.
pub fn log_softmax(x: &Tensor, axis: usize) -> Result<Tensor> {
    along_axis(x, axis, prob::log_softmax)
}

/// 줄마다 같은 길이의 줄을 돌려주는 함수를 적용하고, 축을 제자리로
/// 되돌린다.
fn along_axis(x: &Tensor, axis: usize, f: impl Fn(&[f32]) -> Vec<f32>) -> Result<Tensor> {
    let (moved_shape, lanes) = for_each_lane(x, axis, f)?;
    let moved = Tensor::from_vec(lanes.concat(), &moved_shape)?;
    // 맨 뒤로 보냈던 축을 원래 자리로 옮긴다.
    let last = x.ndim() - 1;
    let mut axes: Vec<usize> = (0..last).collect();
    axes.insert(axis, last);
    Ok(moved.permute(&axes)?.contiguous())
}
// ANCHOR_END: softmax

// ANCHOR: embedding
/// 표에서 번호에 해당하는 행을 뽑아 쌓는다.
///
/// `table`은 `(종류 수, 길이)`인 `f32` 텐서, `ids`는 `i32` 텐서다. 결과의
/// 형상은 `ids`의 형상 뒤에 `길이`를 붙인 것이다. 계산은 없고 복사만 있다.
pub fn embedding(table: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let &[count, width] = table.shape() else {
        return Err(shape_error("embedding", table, ids));
    };
    let table = table.contiguous();
    let rows = table.as_slice::<f32>()?;
    let mut data = Vec::with_capacity(ids.len() * width);
    for id in ids.iter::<i32>()? {
        let row =
            usize::try_from(id)
                .ok()
                .filter(|&row| row < count)
                .ok_or(Error::OutOfBounds {
                    axis: 0,
                    index: id.max(0) as usize,
                    len: count,
                })?;
        data.extend_from_slice(&rows[row * width..(row + 1) * width]);
    }
    let mut shape = ids.shape().to_vec();
    shape.push(width);
    Tensor::from_vec(data, &shape)
}
// ANCHOR_END: embedding

// ANCHOR: rms_norm
/// RMS 정규화를 마지막 축에 적용한다.
///
/// 벡터마다 원소 제곱의 평균을 구하고, 그 제곱근으로 나누어 크기를 1
/// 근처로 맞춘 뒤, 원소마다 학습된 `weight`를 곱한다. `eps`는 0으로
/// 나누는 것을 막는 작은 수다.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (Some(&width), &[weight_len]) = (x.shape().last(), weight.shape()) else {
        return Err(shape_error("rms_norm", x, weight));
    };
    if width != weight_len {
        return Err(shape_error("rms_norm", x, weight));
    }
    let (x, weight) = (x.contiguous(), weight.contiguous());
    let weight = weight.as_slice::<f32>()?;
    let mut data = Vec::with_capacity(x.len());
    if width > 0 {
        for vector in x.as_slice::<f32>()?.chunks_exact(width) {
            let squares: Vec<f32> = vector.iter().map(|value| value * value).collect();
            let mean_square = linalg::sum_pairwise(&squares) / width as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            data.extend(
                vector
                    .iter()
                    .zip(weight)
                    .map(|(value, w)| value * scale * w),
            );
        }
    }
    Tensor::from_vec(data, x.shape())
}
// ANCHOR_END: rms_norm

fn shape_error(operation: &'static str, left: &Tensor, right: &Tensor) -> Error {
    Error::Shape {
        operation,
        left: left.shape().to_vec(),
        right: right.shape().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::{allclose, compare};
    use crate::tensor::DType;

    fn tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    fn arange(shape: &[usize]) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor::from_vec((0..len).map(|i| i as f32).collect(), shape).unwrap()
    }

    #[test]
    fn map_applies_to_every_element() {
        let x = tensor(&[-1.0, 0.0, 2.0, -3.0], &[2, 2]);
        let y = relu(&x).unwrap();
        assert_eq!(y.shape(), [2, 2]);
        assert_eq!(y.as_slice::<f32>().unwrap(), [0.0, 0.0, 2.0, 0.0]);
    }

    // ANCHOR: two_paths_test
    #[test]
    fn fast_and_general_paths_agree() {
        // 같은 값을 연속인 텐서와 연속이 아닌 뷰로 만들어 양쪽 길을 모두 탄다.
        let x = arange(&[3, 4]);
        let view = x.transpose(0, 1).unwrap();
        let copied = view.contiguous();
        assert!(!view.is_contiguous() && copied.is_contiguous());

        let through_general = silu(&view).unwrap();
        let through_fast = silu(&copied).unwrap();
        assert_eq!(
            through_general.to_vec::<f32>().unwrap(),
            through_fast.to_vec::<f32>().unwrap()
        );

        let through_general = add(&view, &view).unwrap();
        let through_fast = add(&copied, &copied).unwrap();
        assert_eq!(
            through_general.to_vec::<f32>().unwrap(),
            through_fast.to_vec::<f32>().unwrap()
        );
    }
    // ANCHOR_END: two_paths_test

    #[test]
    fn binary_ops_match_hand_calculation() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        assert_eq!(
            add(&a, &b).unwrap().to_vec::<f32>().unwrap(),
            [11.0, 22.0, 33.0, 44.0]
        );
        assert_eq!(
            sub(&b, &a).unwrap().to_vec::<f32>().unwrap(),
            [9.0, 18.0, 27.0, 36.0]
        );
        assert_eq!(
            mul(&a, &b).unwrap().to_vec::<f32>().unwrap(),
            [10.0, 40.0, 90.0, 160.0]
        );
        assert_eq!(
            div(&b, &a).unwrap().to_vec::<f32>().unwrap(),
            [10.0, 10.0, 10.0, 10.0]
        );
    }

    #[test]
    fn binary_ops_broadcast() {
        // (3, 1) + (1, 4) = (3, 4)
        let column = tensor(&[10.0, 20.0, 30.0], &[3, 1]);
        let row = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let sum = add(&column, &row).unwrap();
        assert_eq!(sum.shape(), [3, 4]);
        assert_eq!(
            sum.to_vec::<f32>().unwrap(),
            [
                11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0, 32.0, 33.0, 34.0
            ]
        );

        // 스칼라와의 연산
        let scalar = tensor(&[2.0], &[]);
        assert_eq!(
            mul(&row, &scalar).unwrap().to_vec::<f32>().unwrap(),
            [2.0, 4.0, 6.0, 8.0]
        );

        assert!(add(&tensor(&[0.0; 3], &[3]), &tensor(&[0.0; 4], &[4])).is_err());
    }

    #[test]
    fn ops_reject_non_f32_tensors() {
        let ids = Tensor::from_vec(vec![1_i32, 2, 3], &[3]).unwrap();
        assert_eq!(
            relu(&ids).unwrap_err(),
            Error::DType {
                expected: DType::F32,
                actual: DType::I32
            }
        );
    }

    #[test]
    fn matmul_matches_hand_calculation_and_checks_shapes() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.to_vec::<f32>().unwrap(), [19.0, 22.0, 43.0, 50.0]);

        assert!(matmul(&arange(&[2, 3]), &arange(&[2, 3])).is_err());
        assert!(matmul(&arange(&[6]), &arange(&[6])).is_err());
    }

    #[test]
    fn matmul_accepts_views() {
        // A · Aᵀ 를 전치 뷰로 계산한다.
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let product = matmul(&a, &a.transpose(0, 1).unwrap()).unwrap();
        // [1 2 3] · [1 4]   [14 32]
        // [4 5 6]   [2 5] = [32 77]
        //           [3 6]
        assert_eq!(product.to_vec::<f32>().unwrap(), [14.0, 32.0, 32.0, 77.0]);
    }

    #[test]
    fn linear_agrees_with_the_layer_from_chapter_3() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![10.0, 20.0];
        let layer = nn::Linear::new(weight.clone(), bias.clone(), 3, 2);
        let w = Tensor::from_vec(weight, &[2, 3]).unwrap();
        let b = Tensor::from_vec(bias, &[2]).unwrap();

        // 벡터 네 개를 (2, 2, 3) 묶음으로 넣는다.
        let rows = [
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 3.0],
        ];
        let x = Tensor::from_vec(rows.concat(), &[2, 2, 3]).unwrap();
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(y.shape(), [2, 2, 2]);

        let expected: Vec<f32> = rows.iter().flat_map(|row| layer.forward(row)).collect();
        assert_eq!(y.to_vec::<f32>().unwrap(), expected);

        assert!(linear(&arange(&[2, 4]), &w, Some(&b)).is_err());
        assert!(linear(&x, &w, Some(&arange(&[3]))).is_err());
    }

    #[test]
    fn reductions_along_each_axis() {
        // [0 1 2]
        // [3 4 5]
        let x = arange(&[2, 3]);
        assert_eq!(
            sum(&x, 0, false).unwrap().to_vec::<f32>().unwrap(),
            [3.0, 5.0, 7.0]
        );
        assert_eq!(
            sum(&x, 1, false).unwrap().to_vec::<f32>().unwrap(),
            [3.0, 12.0]
        );
        assert_eq!(
            mean(&x, 1, false).unwrap().to_vec::<f32>().unwrap(),
            [1.0, 4.0]
        );
        assert_eq!(
            max(&x, 0, false).unwrap().to_vec::<f32>().unwrap(),
            [3.0, 4.0, 5.0]
        );
        assert_eq!(sum_all(&x).unwrap(), 15.0);

        assert_eq!(sum(&x, 0, false).unwrap().shape(), [3]);
        assert_eq!(sum(&x, 0, true).unwrap().shape(), [1, 3]);
        assert_eq!(sum(&x, 1, true).unwrap().shape(), [2, 1]);
        assert!(sum(&x, 2, false).is_err());
    }

    #[test]
    fn reduction_in_the_middle_of_three_axes() {
        let x = arange(&[2, 3, 4]);
        let s = sum(&x, 1, false).unwrap();
        assert_eq!(s.shape(), [2, 4]);
        // (0, :, 0) = 0 + 4 + 8, (1, :, 3) = 15 + 19 + 23
        assert_eq!(s.get::<f32>(&[0, 0]).unwrap(), 12.0);
        assert_eq!(s.get::<f32>(&[1, 3]).unwrap(), 57.0);
    }

    #[test]
    fn keepdim_makes_the_result_broadcastable() {
        // 행마다 평균을 빼면 각 행의 합이 0이 된다.
        let x = tensor(&[1.0, 2.0, 3.0, 10.0, 20.0, 60.0], &[2, 3]);
        let centered = sub(&x, &mean(&x, 1, true).unwrap()).unwrap();
        assert_eq!(
            centered.to_vec::<f32>().unwrap(),
            [-1.0, 0.0, 1.0, -20.0, -10.0, 30.0]
        );
    }

    #[test]
    fn argmax_returns_integer_positions() {
        let x = tensor(&[0.1, 0.7, 0.2, 0.9, 0.05, 0.05], &[2, 3]);
        let positions = argmax(&x, 1).unwrap();
        assert_eq!(positions.dtype(), DType::I32);
        assert_eq!(positions.to_vec::<i32>().unwrap(), [1, 0]);
    }

    #[test]
    fn softmax_along_an_axis() {
        let x = tensor(&[0.0, 3.0_f32.ln(), 5.0, 5.0], &[2, 2]);
        let rows = softmax(&x, 1).unwrap();
        assert_eq!(rows.shape(), [2, 2]);
        assert!(allclose(
            rows.as_slice().unwrap(),
            &[0.25, 0.75, 0.5, 0.5],
            1e-6,
            0.0
        ));

        // 열 방향: 각 열의 합이 1이어야 한다.
        let columns = softmax(&x, 0).unwrap();
        let totals = sum(&columns, 0, false).unwrap();
        assert!(allclose(totals.as_slice().unwrap(), &[1.0, 1.0], 1e-6, 0.0));

        let logs = log_softmax(&x, 1).unwrap();
        let expected: Vec<f32> = rows
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|p| p.ln())
            .collect();
        assert!(allclose(logs.as_slice().unwrap(), &expected, 1e-6, 1e-6));
    }

    #[test]
    fn empty_axes_are_handled() {
        let x = Tensor::zeros(DType::F32, &[2, 0]);
        assert_eq!(
            sum(&x, 1, false).unwrap().to_vec::<f32>().unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(sum(&x, 0, false).unwrap().shape(), [0]);
        assert_eq!(argmax(&x, 1).unwrap().to_vec::<i32>().unwrap(), [-1, -1]);
    }

    #[test]
    fn mlp_from_tensor_ops_matches_chapter_3() {
        // 3장의 XOR 모델을 텐서 연산자로 다시 조립해 네 입력을 한 번에 넣는다.
        let w1 = tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let b1 = tensor(&[0.0, -1.0], &[2]);
        let w2 = tensor(&[1.0, -2.0], &[1, 2]);
        let b2 = tensor(&[0.0], &[1]);
        let x = tensor(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);

        let hidden = relu(&linear(&x, &w1, Some(&b1)).unwrap()).unwrap();
        let y = linear(&hidden, &w2, Some(&b2)).unwrap();
        assert_eq!(y.shape(), [4, 1]);
        let report = compare(y.as_slice().unwrap(), &[0.0, 1.0, 1.0, 0.0], 0.0, 0.0);
        assert!(report.is_close(), "{report}");
    }

    #[test]
    fn embedding_gathers_rows() {
        let table = arange(&[4, 3]);
        let ids = Tensor::from_vec(vec![2_i32, 0, 2, 3], &[2, 2]).unwrap();
        let out = embedding(&table, &ids).unwrap();
        assert_eq!(out.shape(), [2, 2, 3]);
        assert_eq!(
            out.to_vec::<f32>().unwrap(),
            [6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
        let bad = Tensor::from_vec(vec![4_i32], &[1]).unwrap();
        assert!(embedding(&table, &bad).is_err());
        let negative = Tensor::from_vec(vec![-1_i32], &[1]).unwrap();
        assert!(embedding(&table, &negative).is_err());
    }

    #[test]
    fn rms_norm_matches_hand_calculation() {
        // (3, 4): 제곱의 평균은 12.5, 그 제곱근은 약 3.5355
        let x = tensor(&[3.0, 4.0], &[1, 2]);
        let weight = tensor(&[1.0, 2.0], &[2]);
        let y = rms_norm(&x, &weight, 0.0).unwrap();
        let rms = 12.5_f32.sqrt();
        assert!(allclose(
            y.as_slice().unwrap(),
            &[3.0 / rms, 8.0 / rms],
            1e-6,
            0.0
        ));
        // 영벡터는 eps 덕분에 0으로 남는다. eps가 없으면 0으로 나누게 된다.
        let zeros = rms_norm(&tensor(&[0.0, 0.0], &[1, 2]), &weight, 1e-5).unwrap();
        assert_eq!(zeros.to_vec::<f32>().unwrap(), [0.0, 0.0]);
        // 정규화한 벡터는 제곱의 평균이 1이다(가중치가 1일 때).
        let ones = tensor(&[1.0, 1.0], &[2]);
        let unit = rms_norm(&tensor(&[5.0, -12.0], &[1, 2]), &ones, 0.0).unwrap();
        let mean_square: f32 = unit
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / 2.0;
        assert!((mean_square - 1.0).abs() < 1e-6);
        assert!(rms_norm(&x, &tensor(&[1.0; 3], &[3]), 0.0).is_err());
    }
}
