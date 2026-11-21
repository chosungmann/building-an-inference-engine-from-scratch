//! SME의 외적 명령으로 계산하는 합성곱 (31장).
//!
//! 28장의 "조각마다 펼쳐 곱하기"와 뼈대가 같다. 다른 것은 조각 안의 곱셈을
//! [`badak_core::simd::sme`]의 커널로 한다는 점과, 그러려고 필터를 한 번
//! 다시 늘어놓아 쥐고 있다는 점이다. SME가 없는 기계에서는 28장의 커널로
//! 계산한다.

use std::sync::OnceLock;

use badak_core::simd::sme::{self, Packed};
use badak_core::{arena, threads};

use crate::conv::{self, Conv2d, Method};
use crate::error::Result;
use crate::graph::Op;
use crate::kernels::{Kernel, View, ViewMut, activate};

// ANCHOR: kernel
/// SME로 계산하는 합성곱.
pub struct OuterConv2d {
    /// 형상을 정하는 일과, SME가 없을 때의 계산을 맡는다.
    plain: Conv2d,
    /// 다시 늘어놓은 필터. 처음 실행할 때 만든다. 필터의 주소와 길이를 함께
    /// 적어 두고, 다른 필터가 오면 쓰지 않는다.
    packed: OnceLock<(usize, usize, Packed)>,
}

impl Kernel for OuterConv2d {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        self.plain.shapes(inputs)
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        if !sme::available() {
            return self.plain.run(inputs, outputs);
        }
        let g = self.plain.geometry(inputs[0].shape, inputs[1].shape)?;
        let (x, w) = (inputs[0].data, inputs[1].data);
        let bias = inputs.get(2).map(|bias| bias.data);
        let y = &mut *outputs[0].data;
        let (positions, patch, channels) = (g.positions(), g.patch(), g.out_channels);
        if positions == 0 || channels == 0 || patch == 0 {
            return self.plain.run(inputs, outputs);
        }

        let key = (w.as_ptr() as usize, w.len());
        let cached = self.packed.get_or_init(|| {
            let packed = Packed::new(w, channels, patch).expect("SME가 있음을 확인했다");
            (key.0, key.1, packed)
        });
        // 가중치가 실행마다 바뀌는 그래프라면 매번 다시 늘어놓는다.
        let fresh;
        let packed = if (cached.0, cached.1) == key {
            &cached.2
        } else {
            fresh = Packed::new(w, channels, patch).expect("SME가 있음을 확인했다");
            &fresh
        };

        // 조각 하나의 자리 수. 벡터 하나에 들어가는 값의 수의 배수로 잡아야
        // 타일이 꽉 찬다.
        let lanes = sme::lanes();
        let wanted = conv::wanted_threads(&g);
        let fit = conv::tile_bytes() / (patch * size_of::<f32>());
        let share = positions.div_ceil(wanted * badak_core::simd::PIECES_PER_THREAD);
        let tile = (fit.min(share) / lanes).max(1) * lanes;
        let image = g.rows.input * g.columns.input * g.in_channels;
        let pointwise = patch == g.in_channels
            && positions == g.rows.input * g.columns.input
            && g.rows.before == 0
            && g.columns.before == 0;
        let pool = threads::global();
        for (x, y) in x
            .chunks_exact(image)
            .zip(y.chunks_exact_mut(positions * channels))
        {
            let work = |index: usize, out: &mut [f32]| {
                let (first, count) = (index * tile, out.len() / channels);
                if pointwise {
                    // 창이 1 × 1이면 입력이 이미 펼친 행렬이다(27.5절).
                    sme::linear(&x[first * patch..][..count * patch], count, packed, out);
                } else {
                    // 펼친 행렬을 따로 만들지 않고, 사진에서 곧바로 커널이
                    // 읽을 순서로 늘어놓는다.
                    let len = count.div_ceil(lanes) * patch * lanes;
                    arena::with_scratch(len, |windows| {
                        pack_windows(&g, x, first, count, lanes, windows);
                        sme::linear_packed(windows, count, packed, out);
                    });
                }
                conv::add_bias(bias, out);
                activate(self.plain.activation, out);
            };
            pool.pool().run_chunks(y, tile * channels, wanted, &work);
        }
        Ok(())
    }
}
// ANCHOR_END: kernel

// ANCHOR: pack_windows
/// 출력의 자리 `first`부터 `count`개의 창을, SME 커널이 읽을 순서로 사진에서
/// 곧바로 늘어놓는다.
///
/// 28장의 길은 둘로 나뉘어 있었다. 창을 행렬로 펼치고(`unfold_range`),
/// 커널이 그 행렬을 다시 읽는다. 커널이 열 배 빨라지면 펼치는 복사가 계산과
/// 맞먹게 된다. 여기서는 한 번만 읽고 쓴다.
///
/// 커널은 자리 `lanes`개의 묶음마다 "`k`번째 값들"을 이어서 읽는다. 창의
/// 칸 하나(`tap`)를 보면, 한 줄에 나란히 있는 자리들이 읽는 입력 픽셀도
/// 나란히 있다(보폭이 1일 때). 그러니 그 픽셀들 `(자리, 채널)`을 뒤집어
/// 놓으면 그 칸의 몫이 끝난다. 줄이 바뀌거나, 보폭이 1이 아니거나, 패딩한
/// 자리가 끼면 값을 하나씩 옮긴다.
fn pack_windows(
    g: &conv::Geometry,
    x: &[f32],
    first: usize,
    count: usize,
    lanes: usize,
    packed: &mut [f32],
) {
    let (channels, width) = (g.in_channels, g.columns.input);
    let group = g.patch() * lanes;
    for (block, start) in (first..first + count).step_by(lanes).enumerate() {
        let vectors = lanes.min(first + count - start);
        let target = &mut packed[block * group..][..group];
        let (row, column) = (start / g.columns.output, start % g.columns.output);
        let in_one_row = g.columns.stride == 1 && column + vectors <= g.columns.output;
        for tap_row in 0..g.rows.filter {
            for tap_column in 0..g.columns.filter {
                let tap = tap_row * g.columns.filter + tap_column;
                let target = &mut target[tap * channels * lanes..][..channels * lanes];
                let ends = (
                    g.columns.source(column, tap_column),
                    g.columns.source(column + vectors - 1, tap_column),
                );
                if in_one_row
                    && let (Some(r), (Some(c), Some(_))) = (g.rows.source(row, tap_row), ends)
                {
                    let pixels = &x[(r * width + c) * channels..][..vectors * channels];
                    sme::pack_inputs(pixels, vectors, channels, lanes, target);
                    continue;
                }
                for index in 0..lanes {
                    let position = start + index;
                    let (row, column) = (position / g.columns.output, position % g.columns.output);
                    let pixel = (index < vectors)
                        .then(|| {
                            let r = g.rows.source(row, tap_row)?;
                            let c = g.columns.source(column, tap_column)?;
                            Some(&x[(r * width + c) * channels..][..channels])
                        })
                        .flatten();
                    for channel in 0..channels {
                        // 패딩한 자리와 남는 자리는 0이다.
                        target[channel * lanes + index] = pixel.map_or(0.0, |pixel| pixel[channel]);
                    }
                }
            }
        }
    }
}
// ANCHOR_END: pack_windows

/// SME로 계산하는 `CONV_2D`의 커널을 만든다. `Registry::register`에 넘긴다.
pub fn conv_2d_outer(op: &Op) -> Result<Box<dyn Kernel>> {
    Ok(Box::new(OuterConv2d {
        plain: conv::conv_kernel(op, Method::Auto)?,
        packed: OnceLock::new(),
    }))
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::graph::{Activation, Padding};

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.07 - 1.0)
            .collect()
    }

    #[test]
    fn matches_the_plain_kernel_on_every_kind_of_window() {
        if !sme::available() {
            eprintln!("건너뜀: 이 기계에는 SME가 없습니다");
            return;
        }
        // (높이, 너비, 입력 채널, 출력 채널, 창, 보폭, 간격, 패딩)
        let same = Padding::Same;
        let cases = [
            (9, 21, 5, 7, 3, 1, 1, same),
            (9, 21, 5, 70, 3, 1, 1, Padding::Valid),
            (20, 19, 3, 33, 3, 2, 1, same),
            (12, 40, 6, 20, 1, 1, 1, same),
            (12, 40, 6, 20, 1, 2, 1, same),
            (11, 13, 4, 9, 3, 1, 2, same),
            (16, 16, 3, 8, 7, 2, 1, Padding::Explicit([[3, 3], [2, 3]])),
            (5, 3, 2, 3, 3, 1, 1, same),
        ];
        for (height, width, channels, filters, window, stride, dilation, padding) in cases {
            let op = Op::Conv2d {
                padding,
                stride: [stride; 2],
                dilation: [dilation; 2],
                activation: Activation::Relu6,
            };
            let plain = conv::conv_kernel(&op, Method::Direct).unwrap();
            let outer = conv_2d_outer(&op).unwrap();
            let shapes: [&[usize]; 3] = [
                &[2, height, width, channels],
                &[filters, window, window, channels],
                &[filters],
            ];
            let data = [
                values(2 * height * width * channels, 1),
                values(filters * window * window * channels, 2),
                values(filters, 3),
            ];
            let inputs: Vec<View<'_>> = shapes
                .iter()
                .zip(&data)
                .map(|(&shape, data)| View { shape, data })
                .collect();
            let shape = plain.shapes(&shapes).unwrap().remove(0);
            assert_eq!(outer.shapes(&shapes).unwrap()[0], shape);
            let run = |kernel: &dyn Kernel| {
                let mut y = vec![f32::NAN; shape.iter().product()];
                let mut outputs = [ViewMut {
                    shape: &shape,
                    data: &mut y,
                }];
                kernel.run(&inputs, &mut outputs).unwrap();
                y
            };
            let (expected, actual) = (run(&plain), run(&*outer));
            // 두 번째 실행은 쥐고 있던 필터를 쓴다.
            assert_eq!(run(&*outer), actual);
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() < 1e-4,
                    "{height}×{width}, 창 {window}, 보폭 {stride}의 {index}번째: {a} 대 {e}"
                );
            }
        }
    }
}
