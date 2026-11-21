//! 위노그라드 변환으로 계산하는 `3 × 3` 합성곱 (28장).
//!
//! 출력 `2 × 2`개를 한 묶음으로 계산한다. 정의대로 하면 묶음 하나에 곱셈이
//! `2 × 2 × 3 × 3 = 36`번(채널 쌍마다)인데, 입력과 필터를 미리 더하고 빼서
//! 바꿔 두면 16번으로 줄어든다. 덧셈과 뺄셈이 늘어나는 대신 곱셈이 2.25분의
//! 1이 된다. Lavin과 Gray의 `F(2 × 2, 3 × 3)`이다.

use std::sync::OnceLock;

use badak_core::simd::{Backend, Walk};
use badak_core::{arena, simd, threads};

use crate::error::{Error, Result};
use crate::graph::{Activation, Op, Padding};
use crate::kernels::{Kernel, View, ViewMut, activate};

/// 묶음 하나가 읽는 입력의 한 변, 그리고 그 제곱(곱셈의 수).
const TILE: usize = 4;
const PRODUCTS: usize = TILE * TILE;

/// 위노그라드 변환을 쓰는 합성곱. 창이 `3 × 3`, 스트라이드와 간격이 1, 패딩이
/// `SAME`일 때만 쓸 수 있다.
pub struct Winograd {
    activation: Activation,
    /// 바꿔 둔 필터. 처음 실행할 때 한 번 만든다.
    filter: OnceLock<Vec<f32>>,
}

// ANCHOR: filter
/// 필터를 바꾼다: `U = G g Gᵀ`. 결과는 `(16, 출력 채널, 입력 채널)`이다.
/// 곱셈의 자리(16개)마다 선형 층의 가중치 행렬 하나가 된다.
fn transform_filter(w: &[f32], out_channels: usize, in_channels: usize) -> Vec<f32> {
    // G의 네 행. 3칸짜리 필터를 4칸으로 늘린다.
    let stretch = |g: [f32; 3]| {
        [
            g[0],
            0.5 * (g[0] + g[1] + g[2]),
            0.5 * (g[0] - g[1] + g[2]),
            g[2],
        ]
    };
    let mut u = vec![0.0; PRODUCTS * out_channels * in_channels];
    for o in 0..out_channels {
        for i in 0..in_channels {
            let tap = |row: usize, column: usize| w[((o * 3 + row) * 3 + column) * in_channels + i];
            // 세로로 한 번(G g), 가로로 한 번(… Gᵀ).
            let columns: [[f32; 4]; 3] = std::array::from_fn(|column| {
                stretch([tap(0, column), tap(1, column), tap(2, column)])
            });
            for row in 0..TILE {
                let wide = stretch([columns[0][row], columns[1][row], columns[2][row]]);
                for (column, value) in wide.into_iter().enumerate() {
                    u[((row * TILE + column) * out_channels + o) * in_channels + i] = value;
                }
            }
        }
    }
    u
}
// ANCHOR_END: filter

// ANCHOR: input
/// 네 값을 `Bᵀ`로 바꾼다. 채널마다 같은 계산이므로 슬라이스째로 한다.
fn mix_input(d: [&[f32]; 4], out: [&mut [f32]; 4]) {
    let [o0, o1, o2, o3] = out;
    for i in 0..o0.len() {
        o0[i] = d[0][i] - d[2][i];
        o1[i] = d[1][i] + d[2][i];
        o2[i] = d[2][i] - d[1][i];
        o3[i] = d[1][i] - d[3][i];
    }
}
// ANCHOR_END: input

/// 합성곱의 크기들.
#[derive(Clone, Copy)]
struct Shape {
    height: usize,
    width: usize,
    cin: usize,
    cout: usize,
}

impl Shape {
    // ANCHOR: band
    /// 묶음의 줄 `first`부터 `count`개를 계산해 `out`(출력의 줄들)에 적는다.
    /// `threaded`이면 선형 층의 커널이 스레드를 쓴다.
    #[allow(clippy::too_many_arguments)]
    fn band(
        &self,
        x: &[f32],
        u: &[f32],
        bias: Option<&[f32]>,
        first: usize,
        count: usize,
        out: &mut [f32],
        threaded: bool,
    ) {
        let Self {
            height,
            width,
            cin,
            cout,
        } = *self;
        let tile_columns = width.div_ceil(2);
        let tiles = count * tile_columns;
        // 바꾼 입력 V (16, 묶음, 입력 채널), 곱한 결과 M (16, 묶음, 출력 채널),
        // 그리고 묶음 하나를 바꾸는 데 쓰는 작은 자리 둘.
        let (v_len, m_len) = (PRODUCTS * tiles * cin, PRODUCTS * tiles * cout);
        arena::with_scratch(v_len + m_len + 2 * PRODUCTS * cin, |scratch| {
            let (v, rest) = scratch.split_at_mut(v_len);
            let (m, rest) = rest.split_at_mut(m_len);
            let (patch, half) = rest.split_at_mut(PRODUCTS * cin);
            // 1. 묶음마다 입력 4 × 4를 바꾼다: V = Bᵀ d B.
            for tile in 0..tiles {
                let (row, column) = (2 * (first + tile / tile_columns), 2 * (tile % tile_columns));
                // 묶음이 읽는 입력. 왼쪽 위가 (row - 1, column - 1)이고, 사진
                // 밖은 0이다.
                for r in 0..TILE {
                    for c in 0..TILE {
                        let target = &mut patch[(r * TILE + c) * cin..][..cin];
                        let (source_row, source_column) =
                            ((row + r).wrapping_sub(1), (column + c).wrapping_sub(1));
                        if source_row < height && source_column < width {
                            let pixel = (source_row * width + source_column) * cin;
                            target.copy_from_slice(&x[pixel..pixel + cin]);
                        } else {
                            target.fill(0.0);
                        }
                    }
                }
                // 세로로: 열마다 네 행을 섞는다. half의 (열, 행) 자리에 적는다.
                for c in 0..TILE {
                    let d: [&[f32]; 4] =
                        std::array::from_fn(|r| &patch[(r * TILE + c) * cin..][..cin]);
                    mix_input(d, quarters(&mut half[c * TILE * cin..][..TILE * cin]));
                }
                // 가로로: 행마다 네 열을 섞는다. patch를 다시 써서 (행, 열)
                // 자리에 적고, V의 제자리로 옮긴다.
                for r in 0..TILE {
                    let d: [&[f32]; 4] =
                        std::array::from_fn(|c| &half[(c * TILE + r) * cin..][..cin]);
                    mix_input(d, quarters(&mut patch[r * TILE * cin..][..TILE * cin]));
                }
                for product in 0..PRODUCTS {
                    v[(product * tiles + tile) * cin..][..cin]
                        .copy_from_slice(&patch[product * cin..][..cin]);
                }
            }
            // 2. 곱셈의 자리마다 선형 층 하나: M = V Uᵀ.
            for product in 0..PRODUCTS {
                let vectors = &v[product * tiles * cin..][..tiles * cin];
                let weights = &u[product * cout * cin..][..cout * cin];
                let target = &mut m[product * tiles * cout..][..tiles * cout];
                if threaded {
                    simd::linear(vectors, tiles, weights, cout, cin, target);
                } else {
                    let (backend, walk) = (Backend::detect(), Walk::for_vectors(tiles));
                    simd::linear_with(backend, walk, vectors, tiles, weights, cout, cin, target);
                }
            }
            // 3. 묶음마다 결과 4 × 4를 2 × 2로 되돌린다: Y = Aᵀ M A.
            for tile in 0..tiles {
                let (band_row, column) = (2 * (tile / tile_columns), 2 * (tile % tile_columns));
                let at = |r: usize, c: usize| &m[((r * TILE + c) * tiles + tile) * cout..][..cout];
                for dr in 0..2 {
                    if 2 * first + band_row + dr >= height {
                        continue;
                    }
                    for dc in 0..2 {
                        if column + dc >= width {
                            continue;
                        }
                        let target =
                            &mut out[((band_row + dr) * width + column + dc) * cout..][..cout];
                        for (o, value) in target.iter_mut().enumerate() {
                            // Aᵀ의 두 행은 (1, 1, 1, 0)과 (0, 1, -1, -1)이다.
                            let line = |r: usize| {
                                if dc == 0 {
                                    at(r, 0)[o] + at(r, 1)[o] + at(r, 2)[o]
                                } else {
                                    at(r, 1)[o] - at(r, 2)[o] - at(r, 3)[o]
                                }
                            };
                            let sum = if dr == 0 {
                                line(0) + line(1) + line(2)
                            } else {
                                line(1) - line(2) - line(3)
                            };
                            *value = sum + bias.map_or(0.0, |bias| bias[o]);
                        }
                    }
                }
            }
        });
    }
    // ANCHOR_END: band
}

/// 슬라이스를 길이가 같은 네 조각으로 나눈다.
fn quarters(values: &mut [f32]) -> [&mut [f32]; 4] {
    let (front, back) = values.split_at_mut(values.len() / 2);
    let (a, b) = front.split_at_mut(front.len() / 2);
    let (c, d) = back.split_at_mut(back.len() / 2);
    [a, b, c, d]
}

impl Winograd {
    fn check(input: &[usize], filter: &[usize]) -> Result<(usize, usize, usize, usize, usize)> {
        let (&[batch, height, width, in_channels], &[out_channels, 3, 3, channels]) =
            (input, filter)
        else {
            return Err(Error::Graph(format!(
                "위노그라드 합성곱의 입력 {input:?}는 4차원, 필터 {filter:?}는 3 × 3이어야 합니다"
            )));
        };
        if channels != in_channels {
            return Err(Error::Graph("입력과 필터의 채널이 다릅니다".into()));
        }
        Ok((batch, height, width, in_channels, out_channels))
    }
}

impl Kernel for Winograd {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        if !(2..=3).contains(&inputs.len()) {
            return Err(Error::Graph(format!(
                "CONV_2D의 입력이 {}개입니다",
                inputs.len()
            )));
        }
        let (batch, height, width, _, out_channels) = Self::check(inputs[0], inputs[1])?;
        Ok(vec![vec![batch, height, width, out_channels]])
    }

    // ANCHOR: run
    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (_, height, width, cin, cout) = Self::check(inputs[0].shape, inputs[1].shape)?;
        let u = self
            .filter
            .get_or_init(|| transform_filter(inputs[1].data, cout, cin));
        let bias = inputs.get(2).map(|bias| bias.data);
        let shape = Shape {
            height,
            width,
            cin,
            cout,
        };
        let (tile_rows, tile_columns) = (height.div_ceil(2), width.div_ceil(2));
        let pool = threads::global();
        let wanted = if simd::work(&[tile_rows, tile_columns, PRODUCTS, cin, cout])
            < simd::MIN_PARALLEL_WORK
        {
            1
        } else {
            pool.for_vectors(tile_rows * tile_columns)
        };
        // 묶음의 줄 몇 개를 한 띠로 묶는다. 띠 하나가 스레드 하나의 일이다.
        // 띠가 스레드마다 몇 개씩 돌아가고도 띠 하나에 묶음이 여덟 개 이상이면
        // 띠로 나누고, 아니면(사진이 작으면) 사진 전체를 띠 하나로 두고
        // 선형 층의 커널이 스레드를 쓰게 한다.
        let bands = wanted * simd::PIECES_PER_THREAD;
        let rows_per_band = tile_rows.div_ceil(bands).max(1);
        let split = wanted > 1 && rows_per_band * tile_columns >= 8;
        let rows_per_band = if split { rows_per_band } else { tile_rows };
        let y = &mut *outputs[0].data;
        let image = height * width * cin;
        for (x, y) in inputs[0]
            .data
            .chunks_exact(image)
            .zip(y.chunks_exact_mut(height * width * cout))
        {
            let work = |band: usize, out: &mut [f32]| {
                let first = band * rows_per_band;
                let count = rows_per_band.min(tile_rows - first);
                shape.band(x, u, bias, first, count, out, !split);
                activate(self.activation, out);
            };
            // 묶음의 줄 하나가 출력의 두 줄이다.
            let chunk = rows_per_band * 2 * width * cout;
            pool.pool()
                .run_chunks(y, chunk, if split { wanted } else { 1 }, &work);
        }
        Ok(())
    }
    // ANCHOR_END: run
}

/// 위노그라드 변환을 쓸 수 있는 합성곱인지.
pub fn applies(op: &Op) -> bool {
    matches!(
        op,
        Op::Conv2d {
            padding: Padding::Same,
            stride: [1, 1],
            dilation: [1, 1],
            ..
        }
    )
}

/// 위노그라드 변환을 쓰는 `CONV_2D`의 커널을 만든다. 창이 `3 × 3`이 아니거나
/// 스트라이드가 1이 아닌 합성곱에는 보통의 커널을 만든다.
pub fn conv_2d_winograd(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::Conv2d { activation, .. } = op else {
        return Err(Error::Graph(format!(
            "CONV_2D의 커널에 {}을 주었습니다",
            op.name()
        )));
    };
    if !applies(op) {
        return crate::conv::conv_2d(op);
    }
    Ok(Box::new(WinogradOrDefault {
        fast: Winograd {
            activation,
            filter: OnceLock::new(),
        },
        fallback: crate::conv::conv_2d(op)?,
    }))
}

/// 설정만으로는 창의 크기를 알 수 없다. 필터의 형상을 보고 고른다.
struct WinogradOrDefault {
    fast: Winograd,
    fallback: Box<dyn Kernel>,
}

impl WinogradOrDefault {
    fn pick(&self, filter: &[usize]) -> &dyn Kernel {
        if matches!(filter, [_, 3, 3, _]) {
            &self.fast
        } else {
            &*self.fallback
        }
    }
}

impl Kernel for WinogradOrDefault {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        let filter = inputs.get(1).copied().unwrap_or(&[]);
        self.pick(filter).shapes(inputs)
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        self.pick(inputs[1].shape).run(inputs, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv;
    use badak_core::random::Pcg64;

    fn values(rng: &mut Pcg64, count: usize) -> Vec<f32> {
        (0..count).map(|_| rng.next_f64() as f32 - 0.5).collect()
    }

    fn run(kernel: &dyn Kernel, inputs: &[(&[usize], &[f32])]) -> Vec<f32> {
        let shapes: Vec<&[usize]> = inputs.iter().map(|(shape, _)| *shape).collect();
        let shape = kernel.shapes(&shapes).unwrap().remove(0);
        let mut data = vec![f32::NAN; shape.iter().product()];
        let views: Vec<View<'_>> = inputs
            .iter()
            .map(|&(shape, data)| View { shape, data })
            .collect();
        let mut outputs = [ViewMut {
            shape: &shape,
            data: &mut data,
        }];
        kernel.run(&views, &mut outputs).unwrap();
        data
    }

    #[test]
    fn matches_the_definition() {
        let mut rng = Pcg64::new(28);
        let op = Op::Conv2d {
            padding: Padding::Same,
            stride: [1, 1],
            dilation: [1, 1],
            activation: Activation::Relu,
        };
        // 짝수와 홀수 크기, 사진 여러 장. 홀수이면 마지막 묶음이 사진 밖으로
        // 한 칸 나간다.
        for (batch, height, width, cin, cout) in [
            (1, 8, 8, 3, 4),
            (2, 7, 9, 5, 6),
            (1, 1, 1, 2, 3),
            (1, 14, 14, 16, 8),
        ] {
            let x = values(&mut rng, batch * height * width * cin);
            let w = values(&mut rng, cout * 9 * cin);
            let bias = values(&mut rng, cout);
            let inputs: [(&[usize], &[f32]); 3] = [
                (&[batch, height, width, cin], &x),
                (&[cout, 3, 3, cin], &w),
                (&[cout], &bias),
            ];
            let expected = run(&*conv::conv_2d_direct(&op).unwrap(), &inputs);
            let actual = run(&*conv_2d_winograd(&op).unwrap(), &inputs);
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(&expected) {
                assert!((a - e).abs() < 1e-4, "{a} != {e}");
            }
        }
    }

    #[test]
    fn falls_back_when_the_shape_does_not_fit() {
        let mut rng = Pcg64::new(29);
        let same = Op::Conv2d {
            padding: Padding::Same,
            stride: [1, 1],
            dilation: [1, 1],
            activation: Activation::None,
        };
        // 1 × 1 창은 보통의 커널로 간다.
        let x = values(&mut rng, 4 * 4 * 3);
        let w = values(&mut rng, 2 * 3);
        let inputs: [(&[usize], &[f32]); 2] = [(&[1, 4, 4, 3], &x), (&[2, 1, 1, 3], &w)];
        let expected = run(&*conv::conv_2d_direct(&same).unwrap(), &inputs);
        let actual = run(&*conv_2d_winograd(&same).unwrap(), &inputs);
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-5);
        }
        // 스트라이드가 2이면 처음부터 보통의 커널이다.
        let strided = Op::Conv2d {
            padding: Padding::Same,
            stride: [2, 2],
            dilation: [1, 1],
            activation: Activation::None,
        };
        assert!(!applies(&strided));
        assert!(conv_2d_winograd(&strided).is_ok());
        assert!(conv_2d_winograd(&Op::Relu).is_err());
    }
}
