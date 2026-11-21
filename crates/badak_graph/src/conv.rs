//! 합성곱 커널 (27장).
//!
//! 텐서는 `(장 수, 높이, 너비, 채널)`의 순서로 놓인다(NHWC). 필터는
//! `(출력 채널, 높이, 너비, 입력 채널)`이다. `.tflite`가 쓰는 배치 그대로다.
//!
//! 같은 합성곱을 세 가지로 계산한다. 정의를 그대로 옮긴 것, 창을 행렬로
//! 펼쳐 선형 층에 넘기는 것(im2col), 창이 `1 × 1`일 때 펼칠 것도 없이 곧바로
//! 선형 층에 넘기는 것. 세 방법의 결과는 덧셈의 순서만 다르다.

use badak_core::simd::{Backend, Walk};
use badak_core::{arena, simd, threads};

use crate::error::{Error, Result};
use crate::graph::{Activation, Op, Padding};
use crate::kernels::{Kernel, View, ViewMut, activate};

fn invalid(message: impl Into<String>) -> Error {
    Error::Graph(message.into())
}

// ANCHOR: axis
/// 한 축(세로나 가로)에서 창이 움직이는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    /// 입력의 길이.
    pub input: usize,
    /// 창의 칸 수.
    pub filter: usize,
    /// 스트라이드.
    pub stride: usize,
    /// 창의 칸 사이의 간격.
    pub dilation: usize,
    /// 입력의 앞에 패딩한 길이.
    pub before: usize,
    /// 출력의 길이.
    pub output: usize,
}

impl Axis {
    /// 입력의 길이와 창의 모양에서 출력의 길이와 덧댈 길이를 정한다.
    /// `axis`는 세로 축이면 0, 가로 축이면 1이다. 덧댈 길이가 축마다 따로
    /// 적혀 있을 때만 쓴다.
    pub fn new(
        input: usize,
        filter: usize,
        stride: usize,
        dilation: usize,
        padding: Padding,
        axis: usize,
    ) -> Result<Self> {
        if filter == 0 || stride == 0 || dilation == 0 {
            return Err(invalid("창의 크기, 스트라이드, 간격은 0일 수 없습니다"));
        }
        // 간격까지 넣어 창이 덮는 길이.
        let span = (filter - 1) * dilation + 1;
        let (output, before) = match padding {
            // 패딩하지 않는다. 창이 입력 안에 온전히 들어가는 자리만 센다.
            Padding::Valid => {
                if input < span {
                    return Err(invalid(format!(
                        "길이 {input}인 입력에 길이 {span}인 창이 들어가지 않습니다"
                    )));
                }
                ((input - span) / stride + 1, 0)
            }
            // 출력이 입력 / 스트라이드(올림)이 되도록 패딩한다. 덧댈 길이가 홀수이면
            // 뒤쪽에 하나를 더 둔다.
            Padding::Same => {
                let output = input.div_ceil(stride);
                let needed = (output.max(1) - 1) * stride + span;
                (output, needed.saturating_sub(input) / 2)
            }
            // 30장: 덧댈 길이가 적혀 있다. 패딩한 입력에 `Valid`를 한 것과 같다.
            Padding::Explicit(amounts) => {
                let [before, after] = amounts[axis];
                let padded = input + before + after;
                if padded < span {
                    return Err(invalid(format!(
                        "길이 {padded}인 입력에 길이 {span}인 창이 들어가지 않습니다"
                    )));
                }
                ((padded - span) / stride + 1, before)
            }
        };
        Ok(Self {
            input,
            filter,
            stride,
            dilation,
            before,
            output,
        })
    }

    /// 출력의 `out`번째 자리에서 창의 `tap`번째 칸이 읽는 입력의 위치.
    /// 패딩한 자리를 읽으면 `None`이다.
    #[inline]
    pub fn source(&self, out: usize, tap: usize) -> Option<usize> {
        (out * self.stride + tap * self.dilation)
            .checked_sub(self.before)
            .filter(|&position| position < self.input)
    }
}
// ANCHOR_END: axis

/// 합성곱 하나의 기하. 형상에서 한 번 계산해 두고 커널이 쓴다.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub(crate) batch: usize,
    pub(crate) rows: Axis,
    pub(crate) columns: Axis,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
}

impl Geometry {
    /// 창 하나에 든 값의 수.
    pub(crate) fn patch(&self) -> usize {
        self.rows.filter * self.columns.filter * self.in_channels
    }

    /// 사진 한 장의 출력 자리 수.
    pub(crate) fn positions(&self) -> usize {
        self.rows.output * self.columns.output
    }
}

// ANCHOR: method
/// 합성곱을 계산하는 방법.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// 정의를 그대로 옮긴 일곱 겹의 반복문.
    Direct,
    /// 창들을 행렬로 펼친 뒤 선형 층의 커널을 부른다.
    Im2col,
    /// 창이 `1 × 1`이고 스트라이드가 1이면 입력을 그대로 선형 층에 넘기고, 아니면
    /// 펼친다. 27장의 방법이다.
    Whole,
    /// 출력을 조각으로 나누고, 조각마다 펼쳐 계산한다. 조각을 스레드에
    /// 나눈다 (28장).
    Tiled,
    /// 펼치지 않는다. 창의 칸마다 입력의 어디를 읽을지만 적어 두고, 커널이
    /// 그 자리를 직접 읽는다 (28장).
    Indirect,
    /// 모양을 보고 가장 빠른 방법을 고른다 (28장).
    Auto,
}

/// 합성곱.
pub struct Conv2d {
    padding: Padding,
    stride: [usize; 2],
    dilation: [usize; 2],
    pub(crate) activation: Activation,
    method: Method,
}
// ANCHOR_END: method

impl Conv2d {
    pub(crate) fn geometry(&self, input: &[usize], filter: &[usize]) -> Result<Geometry> {
        let (&[batch, height, width, in_channels], &[out_channels, rows, columns, channels]) =
            (input, filter)
        else {
            return Err(invalid(format!(
                "CONV_2D의 입력 {input:?}와 필터 {filter:?}는 4차원이어야 합니다"
            )));
        };
        if channels != in_channels {
            return Err(Error::Unsupported(format!(
                "입력의 채널 {in_channels}과 필터의 채널 {channels}이 다른 CONV_2D (묶음 합성곱)"
            )));
        }
        Ok(Geometry {
            batch,
            rows: Axis::new(
                height,
                rows,
                self.stride[0],
                self.dilation[0],
                self.padding,
                0,
            )?,
            columns: Axis::new(
                width,
                columns,
                self.stride[1],
                self.dilation[1],
                self.padding,
                1,
            )?,
            in_channels,
            out_channels,
        })
    }
}

impl Kernel for Conv2d {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        if !(2..=3).contains(&inputs.len()) {
            return Err(invalid(format!(
                "CONV_2D의 입력이 {}개입니다",
                inputs.len()
            )));
        }
        let geometry = self.geometry(inputs[0], inputs[1])?;
        if let Some(bias) = inputs.get(2)
            && *bias != [geometry.out_channels]
        {
            return Err(invalid(format!(
                "CONV_2D의 편향 {bias:?}가 ({})이 아닙니다",
                geometry.out_channels
            )));
        }
        Ok(vec![vec![
            geometry.batch,
            geometry.rows.output,
            geometry.columns.output,
            geometry.out_channels,
        ]])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let geometry = self.geometry(inputs[0].shape, inputs[1].shape)?;
        let (x, w) = (inputs[0].data, inputs[1].data);
        let bias = inputs.get(2).map(|bias| bias.data);
        let y = &mut *outputs[0].data;
        let pointwise = geometry.patch() == geometry.in_channels
            && self.stride == [1, 1]
            && geometry.rows.before == 0
            && geometry.columns.before == 0;
        match self.method {
            Method::Direct => direct(&geometry, x, w, bias, y),
            Method::Whole if pointwise => self::pointwise(&geometry, x, w, bias, y),
            Method::Im2col | Method::Whole => im2col(&geometry, x, w, bias, y),
            // 조각 하나에 벡터가 넉넉히 돌아갈 때만 조각낸다. 사진이 작고
            // 필터가 큰 뒤쪽 층에서는 27장의 방법(가중치의 행을 나누기)이
            // 낫다.
            Method::Auto if !worth_tiling(&geometry) => {
                if pointwise {
                    self::pointwise(&geometry, x, w, bias, y);
                } else {
                    im2col(&geometry, x, w, bias, y);
                }
            }
            Method::Indirect => {
                indirect(&geometry, x, w, bias, y, self.activation);
                return Ok(());
            }
            Method::Tiled | Method::Auto => {
                // 조각마다 편향과 활성화 함수까지 끝낸다.
                tiled(&geometry, x, w, bias, y, self.activation, pointwise);
                return Ok(());
            }
        }
        activate(self.activation, y);
        Ok(())
    }
}

// ANCHOR: direct
/// 정의 그대로. 출력의 자리마다, 출력 채널마다, 창의 칸마다, 입력 채널마다.
fn direct(g: &Geometry, x: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    let (width, channels) = (g.columns.input, g.in_channels);
    let mut out = 0;
    for image in 0..g.batch {
        let x = &x[image * g.rows.input * width * channels..];
        for row in 0..g.rows.output {
            for column in 0..g.columns.output {
                for channel in 0..g.out_channels {
                    let filter = &w[channel * g.patch()..];
                    let mut sum = bias.map_or(0.0, |bias| bias[channel]);
                    for tap_row in 0..g.rows.filter {
                        // 패딩한 자리의 값은 0이므로 건너뛰면 된다.
                        let Some(source_row) = g.rows.source(row, tap_row) else {
                            continue;
                        };
                        for tap_column in 0..g.columns.filter {
                            let Some(source_column) = g.columns.source(column, tap_column) else {
                                continue;
                            };
                            let pixel = (source_row * width + source_column) * channels;
                            let tap = (tap_row * g.columns.filter + tap_column) * channels;
                            for c in 0..channels {
                                sum += x[pixel + c] * filter[tap + c];
                            }
                        }
                    }
                    y[out] = sum;
                    out += 1;
                }
            }
        }
    }
}
// ANCHOR_END: direct

// ANCHOR: unfold
/// 사진 한 장의 창들을 행렬로 펼친다(im2col). 행 하나가 창 하나다.
///
/// 창 안의 값을 (창의 행, 창의 열, 채널)의 순서로 늘어놓는다. 필터의 한 출력
/// 채널이 놓인 순서와 같아서, 펼친 행과 필터의 내적이 곧 합성곱의 값이다.
fn unfold(g: &Geometry, x: &[f32], patches: &mut [f32]) {
    unfold_range(g, x, 0, g.positions(), patches);
}

/// [`unfold`]와 같되, 출력의 자리 `first`부터 `count`개만 펼친다.
pub(crate) fn unfold_range(
    g: &Geometry,
    x: &[f32],
    first: usize,
    count: usize,
    patches: &mut [f32],
) {
    let (width, channels) = (g.columns.input, g.in_channels);
    let mut out = 0;
    for position in first..first + count {
        let (row, column) = (position / g.columns.output, position % g.columns.output);
        for tap_row in 0..g.rows.filter {
            let source_row = g.rows.source(row, tap_row);
            for tap_column in 0..g.columns.filter {
                let target = &mut patches[out..out + channels];
                match (source_row, g.columns.source(column, tap_column)) {
                    (Some(source_row), Some(source_column)) => {
                        let pixel = (source_row * width + source_column) * channels;
                        target.copy_from_slice(&x[pixel..pixel + channels]);
                    }
                    // 패딩한 자리. 0을 적는다. 자리에 남아 있던 값을 지워야
                    // 한다.
                    _ => target.fill(0.0),
                }
                out += channels;
            }
        }
    }
}
// ANCHOR_END: unfold

// ANCHOR: im2col
/// 펼친 행렬에 선형 층의 커널을 부른다.
fn im2col(g: &Geometry, x: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    let (positions, patch) = (g.positions(), g.patch());
    let image = g.rows.input * g.columns.input * g.in_channels;
    // 펼친 행렬은 텐서가 아니다. 이 커널 안에서만 쓰고 버리므로 스레드가
    // 돌려 쓰는 자리를 빌린다(22.10절).
    arena::with_scratch(positions * patch, |patches| {
        for (x, y) in x
            .chunks_exact(image)
            .zip(y.chunks_exact_mut(positions * g.out_channels))
        {
            unfold(g, x, patches);
            simd::linear(patches, positions, w, g.out_channels, patch, y);
        }
    });
    add_bias(bias, y);
}
// ANCHOR_END: im2col

// ANCHOR: pointwise
/// 창이 `1 × 1`이면 창 하나가 픽셀 하나다. 입력이 이미 "펼친 행렬"이다.
fn pointwise(g: &Geometry, x: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    let pixels = g.batch * g.positions();
    simd::linear(x, pixels, w, g.out_channels, g.in_channels, y);
    add_bias(bias, y);
}
// ANCHOR_END: pointwise

// ANCHOR: tiled
/// 조각 하나가 펼친 행렬로 쓸 바이트의 기준. 이보다 작게 조각낸다.
pub(crate) fn tile_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("BADAK_CONV_TILE")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(64 * 1024)
    })
}

/// 조각 하나에 돌아가야 하는 벡터의 수. 커널이 벡터를 여덟 개 이상 받아야
/// 제 속도를 낸다(17.5절).
const MIN_TILE_VECTORS: usize = 8;

/// 이 합성곱에 쓸 스레드의 수.
pub(crate) fn wanted_threads(g: &Geometry) -> usize {
    if simd::work(&[g.positions(), g.patch(), g.out_channels]) < simd::MIN_PARALLEL_WORK {
        1
    } else {
        threads::global().for_vectors(g.positions())
    }
}

// ANCHOR: choose
/// 출력을 조각내는 쪽이 나은가.
///
/// 스레드마다 조각을 몇 개씩 주고도(18.6절) 조각 하나에 벡터가 여덟 개
/// 이상 돌아가면 조각낸다. 스레드가 12개이면 출력의 자리가 384개
/// (`20 × 20`쯤) 이상일 때다. 그보다 작은 사진에서는 조각이 너무 잘아져서,
/// 스레드마다 가중치 전체를 읽는 수고만 남는다.
fn worth_tiling(g: &Geometry) -> bool {
    let tiles = wanted_threads(g) * simd::PIECES_PER_THREAD;
    g.positions() / tiles >= MIN_TILE_VECTORS
}
// ANCHOR_END: choose

/// 출력의 자리를 조각으로 나누어, 조각마다 펼치고 곱한다.
///
/// 27장의 `im2col`은 사진 한 장의 창을 모두 펼친 뒤 커널을 한 번 불렀다.
/// 여기서는 일의 단위가 조각이다. 스레드 하나가 조각 하나를 맡아 자기
/// 자리에 펼치고, 커널을 스레드 하나짜리로 부르고, 편향과 활성화 함수까지
/// 끝낸다. 펼친 행렬이 캐시에 들어가고, 펼치는 일도 스레드에 나뉜다.
fn tiled(
    g: &Geometry,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    activation: Activation,
    pointwise: bool,
) {
    let (positions, patch, channels) = (g.positions(), g.patch(), g.out_channels);
    if positions == 0 || channels == 0 {
        return;
    }
    let pool = threads::global();
    // 조각 하나에 담을 자리의 수. 커널이 벡터를 여덟 개 이상 받아야 제 속도를
    // 내므로(17.5절) 그보다 작게 자르지 않는다.
    let fit = (tile_bytes() / (patch * size_of::<f32>())).max(MIN_TILE_VECTORS);
    let wanted = wanted_threads(g);
    // 스레드마다 조각이 몇 개씩은 돌아가게 한다(18.6절).
    let tile = fit
        .min(positions.div_ceil(wanted * simd::PIECES_PER_THREAD))
        .max(1);
    let (backend, image) = (
        Backend::detect(),
        g.rows.input * g.columns.input * g.in_channels,
    );
    for (x, y) in x
        .chunks_exact(image)
        .zip(y.chunks_exact_mut(positions * channels))
    {
        let work = |index: usize, out: &mut [f32]| {
            let count = out.len() / channels;
            let first = index * tile;
            let walk = Walk::for_vectors(count);
            if pointwise {
                let vectors = &x[first * patch..(first + count) * patch];
                simd::linear_with(backend, walk, vectors, count, w, channels, patch, out);
            } else {
                arena::with_scratch(count * patch, |patches| {
                    unfold_range(g, x, first, count, patches);
                    simd::linear_with(backend, walk, patches, count, w, channels, patch, out);
                });
            }
            add_bias(bias, out);
            activate(activation, out);
        };
        pool.pool().run_chunks(y, tile * channels, wanted, &work);
    }
}
// ANCHOR_END: tiled

// ANCHOR: indirect_dots
/// 한 번에 누산하는 값의 수. 17.3절의 정의와 같은 여덟이다.
const LANES: usize = 8;

/// 창 둘과 필터 넷의 내적 여덟 개를 한꺼번에 계산한다. 창은 이어진 메모리가
/// 아니라 "칸마다의 픽셀"들의 목록으로 받는다.
///
/// `taps[t]`가 창의 `t`번째 칸이 읽는 픽셀 둘(출력의 자리 둘에 하나씩)이고,
/// `filters[n]`이 필터 하나 전체다. 필터 안에서 `t`번째 칸의 가중치는
/// `t × 채널`번째부터 있다. 17.4절의 커널과 같은 모양이고, 입력을 읽는
/// 주소만 칸마다 바뀐다.
fn indirect_dots(taps: &[[&[f32]; 2]], filters: [&[f32]; 4], channels: usize) -> [[f32; 4]; 2] {
    let mut sums = [[[0.0_f32; LANES]; 4]; 2];
    let mut tails = [[0.0_f32; 4]; 2];
    let full = channels - channels % LANES;
    for (tap, pixels) in taps.iter().enumerate() {
        let weights: [&[f32]; 4] =
            std::array::from_fn(|n| &filters[n][tap * channels..(tap + 1) * channels]);
        for start in (0..full).step_by(LANES) {
            let x: [&[f32; LANES]; 2] = std::array::from_fn(|m| {
                pixels[m][start..start + LANES]
                    .try_into()
                    .expect("길이가 여덟이다")
            });
            for n in 0..4 {
                let w: &[f32; LANES] = weights[n][start..start + LANES]
                    .try_into()
                    .expect("길이가 여덟이다");
                for m in 0..2 {
                    for lane in 0..LANES {
                        sums[m][n][lane] = x[m][lane].mul_add(w[lane], sums[m][n][lane]);
                    }
                }
            }
        }
        for index in full..channels {
            for m in 0..2 {
                for n in 0..4 {
                    tails[m][n] = pixels[m][index].mul_add(weights[n][index], tails[m][n]);
                }
            }
        }
    }
    std::array::from_fn(|m| std::array::from_fn(|n| simd::reduce(sums[m][n]) + tails[m][n]))
}
// ANCHOR_END: indirect_dots

// ANCHOR: indirect
/// 펼치지 않는 합성곱. 출력의 한 줄이 일의 단위다.
///
/// 출력의 자리 둘마다 "창의 칸이 읽는 픽셀"의 목록을 만든다. 픽셀을 복사하지
/// 않고 그 자리를 가리키기만 한다. 패딩한 자리는 0으로 찬 픽셀 하나를 함께
/// 가리킨다. 목록의 크기는 창의 칸 수이고 채널의 수와 상관이 없다.
fn indirect(
    g: &Geometry,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    activation: Activation,
) {
    let (channels, out_channels, patch) = (g.in_channels, g.out_channels, g.patch());
    let taps = g.rows.filter * g.columns.filter;
    let zeros = vec![0.0_f32; channels];
    let image = g.rows.input * g.columns.input * channels;
    let line = g.columns.output * out_channels;
    let pool = threads::global();
    let wanted = if simd::work(&[g.positions(), patch, out_channels]) < simd::MIN_PARALLEL_WORK {
        1
    } else {
        pool.for_vectors(g.rows.output)
    };
    for (x, y) in x
        .chunks_exact(image)
        .zip(y.chunks_exact_mut(g.positions() * out_channels))
    {
        let pixel = |row: usize, column: usize, tap: usize| -> &[f32] {
            let (tap_row, tap_column) = (tap / g.columns.filter, tap % g.columns.filter);
            match (
                g.rows.source(row, tap_row),
                g.columns.source(column, tap_column),
            ) {
                (Some(source_row), Some(source_column)) => {
                    &x[(source_row * g.columns.input + source_column) * channels..][..channels]
                }
                _ => &zeros,
            }
        };
        let work = |row: usize, out: &mut [f32]| {
            // 이 줄의 자리 둘이 읽을 픽셀들. 스택에 둘 수 있을 만큼 작다.
            let mut list: Vec<[&[f32]; 2]> = vec![[&zeros[..], &zeros[..]]; taps];
            for column in (0..g.columns.output).step_by(2) {
                // 자리가 홀수 개이면 마지막 짝의 둘째는 첫째와 같은 것을 읽는다.
                let second = (column + 1).min(g.columns.output - 1);
                for (tap, entry) in list.iter_mut().enumerate() {
                    *entry = [pixel(row, column, tap), pixel(row, second, tap)];
                }
                let mut channel = 0;
                while channel < out_channels {
                    // 필터가 넷으로 나누어떨어지지 않으면 마지막 것을 거듭 쓴다.
                    let filters: [&[f32]; 4] = std::array::from_fn(|n| {
                        let index = (channel + n).min(out_channels - 1);
                        &w[index * patch..(index + 1) * patch]
                    });
                    let sums = indirect_dots(&list, filters, channels);
                    for n in 0..4.min(out_channels - channel) {
                        let offset = bias.map_or(0.0, |bias| bias[channel + n]);
                        out[column * out_channels + channel + n] = sums[0][n] + offset;
                        out[second * out_channels + channel + n] = sums[1][n] + offset;
                    }
                    channel += 4;
                }
            }
            activate(activation, out);
        };
        pool.pool().run_chunks(y, line, wanted, &work);
    }
}
// ANCHOR_END: indirect

/// 출력의 자리마다 채널별 편향을 더한다.
pub(crate) fn add_bias(bias: Option<&[f32]>, y: &mut [f32]) {
    let Some(bias) = bias else {
        return;
    };
    for pixel in y.chunks_exact_mut(bias.len()) {
        for (value, &b) in pixel.iter_mut().zip(bias) {
            *value += b;
        }
    }
}

// ANCHOR: depthwise
/// 채널마다 따로 하는 합성곱. 출력 채널 하나가 입력 채널 하나만 본다.
pub struct DepthwiseConv2d {
    padding: Padding,
    stride: [usize; 2],
    dilation: [usize; 2],
    multiplier: usize,
    activation: Activation,
}

impl DepthwiseConv2d {
    fn geometry(&self, input: &[usize], filter: &[usize]) -> Result<Geometry> {
        let (&[batch, height, width, in_channels], &[1, rows, columns, out_channels]) =
            (input, filter)
        else {
            return Err(invalid(format!(
                "DEPTHWISE_CONV_2D의 입력 {input:?}는 4차원, 필터 {filter:?}는 (1, 높이, 너비, 채널)이어야 합니다"
            )));
        };
        if out_channels != in_channels * self.multiplier {
            return Err(invalid(format!(
                "DEPTHWISE_CONV_2D의 출력 채널 {out_channels}이 입력 채널 {in_channels} × {}가 아닙니다",
                self.multiplier
            )));
        }
        Ok(Geometry {
            batch,
            rows: Axis::new(
                height,
                rows,
                self.stride[0],
                self.dilation[0],
                self.padding,
                0,
            )?,
            columns: Axis::new(
                width,
                columns,
                self.stride[1],
                self.dilation[1],
                self.padding,
                1,
            )?,
            in_channels,
            out_channels,
        })
    }
}

impl Kernel for DepthwiseConv2d {
    fn shapes(&self, inputs: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        if !(2..=3).contains(&inputs.len()) {
            return Err(invalid(format!(
                "DEPTHWISE_CONV_2D의 입력이 {}개입니다",
                inputs.len()
            )));
        }
        let geometry = self.geometry(inputs[0], inputs[1])?;
        if let Some(bias) = inputs.get(2)
            && *bias != [geometry.out_channels]
        {
            return Err(invalid("DEPTHWISE_CONV_2D의 편향의 길이가 맞지 않습니다"));
        }
        Ok(vec![vec![
            geometry.batch,
            geometry.rows.output,
            geometry.columns.output,
            geometry.out_channels,
        ]])
    }

    fn run(&self, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let g = self.geometry(inputs[0].shape, inputs[1].shape)?;
        let (x, w) = (inputs[0].data, inputs[1].data);
        let bias = inputs.get(2).map(|bias| bias.data);
        let y = &mut *outputs[0].data;
        let (width, channels, out_channels) = (g.columns.input, g.in_channels, g.out_channels);
        let image = g.rows.input * width * channels;
        let line = g.columns.output * out_channels;
        let pool = threads::global();
        let wanted = if simd::work(&[g.positions(), g.patch(), self.multiplier])
            < simd::MIN_PARALLEL_WORK / 8
        {
            1
        } else {
            pool.for_vectors(g.rows.output)
        };
        for (x, y) in x
            .chunks_exact(image)
            .zip(y.chunks_exact_mut(g.positions() * out_channels))
        {
            // 28장: 출력의 한 줄이 일의 단위다. 줄마다 따로 계산할 수 있다.
            let work = |row: usize, out: &mut [f32]| {
                for (column, sums) in out.chunks_exact_mut(out_channels).enumerate() {
                    match bias {
                        Some(bias) => sums.copy_from_slice(bias),
                        None => sums.fill(0.0),
                    }
                    for tap_row in 0..g.rows.filter {
                        let Some(source_row) = g.rows.source(row, tap_row) else {
                            continue;
                        };
                        for tap_column in 0..g.columns.filter {
                            let Some(source_column) = g.columns.source(column, tap_column) else {
                                continue;
                            };
                            let pixel =
                                &x[(source_row * width + source_column) * channels..][..channels];
                            let tap = (tap_row * g.columns.filter + tap_column) * out_channels;
                            let weights = &w[tap..tap + out_channels];
                            if self.multiplier == 1 {
                                // 채널들이 나란히 놓여 있어서 이 반복문이
                                // 벡터 명령으로 바뀐다.
                                for ((sum, &value), &weight) in
                                    sums.iter_mut().zip(pixel).zip(weights)
                                {
                                    *sum += value * weight;
                                }
                            } else {
                                for (index, (sum, &weight)) in
                                    sums.iter_mut().zip(weights).enumerate()
                                {
                                    *sum += pixel[index / self.multiplier] * weight;
                                }
                            }
                        }
                    }
                }
                activate(self.activation, out);
            };
            pool.pool().run_chunks(y, line, wanted, &work);
        }
        Ok(())
    }
}
// ANCHOR_END: depthwise

/// 노드의 설정으로 합성곱 커널을 만든다.
pub(crate) fn conv_kernel(op: &Op, method: Method) -> Result<Conv2d> {
    let &Op::Conv2d {
        padding,
        stride,
        dilation,
        activation,
    } = op
    else {
        return Err(invalid(format!(
            "CONV_2D의 커널에 {}을 주었습니다",
            op.name()
        )));
    };
    Ok(Conv2d {
        padding,
        stride,
        dilation,
        activation,
        method,
    })
}

fn conv_with(op: &Op, method: Method) -> Result<Box<dyn Kernel>> {
    Ok(Box::new(conv_kernel(op, method)?))
}

/// `CONV_2D`의 커널을 만든다. 창의 모양을 보고 방법을 고른다.
pub fn conv_2d(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Auto)
}

/// 언제나 조각마다 펼치는 `CONV_2D`의 커널을 만든다. 견주는 데 쓴다.
pub fn conv_2d_tiled(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Tiled)
}

/// 펼치지 않는 `CONV_2D`의 커널을 만든다. 견주는 데 쓴다.
pub fn conv_2d_indirect(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Indirect)
}

/// 27장의 `CONV_2D` 커널을 만든다. 사진 한 장을 통째로 펼치고, `1 × 1`은
/// 펼치지 않는다. 견주는 데 쓴다.
pub fn conv_2d_whole(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Whole)
}

/// 정의를 그대로 옮긴 `CONV_2D`의 커널을 만든다. 견주는 데 쓴다.
pub fn conv_2d_direct(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Direct)
}

/// 언제나 창을 펼치는 `CONV_2D`의 커널을 만든다. 견주는 데 쓴다.
pub fn conv_2d_im2col(op: &Op) -> Result<Box<dyn Kernel>> {
    conv_with(op, Method::Im2col)
}

/// `DEPTHWISE_CONV_2D`의 커널을 만든다.
pub fn depthwise_conv_2d(op: &Op) -> Result<Box<dyn Kernel>> {
    let &Op::DepthwiseConv2d {
        padding,
        stride,
        dilation,
        multiplier,
        activation,
    } = op
    else {
        return Err(invalid(format!(
            "DEPTHWISE_CONV_2D의 커널에 {}을 주었습니다",
            op.name()
        )));
    };
    Ok(Box::new(DepthwiseConv2d {
        padding,
        stride,
        dilation,
        multiplier,
        activation,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use badak_core::random::Pcg64;

    fn values(rng: &mut Pcg64, count: usize) -> Vec<f32> {
        (0..count).map(|_| rng.next_f64() as f32 - 0.5).collect()
    }

    fn run(kernel: &dyn Kernel, inputs: &[(&[usize], &[f32])]) -> (Vec<usize>, Vec<f32>) {
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
        (shape, data)
    }

    fn conv(padding: Padding, stride: usize, dilation: usize, method: Method) -> Conv2d {
        Conv2d {
            padding,
            stride: [stride, stride],
            dilation: [dilation, dilation],
            activation: Activation::None,
            method,
        }
    }

    #[test]
    fn axis_follows_the_two_padding_rules() {
        // 패딩하지 않으면 (입력 - 창) / 스트라이드 + 1.
        let valid = Axis::new(230, 7, 2, 1, Padding::Valid, 0).unwrap();
        assert_eq!((valid.output, valid.before), (112, 0));
        // 패딩하면 입력 / 스트라이드(올림). 3칸짜리 창은 앞뒤에 하나씩.
        let same = Axis::new(56, 3, 1, 1, Padding::Same, 0).unwrap();
        assert_eq!((same.output, same.before), (56, 1));
        assert_eq!(same.source(0, 0), None);
        assert_eq!(same.source(0, 1), Some(0));
        assert_eq!(same.source(55, 2), None);
        // 덧댈 길이가 홀수이면 뒤쪽에 하나 더. 길이 4, 창 3, 스트라이드 2 → 출력 2,
        // 덧댈 길이 1은 모두 뒤로 간다.
        let odd = Axis::new(4, 3, 2, 1, Padding::Same, 0).unwrap();
        assert_eq!((odd.output, odd.before), (2, 0));
        assert_eq!(odd.source(1, 2), None);
        // 간격이 2이면 3칸짜리 창이 5칸을 덮는다.
        let dilated = Axis::new(9, 3, 1, 2, Padding::Valid, 0).unwrap();
        assert_eq!(dilated.output, 5);
        assert!(Axis::new(2, 3, 1, 1, Padding::Valid, 0).is_err());
        assert!(Axis::new(2, 3, 0, 1, Padding::Same, 0).is_err());
    }

    #[test]
    fn direct_matches_a_hand_computed_case() {
        // 3 × 3 입력, 2 × 2 창, 채널 하나. 창의 값이 모두 1이면 합이다.
        let x: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let w: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        let kernel = conv(Padding::Valid, 1, 1, Method::Direct);
        let (shape, y) = run(
            &kernel,
            &[(&[1, 3, 3, 1], x), (&[1, 2, 2, 1], w), (&[1], &[0.5])],
        );
        assert_eq!(shape, [1, 2, 2, 1]);
        assert_eq!(y, [12.5, 16.5, 24.5, 28.5]);
        // 패딩하면 가장자리의 창은 값이 모자란다.
        let kernel = conv(Padding::Same, 1, 1, Method::Direct);
        let (shape, y) = run(&kernel, &[(&[1, 3, 3, 1], x), (&[1, 2, 2, 1], w)]);
        assert_eq!(shape, [1, 3, 3, 1]);
        assert_eq!(y, [12.0, 16.0, 9.0, 24.0, 28.0, 15.0, 15.0, 17.0, 9.0]);
    }

    #[test]
    fn three_methods_agree() {
        let mut rng = Pcg64::new(27);
        let cases = [
            // (높이, 너비, 입력 채널, 출력 채널, 창, 스트라이드, 간격)
            (9, 7, 3, 5, 3, 1, 1),
            (10, 10, 4, 6, 3, 2, 1),
            (8, 8, 8, 4, 1, 1, 1),
            (8, 9, 8, 4, 1, 2, 1),
            (12, 11, 2, 3, 3, 1, 2),
            (15, 15, 3, 8, 7, 2, 1),
        ];
        for (height, width, cin, cout, filter, stride, dilation) in cases {
            for padding in [Padding::Same, Padding::Valid] {
                let x = values(&mut rng, 2 * height * width * cin);
                let w = values(&mut rng, cout * filter * filter * cin);
                let bias = values(&mut rng, cout);
                let inputs: [(&[usize], &[f32]); 3] = [
                    (&[2, height, width, cin], &x),
                    (&[cout, filter, filter, cin], &w),
                    (&[cout], &bias),
                ];
                let (shape, expected) =
                    run(&conv(padding, stride, dilation, Method::Direct), &inputs);
                for method in [
                    Method::Im2col,
                    Method::Whole,
                    Method::Tiled,
                    Method::Indirect,
                    Method::Auto,
                ] {
                    let (same_shape, actual) =
                        run(&conv(padding, stride, dilation, method), &inputs);
                    assert_eq!(same_shape, shape);
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!((a - e).abs() < 1e-5, "{method:?}: {a} != {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_equals_a_convolution_with_zeroed_cross_terms() {
        // 깊이별 합성곱은 "다른 채널의 가중치가 모두 0인" 합성곱이다.
        let mut rng = Pcg64::new(28);
        let (height, width, channels, filter) = (9, 8, 6, 3);
        for (stride, padding) in [(1, Padding::Same), (2, Padding::Same), (1, Padding::Valid)] {
            let x = values(&mut rng, height * width * channels);
            let w = values(&mut rng, filter * filter * channels);
            let bias = values(&mut rng, channels);
            let mut full = vec![0.0_f32; channels * filter * filter * channels];
            for channel in 0..channels {
                for tap in 0..filter * filter {
                    full[(channel * filter * filter + tap) * channels + channel] =
                        w[tap * channels + channel];
                }
            }
            let depthwise = DepthwiseConv2d {
                padding,
                stride: [stride, stride],
                dilation: [1, 1],
                multiplier: 1,
                activation: Activation::Relu6,
            };
            let (shape, actual) = run(
                &depthwise,
                &[
                    (&[1, height, width, channels], &x),
                    (&[1, filter, filter, channels], &w),
                    (&[channels], &bias),
                ],
            );
            let mut reference = conv(padding, stride, 1, Method::Direct);
            reference.activation = Activation::Relu6;
            let (same_shape, expected) = run(
                &reference,
                &[
                    (&[1, height, width, channels], &x),
                    (&[channels, filter, filter, channels], &full),
                    (&[channels], &bias),
                ],
            );
            assert_eq!(shape, same_shape);
            for (a, e) in actual.iter().zip(&expected) {
                assert!((a - e).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn depthwise_multiplier_repeats_input_channels() {
        // 입력 채널 둘, 배수 2. 출력 채널 0과 1이 입력 채널 0을 본다.
        let x: &[f32] = &[1.0, 10.0];
        let w: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let kernel = DepthwiseConv2d {
            padding: Padding::Valid,
            stride: [1, 1],
            dilation: [1, 1],
            multiplier: 2,
            activation: Activation::None,
        };
        let (shape, y) = run(&kernel, &[(&[1, 1, 1, 2], x), (&[1, 1, 1, 4], w)]);
        assert_eq!(shape, [1, 1, 1, 4]);
        assert_eq!(y, [1.0, 2.0, 30.0, 40.0]);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let kernel = conv(Padding::Same, 1, 1, Method::Auto);
        // 채널이 다르다.
        assert!(kernel.shapes(&[&[1, 8, 8, 3], &[4, 3, 3, 2]]).is_err());
        // 4차원이 아니다.
        assert!(kernel.shapes(&[&[8, 8, 3], &[4, 3, 3, 3]]).is_err());
        // 편향의 길이가 다르다.
        assert!(
            kernel
                .shapes(&[&[1, 8, 8, 3], &[4, 3, 3, 3], &[5]])
                .is_err()
        );
        assert!(conv_2d(&Op::Relu).is_err());
        assert!(depthwise_conv_2d(&Op::Relu).is_err());
    }
}
