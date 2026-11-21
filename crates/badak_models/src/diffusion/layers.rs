//! 확산 모델의 부품들이 함께 쓰는 층 (35장).
//!
//! 특징 맵을 `(자리 수, 채널)`의 줄들로 담는다. 자리는 `세로 × 가로`를
//! 줄 순서로 편 것이다. 이렇게 두면 `1 × 1` 합성곱이 그대로 선형 층이고,
//! 어텐션이 보는 토큰들이 그대로 줄들이다.

use badak_core::{Tensor, nn, prob, profile, simd, threads};
use badak_formats::safetensors::SafeTensors;
use badak_formats::{Error, Result};

pub(crate) fn invalid(message: String) -> Error {
    Error::Parse {
        format: "확산 모델",
        offset: 0,
        message,
    }
}

// ANCHOR: map
/// 특징 맵. 값은 `(세로 × 가로, 채널)`의 순서다.
#[derive(Debug, Clone)]
pub struct Map {
    /// 세로.
    pub height: usize,
    /// 가로.
    pub width: usize,
    /// 채널의 수.
    pub channels: usize,
    /// 값들.
    pub data: Vec<f32>,
}

impl Map {
    /// 값들로 특징 맵을 만든다.
    ///
    /// # Panics
    ///
    /// 값의 수가 맞지 않으면 공황에 빠진다.
    pub fn new(height: usize, width: usize, channels: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            height * width * channels,
            "값의 수가 맞지 않습니다"
        );
        Self {
            height,
            width,
            channels,
            data,
        }
    }

    /// PyTorch의 순서 `(채널, 세로, 가로)`로 놓인 값들에서 만든다.
    pub fn from_planes(height: usize, width: usize, channels: usize, planes: &[f32]) -> Self {
        let cells = height * width;
        let mut data = vec![0.0; planes.len()];
        for (channel, plane) in planes.chunks_exact(cells).enumerate() {
            for (cell, &value) in plane.iter().enumerate() {
                data[cell * channels + channel] = value;
            }
        }
        Self::new(height, width, channels, data)
    }

    /// PyTorch의 순서 `(채널, 세로, 가로)`로 바꿔 돌려준다.
    pub fn to_planes(&self) -> Vec<f32> {
        let cells = self.cells();
        let mut planes = vec![0.0; self.data.len()];
        for (cell, row) in self.data.chunks_exact(self.channels).enumerate() {
            for (channel, &value) in row.iter().enumerate() {
                planes[channel * cells + cell] = value;
            }
        }
        planes
    }

    /// 자리의 수.
    pub fn cells(&self) -> usize {
        self.height * self.width
    }

    /// 같은 크기의 맵을 더한다.
    pub(crate) fn add(&mut self, other: &Self) {
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += b);
    }

    /// 채널 방향으로 이어 붙인다. U-Net의 건너뛰는 길이 쓴다.
    pub(crate) fn concat(&self, other: &Self) -> Self {
        let channels = self.channels + other.channels;
        let mut data = Vec::with_capacity(self.cells() * channels);
        for (a, b) in self
            .data
            .chunks_exact(self.channels)
            .zip(other.data.chunks_exact(other.channels))
        {
            data.extend_from_slice(a);
            data.extend_from_slice(b);
        }
        Self::new(self.height, self.width, channels, data)
    }

    // ANCHOR: upsample
    /// 가로와 세로를 두 배로 늘린다. 자리마다 값을 네 번 적는다.
    pub(crate) fn upsample(&self) -> Self {
        let (height, width, channels) = (self.height * 2, self.width * 2, self.channels);
        let mut data = Vec::with_capacity(height * width * channels);
        for y in 0..height {
            for x in 0..width {
                let from = ((y / 2) * self.width + x / 2) * channels;
                data.extend_from_slice(&self.data[from..from + channels]);
            }
        }
        Self::new(height, width, channels, data)
    }
    // ANCHOR_END: upsample
}
// ANCHOR_END: map

/// 한 번에 늘어놓을 창들의 값의 수. 이보다 큰 합성곱은 여러 토막으로
/// 나눠 계산한다.
const STRIP_VALUES: usize = 1 << 22;

// ANCHOR: kernel
/// 선형 층이나 합성곱 하나의 가중치.
///
/// 합성곱의 필터 `(출력, 입력, k, k)`는 파일에 놓인 그대로 `(출력, 입력 ×
/// k × k)`의 행렬로 본다. 창을 같은 순서(채널, 세로, 가로)로 늘어놓으면
/// 합성곱이 선형 층이 된다(27.5절).
pub(crate) struct Kernel {
    weight: Tensor,
    bias: Option<Vec<f32>>,
    /// 출력의 폭.
    pub(crate) outputs: usize,
    /// 입력 채널의 수.
    pub(crate) inputs: usize,
    /// 창의 한 변. 선형 층이면 1.
    size: usize,
    #[cfg(target_arch = "aarch64")]
    packed: Option<simd::sme::Packed>,
}

impl Kernel {
    /// 파일에서 `이름.weight`와 `이름.bias`(있으면)를 읽는다.
    pub(crate) fn load(file: &SafeTensors, name: &str) -> Result<Self> {
        let weight = file.tensor(&format!("{name}.weight"))?;
        let (outputs, inputs, size) = match *weight.shape() {
            [outputs, inputs] => (outputs, inputs, 1),
            [outputs, inputs, size, same] if size == same => (outputs, inputs, size),
            _ => {
                return Err(invalid(format!(
                    "{name}의 형상이 {:?}입니다",
                    weight.shape()
                )));
            }
        };
        let bias = format!("{name}.bias");
        let bias = if file.names().any(|candidate| candidate == bias) {
            Some(file.tensor(&bias)?.to_vec::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            weight: weight.reshape(&[outputs, inputs * size * size])?,
            bias,
            outputs,
            inputs,
            size,
            #[cfg(target_arch = "aarch64")]
            packed: None,
        })
    }

    /// 창과 보폭이 같은 합성곱의 필터를 선형 층으로 읽는다. 그런 합성곱은
    /// 조각마다 한 번씩 하는 선형 층이다(27.5절, 32.2절).
    pub(crate) fn load_flat(file: &SafeTensors, name: &str) -> Result<Self> {
        let mut kernel = Self::load(file, name)?;
        kernel.inputs *= kernel.size * kernel.size;
        kernel.size = 1;
        Ok(kernel)
    }

    /// SME 커널이 읽을 수 있게 가중치를 다시 늘어놓아 둔다.
    pub(crate) fn pack(&mut self) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        {
            let &[rows, cols] = self.weight.shape() else {
                return Ok(());
            };
            self.packed = simd::sme::Packed::new(self.weight.as_slice::<f32>()?, rows, cols);
        }
        Ok(())
    }

    /// 가중치의 수.
    pub(crate) fn parameters(&self) -> usize {
        self.weight.len() + self.bias.as_ref().map_or(0, Vec::len)
    }

    /// 줄 `count`개에 가중치를 곱한다. 줄 `first`번부터 `n`개의 입력을
    /// `fill`이 적어 준다. 큰 일은 토막으로 나눠, 창들을 한꺼번에 다
    /// 늘어놓지 않는다.
    fn product(
        &self,
        count: usize,
        fill: &(dyn Fn(usize, usize, &mut [f32]) + Sync),
    ) -> Result<Vec<f32>> {
        let _scope = profile::scope(if self.size == 1 { "linear" } else { "conv" });
        let (rows, cols) = (self.outputs, self.inputs * self.size * self.size);
        let weight = self.weight.as_slice::<f32>()?;
        let mut y = vec![0.0; count * rows];
        let pool = threads::global().pool();
        #[cfg(target_arch = "aarch64")]
        if let Some(packed) = &self.packed {
            // SME 커널은 스레드 하나로 돈다. 토막들을 스레드에 나눈다
            // (34.8절). 토막은 커널이 한 번에 받는 16줄의 배수로 한다.
            let even = count.div_ceil(4 * pool.threads()).next_multiple_of(16);
            let strip = even.min((STRIP_VALUES / cols).max(16) / 16 * 16);
            pool.run_chunks(&mut y, strip * rows, pool.threads(), &|index, out| {
                let n = out.len() / rows;
                let mut input = vec![0.0; n * cols];
                fill(index * strip, n, &mut input);
                simd::sme::linear(&input, n, packed, out);
            });
            self.add_bias(&mut y);
            return Ok(y);
        }
        // NEON 커널은 제 안에서 스레드를 나눈다. 토막은 차례로 한다.
        let strip = (STRIP_VALUES / cols).max(16);
        let mut input = vec![0.0; strip.min(count) * cols];
        for (index, out) in y.chunks_mut(strip * rows).enumerate() {
            let n = out.len() / rows;
            fill(index * strip, n, &mut input[..n * cols]);
            simd::linear_threaded(
                pool,
                pool.threads(),
                simd::PIECES_PER_THREAD,
                &input[..n * cols],
                n,
                weight,
                rows,
                cols,
                out,
            );
        }
        self.add_bias(&mut y);
        Ok(y)
    }

    fn add_bias(&self, y: &mut [f32]) {
        if let Some(bias) = &self.bias {
            for row in y.chunks_exact_mut(self.outputs) {
                row.iter_mut().zip(bias).for_each(|(value, b)| *value += b);
            }
        }
    }

    // ANCHOR: linear
    /// 선형 층. `x`는 `(줄 수, 입력 폭)`이다. `1 × 1` 합성곱도 이것이다.
    pub(crate) fn linear(&self, x: &[f32]) -> Result<Vec<f32>> {
        let cols = self.inputs;
        if self.size != 1 || !x.len().is_multiple_of(cols) {
            return Err(invalid(format!(
                "선형 층의 입력이 {}개입니다. 폭은 {cols}입니다",
                x.len()
            )));
        }
        self.product(x.len() / cols, &|first, n, input| {
            input.copy_from_slice(&x[first * cols..][..n * cols]);
        })
    }
    // ANCHOR_END: linear

    // ANCHOR: conv
    /// `3 × 3` 합성곱. 가장자리에 0을 한 줄씩 덧댄다. `stride`가 2이면
    /// 가로와 세로가 절반이 된다.
    pub(crate) fn conv(&self, map: &Map, stride: usize) -> Result<Map> {
        if self.size == 1 && stride == 1 {
            let data = self.linear(&map.data)?;
            return Ok(Map::new(map.height, map.width, self.outputs, data));
        }
        if self.size != 3 || map.channels != self.inputs {
            return Err(invalid(format!(
                "합성곱의 창이 {}이고 입력 채널이 {}입니다 (맵은 {})",
                self.size, self.inputs, map.channels
            )));
        }
        let (channels, height, width) = (map.channels, map.height, map.width);
        let (out_height, out_width) = (height.div_ceil(stride), width.div_ceil(stride));
        let data = self.product(out_height * out_width, &|first, n, input| {
            // 창 하나는 (채널, 세로 3, 가로 3)의 순서다. 필터가 파일에 놓인
            // 순서와 같다.
            input.fill(0.0);
            for (cell, window) in (first..first + n).zip(input.chunks_exact_mut(channels * 9)) {
                let (y, x) = (cell / out_width * stride, cell % out_width * stride);
                for tap in 0..9 {
                    // 덧댄 자리는 0으로 둔다.
                    let (Some(row), Some(column)) =
                        ((y + tap / 3).checked_sub(1), (x + tap % 3).checked_sub(1))
                    else {
                        continue;
                    };
                    if row >= height || column >= width {
                        continue;
                    }
                    let source = &map.data[(row * width + column) * channels..][..channels];
                    for (channel, &value) in source.iter().enumerate() {
                        window[channel * 9 + tap] = value;
                    }
                }
            }
        })?;
        Ok(Map::new(out_height, out_width, self.outputs, data))
    }
    // ANCHOR_END: conv
}
// ANCHOR_END: kernel

// ANCHOR: group_norm
/// 그룹 정규화의 가중치.
pub(crate) struct GroupNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub(crate) fn load(file: &SafeTensors, name: &str, groups: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: file.tensor(&format!("{name}.weight"))?.to_vec::<f32>()?,
            bias: file.tensor(&format!("{name}.bias"))?.to_vec::<f32>()?,
            groups,
            eps,
        })
    }

    /// 채널을 `groups`개의 묶음으로 나눠, 묶음마다 **모든 자리에 걸친**
    /// 평균과 분산으로 정규화한다. `silu`가 참이면 이어서 SiLU를 적용한다
    /// (이 모델들에서 그룹 정규화 뒤에는 거의 늘 SiLU가 온다).
    pub(crate) fn apply(&self, map: &Map, silu: bool, fast: bool) -> Map {
        let _scope = profile::scope("group_norm");
        let channels = map.channels;
        let per_group = channels / self.groups;
        // 1. 채널마다 합과 제곱의 합. 값이 수백만 개일 수 있어 f64로 더한다.
        let (mut sums, mut squares) = (vec![0.0_f64; channels], vec![0.0_f64; channels]);
        for row in map.data.chunks_exact(channels) {
            for ((sum, square), &value) in sums.iter_mut().zip(&mut squares).zip(row) {
                let value = f64::from(value);
                *sum += value;
                *square += value * value;
            }
        }
        // 2. 묶음마다 평균과 분산을 내고, 채널마다 곱할 값과 더할 값으로
        //    바꿔 둔다. (x − 평균) / √(분산 + eps) × w + b = x × a + c.
        let count = (map.cells() * per_group) as f64;
        let (mut scale, mut shift) = (vec![0.0_f32; channels], vec![0.0_f32; channels]);
        for group in 0..self.groups {
            let range = group * per_group..(group + 1) * per_group;
            let mean = sums[range.clone()].iter().sum::<f64>() / count;
            let variance = squares[range.clone()].iter().sum::<f64>() / count - mean * mean;
            let inverse = 1.0 / (variance.max(0.0) + f64::from(self.eps)).sqrt();
            for channel in range {
                let a = inverse * f64::from(self.weight[channel]);
                scale[channel] = a as f32;
                shift[channel] = (f64::from(self.bias[channel]) - mean * a) as f32;
            }
        }
        // 3. 자리마다 적용한다. 줄들은 서로 무관하므로 스레드에 나눈다.
        let mut data = map.data.clone();
        let pool = threads::global().pool();
        let piece = map.cells().div_ceil(4 * pool.threads()).max(1) * channels;
        pool.run_chunks(&mut data, piece, pool.threads(), &|_, rows| {
            for row in rows.chunks_exact_mut(channels) {
                for ((value, a), c) in row.iter_mut().zip(&scale).zip(&shift) {
                    *value = *value * a + c;
                }
            }
            if silu {
                activate(rows, fast);
            }
        });
        Map::new(map.height, map.width, channels, data)
    }
}
// ANCHOR_END: group_norm

/// SiLU를 제자리에서. 빠른 길에서는 지수 함수를 근사한 판을 쓴다(32장).
pub(crate) fn activate(values: &mut [f32], fast: bool) {
    if fast {
        nn::silu_in_place(values);
    } else {
        values
            .iter_mut()
            .for_each(|value| *value = nn::silu(*value));
    }
}

// ANCHOR: attend
/// 어텐션. 질의 `n`줄이 키와 값 `m`줄을 본다. 셋 다 폭이 `width`이고
/// 헤드 `heads`개로 나눈다. 마스크가 없다.
///
/// 점수의 표는 `n × m`이다. 그림의 가장 큰 단계에서 4,096 × 4,096이라,
/// 질의를 토막으로 나눠 토막마다 점수, 소프트맥스, 섞기를 끝낸다.
pub(crate) fn attend(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    width: usize,
    heads: usize,
    fast: bool,
) -> Vec<f32> {
    let _scope = profile::scope("attention");
    let (n, m, head) = (queries.len() / width, keys.len() / width, width / heads);
    let scale = 1.0 / (head as f32).sqrt();
    let mut out = vec![0.0; n * width];
    let pool = threads::global().pool();
    let strip = n.div_ceil(4 * pool.threads()).clamp(1, 256);
    let (mut k, mut v) = (vec![0.0; m * head], vec![0.0; head * m]);
    let mut mixed = vec![0.0; n * head];
    for index in 0..heads {
        // 이 헤드의 키 (m, head)와, 뒤집은 값 (head, m).
        for row in 0..m {
            let from = row * width + index * head;
            k[row * head..][..head].copy_from_slice(&keys[from..][..head]);
            for (lane, &value) in values[from..][..head].iter().enumerate() {
                v[lane * m + row] = value;
            }
        }
        #[cfg(target_arch = "aarch64")]
        let packed = if fast && m >= 32 {
            simd::sme::Packed::new(&k, m, head).zip(simd::sme::Packed::new(&v, head, m))
        } else {
            None
        };
        pool.run_chunks(&mut mixed, strip * head, pool.threads(), &|piece, slot| {
            let count = slot.len() / head;
            let mut q = vec![0.0; count * head];
            for (row, target) in q.chunks_exact_mut(head).enumerate() {
                let from = (piece * strip + row) * width + index * head;
                target.copy_from_slice(&queries[from..][..head]);
            }
            let mut scores = vec![0.0; count * m];
            #[cfg(target_arch = "aarch64")]
            if let Some((packed_k, _)) = &packed {
                simd::sme::linear(&q, count, packed_k, &mut scores);
            } else {
                single(&q, count, &k, m, head, &mut scores);
            }
            #[cfg(not(target_arch = "aarch64"))]
            single(&q, count, &k, m, head, &mut scores);
            for row in scores.chunks_exact_mut(m) {
                row.iter_mut().for_each(|score| *score *= scale);
                if fast {
                    nn::softmax_fast_in_place(row);
                } else {
                    prob::softmax_in_place(row);
                }
            }
            #[cfg(target_arch = "aarch64")]
            if let Some((_, packed_v)) = &packed {
                simd::sme::linear(&scores, count, packed_v, slot);
                return;
            }
            single(&scores, count, &v, head, m, slot);
        });
        for (row, part) in mixed.chunks_exact(head).enumerate() {
            out[row * width + index * head..][..head].copy_from_slice(part);
        }
    }
    out
}

/// 스레드 하나로 하는 `y = x · wᵀ`. 이미 스레드에 나뉜 일 안에서 부른다.
fn single(x: &[f32], count: usize, w: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
    let pool = threads::global().pool();
    simd::linear_threaded(pool, 1, 1, x, count, w, rows, cols, y);
}
// ANCHOR_END: attend

/// 층 정규화를 줄마다 제자리에서. `(줄 수, 폭)`.
pub(crate) fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let width = weight.len();
    for row in x.chunks_exact_mut(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let variance = row
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for ((value, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *value = (*value - mean) * scale * w + b;
        }
    }
}

/// 층 정규화의 가중치와 치우침을 읽는다.
pub(crate) fn pair(file: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        file.tensor(&format!("{name}.weight"))?.to_vec::<f32>()?,
        file.tensor(&format!("{name}.bias"))?.to_vec::<f32>()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_and_rows_round_trip() {
        // (채널 2, 세로 1, 가로 3).
        let planes = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let map = Map::from_planes(1, 3, 2, &planes);
        assert_eq!(map.data, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        assert_eq!(map.to_planes(), planes);
    }

    #[test]
    fn upsampling_repeats_each_cell_four_times() {
        let map = Map::new(1, 2, 1, vec![1.0, 2.0]);
        let big = map.upsample();
        assert_eq!((big.height, big.width), (2, 4));
        assert_eq!(big.data, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn concatenation_joins_channels_per_cell() {
        let a = Map::new(1, 2, 1, vec![1.0, 2.0]);
        let b = Map::new(1, 2, 2, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.concat(&b).data, [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn group_norm_normalizes_each_group_over_all_cells() {
        let norm = GroupNorm {
            weight: vec![1.0, 1.0, 2.0, 2.0],
            bias: vec![0.0, 0.0, 1.0, 1.0],
            groups: 2,
            eps: 0.0,
        };
        // 자리 둘, 채널 넷. 첫 묶음의 값은 1, 3, 5, 7이다(평균 4, 분산 5).
        let map = Map::new(1, 2, 4, vec![1.0, 3.0, 0.0, 0.0, 5.0, 7.0, 2.0, 2.0]);
        let out = norm.apply(&map, false, false);
        let root = 5.0_f32.sqrt();
        let expected = [
            -3.0 / root,
            -1.0 / root,
            -1.0,
            -1.0,
            1.0 / root,
            3.0 / root,
            3.0,
            3.0,
        ];
        for (value, wanted) in out.data.iter().zip(expected) {
            assert!((value - wanted).abs() < 1e-6, "{value} {wanted}");
        }
    }

    #[test]
    fn attention_with_one_key_copies_its_value() {
        // 키가 하나면 점수가 무엇이든 그 값이 그대로 나온다.
        let out = attend(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0], 2, 1, false);
        assert_eq!(out, [7.0, 8.0, 7.0, 8.0]);
    }
}
