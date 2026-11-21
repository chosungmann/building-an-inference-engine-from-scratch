//! 선형 층의 커널 (23.5절).
//!
//! 같은 계산 `y = x Wᵀ`를 하는 셰이더가 여럿 있다. 단순한 것부터 빠른
//! 것까지 모두 남겨 두고 서로 견준다.

use crate::{Buffer, Gpu, Kernel};

// ANCHOR: variant
/// 선형 층을 계산하는 셰이더의 종류.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// 스레드 하나가 값 하나를, 곱셈 하나씩.
    Simple,
    /// 스레드 하나가 값 하나를, 넷씩 묶어 네 줄기로.
    Wide,
    /// 스레드 하나가 값 하나를, 17장의 CPU 커널과 같은 순서로.
    Exact,
    /// 스레드 하나가 값 열여섯(벡터 넷 × 행 넷)을.
    Block,
    /// 작업 그룹 하나가 8 × 8 행렬의 곱을 함께. 표준 밖의 기능이다.
    Cooperative,
}
// ANCHOR_END: variant

impl Variant {
    /// 모든 종류.
    pub const ALL: [Self; 5] = [
        Self::Simple,
        Self::Wide,
        Self::Exact,
        Self::Block,
        Self::Cooperative,
    ];

    /// 셰이더의 소스.
    pub fn source(self) -> &'static str {
        match self {
            Self::Simple => include_str!("shaders/linear_simple.wgsl"),
            Self::Wide => include_str!("shaders/linear_wide.wgsl"),
            Self::Exact => include_str!("shaders/linear_exact.wgsl"),
            Self::Block => include_str!("shaders/linear_block.wgsl"),
            Self::Cooperative => include_str!("shaders/linear_cooperative.wgsl"),
        }
    }

    /// 표에 적을 이름.
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "단순",
            Self::Wide => "넷씩 네 줄기",
            Self::Exact => "CPU와 같은 순서",
            Self::Block => "값 열여섯",
            Self::Cooperative => "행렬 연산",
        }
    }

    /// 계산의 단위 하나가 맡는 (행의 수, 벡터의 수)와, 그룹 하나에 든 단위의
    /// (가로, 세로) 수.
    fn tile(self) -> ((usize, usize), (usize, usize)) {
        match self {
            Self::Simple | Self::Wide | Self::Exact => ((1, 1), (8, 8)),
            Self::Block => ((4, 4), (8, 8)),
            // 그룹 하나가 통째로 단위 하나다.
            Self::Cooperative => ((32, 32), (1, 1)),
        }
    }

    /// 열의 수가 이 수의 배수여야 한다.
    fn step(self) -> usize {
        match self {
            Self::Simple => 1,
            Self::Wide => 16,
            Self::Exact | Self::Cooperative => 8,
            Self::Block => 4,
        }
    }

    /// 이 셰이더가 이 형상을 계산할 수 있는가.
    pub fn supports(self, count: usize, rows: usize, cols: usize) -> bool {
        let ((tile_rows, tile_vectors), _) = self.tile();
        rows.is_multiple_of(tile_rows)
            && count.is_multiple_of(tile_vectors)
            && cols.is_multiple_of(self.step())
    }

    /// 이 GPU에서 쓸 수 있는가.
    pub fn available(self, gpu: &Gpu) -> bool {
        self != Self::Cooperative || gpu.can_cooperate()
    }
}

// ANCHOR: linear
/// 선형 층 커널 하나.
pub struct Linear {
    kernel: Kernel,
    variant: Variant,
}

impl Linear {
    /// 셰이더를 옮겨 커널을 만든다.
    pub fn new(gpu: &Gpu, variant: Variant) -> Self {
        Self {
            kernel: gpu.kernel("linear", variant.source(), "main"),
            variant,
        }
    }

    /// `y = x Wᵀ`를 계산하라는 명령을 보낸다. 끝나기를 기다리지 않는다.
    ///
    /// `x`는 `(count, cols)`, `w`는 `(rows, cols)`, `y`는 `(count, rows)`다.
    pub fn run(
        &self,
        gpu: &Gpu,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        (count, rows, cols): (usize, usize, usize),
    ) {
        assert!(
            self.variant.supports(count, rows, cols),
            "{:?} 커널은 형상 ({count}, {rows}, {cols})를 계산하지 못합니다",
            self.variant
        );
        let shape = gpu.uniform(&[count as u32, rows as u32, cols as u32, 0]);
        gpu.run(&self.kernel, &[&shape, x, w, y], self.groups(count, rows));
    }

    /// 그룹은 가로로 행을, 세로로 벡터를 덮는다.
    pub fn groups(&self, count: usize, rows: usize) -> [u32; 3] {
        let ((tile_rows, tile_vectors), (across, down)) = self.variant.tile();
        [
            rows.div_ceil(tile_rows * across) as u32,
            count.div_ceil(tile_vectors * down) as u32,
            1,
        ]
    }

    /// 안에 든 커널.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
}
// ANCHOR_END: linear

#[cfg(test)]
mod tests {
    use super::*;
    use badak_core::random::Pcg64;
    use badak_core::{Tensor, ops};

    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64::new(seed);
        (0..len)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1 << 23) as f32 - 1.0)
            .collect()
    }

    #[test]
    fn every_variant_agrees_with_the_cpu() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("GPU가 없어 건너뜁니다");
            return;
        };
        for (count, rows, cols) in [
            (4, 8, 32),
            (12, 20, 64),
            (1, 4, 16),
            (8, 64, 2048),
            (32, 64, 128),
            (64, 32, 2048),
        ] {
            let (x, w) = (values(count * cols, 1), values(rows * cols, 2));
            let expected = ops::linear(
                &Tensor::from_vec(x.clone(), &[count, cols]).unwrap(),
                &Tensor::from_vec(w.clone(), &[rows, cols]).unwrap(),
                None,
            )
            .unwrap();
            let expected = expected.as_slice::<f32>().unwrap();
            let (x_buffer, w_buffer) = (gpu.upload(&x), gpu.upload(&w));
            for variant in Variant::ALL {
                if !variant.supports(count, rows, cols) || !variant.available(&gpu) {
                    continue;
                }
                let y_buffer = gpu.zeros::<f32>(count * rows);
                Linear::new(&gpu, variant).run(
                    &gpu,
                    &x_buffer,
                    &w_buffer,
                    &y_buffer,
                    (count, rows, cols),
                );
                let y: Vec<f32> = gpu.read(&y_buffer, count * rows);
                for (index, (&got, &want)) in y.iter().zip(expected).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "{variant:?} ({count}, {rows}, {cols})의 {index}번째: {got} 대 {want}"
                    );
                }
            }
        }
    }
}
