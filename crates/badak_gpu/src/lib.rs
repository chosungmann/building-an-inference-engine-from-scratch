//! badak의 GPU 백엔드 (23장).
//!
//! 엔진의 다른 부분은 Rust 표준 라이브러리만 쓴다. 이 크레이트만 예외로
//! `wgpu`를 쓴다. GPU에 일을 시키려면 운영체제와 드라이버를 거쳐야 하고,
//! 그 길은 운영체제마다 다르다. `wgpu`는 WebGPU라는 하나의 약속으로 그
//! 길들을 감싼다. 같은 코드가 Mac(Metal), Linux(Vulkan), Windows(Direct3D)에서
//! 돈다.
//!
//! 브라우저에서는 `wgpu`를 쓰지 않는다(49장). 브라우저에는 WebGPU가 이미
//! 들어 있으므로, 엔진은 그것을 부르는 함수들을 JavaScript에서 들여온다.
//! [`Gpu`]의 구현이 둘이고([`native`]와 [`web`]), 그 위의 커널과
//! [`offload`]는 하나다. 셰이더도 같은 파일이다.
//!
//! GPU가 없는 기계에서는 [`Gpu::new`]가 `None`을 돌려준다.

#[cfg(not(target_arch = "wasm32"))]
mod executor;
pub mod linear;
#[cfg(not(target_arch = "wasm32"))]
mod native;
pub mod offload;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Buffer, Encoder, Gpu, Kernel};
#[cfg(target_arch = "wasm32")]
pub use web::{Buffer, Encoder, Gpu, Kernel};

/// 이 크레이트에 든 셰이더의 소스.
pub mod shaders {
    /// 원소마다 SiLU를 적용한다. 버퍼는 `x`, `y` 순서다.
    pub const SILU: &str = include_str!("shaders/silu.wgsl");
    /// MLP의 가운데 `silu(gate) × up`. 버퍼는 형상, `gate`, `up` 순서다.
    pub const SILU_MUL: &str = include_str!("shaders/silu_mul.wgsl");
}

/// 원소 `len`개를 스레드 256개짜리 그룹으로 덮을 때 그룹을 어떻게 놓을지.
///
/// 한 축에 놓을 수 있는 그룹의 수에 한도(65,535)가 있어서, 한 줄에 32,768개씩
/// 여러 줄로 놓는다. 셰이더는 `id.y * 32768 * 256 + id.x`로 원소의 번호를
/// 되찾는다.
pub fn groups_for(len: usize) -> [u32; 3] {
    const PER_ROW: usize = 32768;
    let groups = len.div_ceil(256).max(1);
    [
        groups.min(PER_ROW) as u32,
        groups.div_ceil(PER_ROW) as u32,
        1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_on_the_gpu_agrees_with_the_cpu() {
        // GPU가 없는 기계(CI의 리눅스)에서는 건너뛴다.
        let Some(gpu) = Gpu::new() else {
            eprintln!("GPU가 없어 건너뜁니다");
            return;
        };
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 50.0).collect();
        let (input, output) = (gpu.upload(&x), gpu.zeros::<f32>(x.len()));
        let kernel = gpu.kernel("silu", shaders::SILU, "main");
        gpu.run(&kernel, &[&input, &output], groups_for(x.len()));
        let y: Vec<f32> = gpu.read(&output, x.len());
        for (&x, &y) in x.iter().zip(&y) {
            let expected = badak_core::nn::silu(x);
            assert!(
                (y - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "silu({x}) = {y}, CPU에서는 {expected}"
            );
        }
    }

    #[test]
    fn groups_cover_every_element_within_the_per_axis_limit() {
        assert_eq!(groups_for(1), [1, 1, 1]);
        assert_eq!(groups_for(256), [1, 1, 1]);
        assert_eq!(groups_for(257), [2, 1, 1]);
        assert_eq!(groups_for(32768 * 256), [32768, 1, 1]);
        assert_eq!(groups_for(32768 * 256 + 1), [32768, 2, 1]);
    }

    #[test]
    fn a_kernel_reaches_elements_beyond_one_row_of_groups() {
        let Some(gpu) = Gpu::new() else {
            return;
        };
        // 한 줄(32,768그룹)을 넘는 크기. 마지막 원소까지 계산되는지 본다.
        let len = 32768 * 256 + 1000;
        let x = vec![1.0_f32; len];
        let (input, output) = (gpu.upload(&x), gpu.zeros::<f32>(len));
        let kernel = gpu.kernel("silu", shaders::SILU, "main");
        gpu.run(&kernel, &[&input, &output], groups_for(len));
        let y: Vec<f32> = gpu.read(&output, len);
        let expected = badak_core::nn::silu(1.0);
        assert!((y[0] - expected).abs() < 1e-6);
        assert!((y[len - 1] - expected).abs() < 1e-6);
    }

    #[test]
    fn plain_values_round_trip_through_a_buffer() {
        let Some(gpu) = Gpu::new() else {
            return;
        };
        let values: Vec<u16> = (0..1001).map(|i| (i * 7) as u16).collect();
        let buffer = gpu.upload(&values);
        assert_eq!(gpu.read::<u16>(&buffer, values.len()), values);
        let floats = [1.5_f32, -2.25, 0.0];
        let buffer = gpu.zeros::<f32>(3);
        gpu.write(&buffer, 0, &floats);
        assert_eq!(gpu.read::<f32>(&buffer, 3), floats);
    }
}

mod plain {
    /// [`Plain`](super::Plain)을 이 크레이트 안에서만 구현하게 막는다.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u16 {}
    impl Sealed for u8 {}
}

/// 바이트의 나열로 그대로 읽고 쓸 수 있는 값. 어떤 비트 무늬든 유효하고
/// 빈틈(패딩)이 없는 타입에만 구현되어 있다.
pub trait Plain: Copy + Default + plain::Sealed + 'static {}
impl Plain for f32 {}
impl Plain for u32 {}
impl Plain for i32 {}
impl Plain for u16 {}
impl Plain for u8 {}

/// 값들을 바이트로 읽는다. 복사하지 않는다.
pub fn as_bytes<T: Plain>(data: &[T]) -> &[u8] {
    // SAFETY: `Plain`은 빈틈이 없는 수 타입에만 구현되어 있으므로 그 메모리를
    // 바이트로 읽는 것은 언제나 유효하다. `u8`은 정렬을 요구하지 않는다.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

pub(crate) fn as_bytes_mut<T: Plain>(data: &mut [T]) -> &mut [u8] {
    // SAFETY: 위와 같고, `Plain`은 어떤 비트 무늬든 유효한 값이므로 바이트를
    // 무엇으로 덮어써도 된다.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), size_of_val(data)) }
}
