//! 브라우저의 GPU: JavaScript에서 들여온 함수로 연다 (49장).
//!
//! 브라우저에는 WebGPU가 들어 있다. 이 모듈은 그것을 직접 부르지 못한다.
//! WebAssembly 모듈이 할 수 있는 일은 들여온 함수를 부르는 것뿐이다(48장).
//! 그래서 GPU에 시킬 일을 열두 가지 함수로 정해 두고, JavaScript가 그것들을
//! WebGPU로 구현해 건넨다(`crates/badak_web/js/gpu.js`).
//!
//! 버퍼와 커널은 JavaScript 쪽에 있고, 여기에서는 번호로만 안다.
//!
//! [`native`](../native/index.html)의 [`Gpu`]와 같은 이름의 함수들을 갖는다.
//! 다른 점은 둘이다. 표준 밖의 기능이 없고, 기다리는 방법이 다르다. 기다려야
//! 하는 함수(`wait`, `read`)는 JavaScript에서 "나중에 끝나는" 함수인데, 이
//! 코드는 그것을 그냥 부르고 돌아오면 끝난 것으로 안다. 그 사이에 모듈을
//! 멈춰 두었다가 다시 깨우는 일은 브라우저가 한다(JSPI).

use std::time::{Duration, Instant};

use crate::{Plain, as_bytes, as_bytes_mut};

// ANCHOR: imports
#[link(wasm_import_module = "gpu")]
unsafe extern "C" {
    /// 쓸 수 있는 GPU가 있으면 1.
    #[link_name = "open"]
    fn gpu_open() -> i32;
    /// GPU의 이름을 `pointer`에 적고 전체 길이를 돌려준다.
    #[link_name = "name"]
    fn gpu_name(pointer: *mut u8, capacity: usize) -> usize;
    /// GPU의 시계를 읽을 수 있으면 1.
    #[link_name = "can_time"]
    fn gpu_can_time() -> i32;
    /// WGSL 소스로 커널을 만들고 번호를 돌려준다.
    #[link_name = "kernel"]
    fn gpu_kernel(source: *const u8, source_len: usize, entry: *const u8, entry_len: usize) -> u32;
    /// `bytes`바이트짜리 버퍼를 만들고 번호를 돌려준다.
    #[link_name = "buffer"]
    fn gpu_buffer(bytes: f64, uniform: i32) -> u32;
    /// 모듈의 메모리에 있는 값을 버퍼에 적는다.
    #[link_name = "write"]
    fn gpu_write(buffer: u32, offset: f64, pointer: *const u8, len: usize);
    /// 명령 묶음에 커널 실행 하나를 적는다.
    #[link_name = "encode"]
    fn gpu_encode(
        kernel: u32,
        buffers: *const u32,
        count: usize,
        x: u32,
        y: u32,
        z: u32,
        timed: i32,
    );
    /// 명령 묶음을 GPU에 보낸다.
    #[link_name = "submit"]
    fn gpu_submit();
    /// 보낸 명령이 모두 끝날 때까지 기다린다. (모듈이 멈췄다가 깨어난다.)
    #[link_name = "wait"]
    fn gpu_wait();
    /// 버퍼의 값을 모듈의 메모리로 읽어 온다. (모듈이 멈췄다가 깨어난다.)
    #[link_name = "read"]
    fn gpu_read(buffer: u32, pointer: *mut u8, len: usize);
    /// 시계를 달고 실행한 마지막 커널이 걸린 시간(나노초). (멈췄다가 깨어난다.)
    #[link_name = "elapsed"]
    fn gpu_elapsed() -> f64;
    /// 버퍼를 지운다.
    #[link_name = "release"]
    fn gpu_release(buffer: u32);
}
// ANCHOR_END: imports

/// 브라우저가 내준 GPU.
pub struct Gpu {
    name: String,
    timer: bool,
}

/// GPU의 메모리 한 덩이. JavaScript 쪽에 있는 버퍼의 번호다.
#[derive(Debug)]
pub struct Buffer {
    id: u32,
    size: u64,
}

impl Buffer {
    /// 버퍼의 크기(바이트).
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: 이 번호는 `gpu_buffer`가 준 것이고 한 번만 지운다.
        unsafe { gpu_release(self.id) };
    }
}

/// GPU에서 실행할 수 있게 준비된 컴퓨트 셰이더.
pub struct Kernel {
    id: u32,
}

/// GPU에 한 번에 보낼 명령의 묶음. 묶음 자체는 JavaScript 쪽에 있다.
pub struct Encoder(());

impl Gpu {
    /// 브라우저가 내준 GPU를 연다. 없으면 `None`.
    ///
    /// GPU를 고르고 여는 일은 JavaScript가 모듈을 올리기 전에 끝내 둔다.
    /// 그 일은 "나중에 끝나는" 일이어서 모듈 안에서 기다릴 수 없다.
    pub fn new() -> Option<Self> {
        // SAFETY: 들여온 함수들은 인자 없이 값을 돌려줄 뿐이다.
        if unsafe { gpu_open() } == 0 {
            return None;
        }
        let mut name = vec![0_u8; 256];
        // SAFETY: `name`은 256바이트를 쓸 수 있는 자리다.
        let len = unsafe { gpu_name(name.as_mut_ptr(), name.len()) }.min(name.len());
        name.truncate(len);
        Some(Self {
            name: String::from_utf8_lossy(&name).into_owned(),
            // SAFETY: 위와 같다.
            timer: unsafe { gpu_can_time() } != 0,
        })
    }

    /// 브라우저에는 표준의 기능뿐이다. [`new`](Self::new)와 같다.
    pub fn standard() -> Option<Self> {
        Self::new()
    }

    /// GPU의 이름.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 계산에 쓰는 버퍼를 CPU가 곧바로 열 수 있는가. 브라우저에서는 없다.
    pub fn can_map_storage(&self) -> bool {
        false
    }

    /// 행렬 연산을 쓸 수 있는가. 브라우저에서는 없다.
    pub fn can_cooperate(&self) -> bool {
        false
    }

    /// GPU의 시계를 읽을 수 있는가.
    pub fn can_time(&self) -> bool {
        self.timer
    }

    /// WGSL로 쓴 컴퓨트 셰이더를 커널로 만든다.
    pub fn kernel(&self, _label: &str, source: &str, entry: &str) -> Kernel {
        // SAFETY: 두 글은 이 호출 동안 살아 있다.
        let id = unsafe { gpu_kernel(source.as_ptr(), source.len(), entry.as_ptr(), entry.len()) };
        Kernel { id }
    }

    fn create(&self, bytes: usize, uniform: bool) -> Buffer {
        // 버퍼의 크기는 4의 배수여야 한다.
        let size = (bytes as u64).next_multiple_of(4);
        // SAFETY: 수 둘을 넘길 뿐이다.
        let id = unsafe { gpu_buffer(size as f64, i32::from(uniform)) };
        Buffer { id, size }
    }

    /// 값을 담은 버퍼를 만든다.
    pub fn upload<T: Plain>(&self, data: &[T]) -> Buffer {
        let buffer = self.create(size_of_val(data), false);
        self.write(&buffer, 0, data);
        buffer
    }

    /// 0으로 채운 버퍼를 만든다.
    pub fn zeros<T: Plain>(&self, len: usize) -> Buffer {
        self.create(len * size_of::<T>(), false)
    }

    /// CPU가 값을 적어 넣을 버퍼를 만든다.
    pub fn input<T: Plain>(&self, len: usize) -> Buffer {
        self.zeros::<T>(len)
    }

    /// 셰이더에 넘기는 작은 값을 담은 버퍼를 만든다.
    pub fn uniform(&self, values: &[u32]) -> Buffer {
        let buffer = self.create(size_of_val(values), true);
        self.write(&buffer, 0, values);
        buffer
    }

    /// 커널을 실행하라는 명령을 GPU에 보낸다. 끝나기를 기다리지 않는다.
    pub fn run(&self, kernel: &Kernel, buffers: &[&Buffer], groups: [u32; 3]) {
        let mut encoder = self.begin();
        self.encode(&mut encoder, kernel, buffers, groups);
        self.submit(encoder);
    }

    /// 빈 명령 묶음을 만든다.
    pub fn begin(&self) -> Encoder {
        Encoder(())
    }

    /// 명령 묶음에 커널 실행 하나를 적는다.
    pub fn encode(
        &self,
        _encoder: &mut Encoder,
        kernel: &Kernel,
        buffers: &[&Buffer],
        groups: [u32; 3],
    ) {
        self.encode_with(kernel, buffers, groups, false);
    }

    fn encode_with(&self, kernel: &Kernel, buffers: &[&Buffer], groups: [u32; 3], timed: bool) {
        let ids: Vec<u32> = buffers.iter().map(|buffer| buffer.id).collect();
        // SAFETY: `ids`는 이 호출 동안 살아 있는 번호 `ids.len()`개다.
        unsafe {
            gpu_encode(
                kernel.id,
                ids.as_ptr(),
                ids.len(),
                groups[0],
                groups[1],
                groups[2],
                i32::from(timed),
            );
        }
    }

    /// 명령 묶음을 GPU에 보낸다. 끝나기를 기다리지 않는다.
    pub fn submit(&self, _encoder: Encoder) {
        // SAFETY: 인자가 없다.
        unsafe { gpu_submit() };
    }

    // ANCHOR: wait
    /// 지금까지 보낸 명령이 모두 끝날 때까지 기다린다.
    pub fn wait(&self) {
        // SAFETY: 인자가 없다. 이 함수가 돌아오는 것은 GPU가 일을 마친
        // 뒤다. 그 사이에 이 모듈은 멈춰 있다.
        unsafe { gpu_wait() };
    }

    /// 버퍼의 값을 `values`에 읽어 온다.
    pub fn read_into<T: Plain>(&self, buffer: &Buffer, values: &mut [T]) {
        let bytes = as_bytes_mut(values);
        // SAFETY: `bytes`는 이 호출 동안 살아 있고 쓸 수 있는 자리다.
        unsafe { gpu_read(buffer.id, bytes.as_mut_ptr(), bytes.len()) };
    }
    // ANCHOR_END: wait

    /// 버퍼의 값을 CPU로 읽어 온다.
    pub fn read<T: Plain>(&self, buffer: &Buffer, len: usize) -> Vec<T> {
        let mut values = vec![T::default(); len];
        self.read_into(buffer, &mut values);
        values
    }

    /// CPU의 값을 이미 있는 버퍼의 `offset`번째 값부터 적는다.
    pub fn write<T: Plain>(&self, buffer: &Buffer, offset: usize, data: &[T]) {
        let bytes = as_bytes(data);
        let start = (offset * size_of::<T>()) as f64;
        if bytes.len().is_multiple_of(4) {
            // SAFETY: `bytes`는 이 호출 동안 살아 있다.
            unsafe { gpu_write(buffer.id, start, bytes.as_ptr(), bytes.len()) };
        } else {
            // WebGPU는 4바이트 단위로만 적는다. 모자란 만큼 0을 붙인다.
            let mut padded = bytes.to_vec();
            padded.resize(bytes.len().next_multiple_of(4), 0);
            // SAFETY: 위와 같다.
            unsafe { gpu_write(buffer.id, start, padded.as_ptr(), padded.len()) };
        }
    }

    /// 커널을 실행하고 끝날 때까지 기다린다. 걸린 시간을 돌려준다.
    pub fn run_and_wait(&self, kernel: &Kernel, buffers: &[&Buffer], groups: [u32; 3]) -> Duration {
        let start = Instant::now();
        self.run(kernel, buffers, groups);
        self.wait();
        start.elapsed()
    }

    /// 커널을 실행하고 끝날 때까지 기다린다. 돌려주는 것은 (부른 쪽에서 잰
    /// 시간, GPU가 스스로 잰 계산 시간)이다.
    pub fn run_timed(
        &self,
        kernel: &Kernel,
        buffers: &[&Buffer],
        groups: [u32; 3],
    ) -> (Duration, Option<Duration>) {
        if !self.timer {
            return (self.run_and_wait(kernel, buffers, groups), None);
        }
        let start = Instant::now();
        self.encode_with(kernel, buffers, groups, true);
        // SAFETY: 인자가 없다.
        unsafe { gpu_submit() };
        self.wait();
        let wall = start.elapsed();
        // SAFETY: 인자가 없다.
        let nanoseconds = unsafe { gpu_elapsed() };
        (
            wall,
            Some(Duration::from_nanos(nanoseconds.max(0.0) as u64)),
        )
    }
}
