//! 브라우저 밖의 GPU: `wgpu`로 연다 (23장).

use std::time::{Duration, Instant};

use crate::executor::block_on;
use crate::{Plain, as_bytes, as_bytes_mut};

/// GPU의 메모리 한 덩이.
pub type Buffer = wgpu::Buffer;
/// GPU에 한 번에 보낼 명령의 묶음.
pub type Encoder = wgpu::CommandEncoder;

// ANCHOR: gpu
/// GPU 하나와, 거기에 일을 보내는 줄.
pub struct Gpu {
    /// 버퍼와 파이프라인을 만드는 곳.
    device: wgpu::Device,
    /// 명령을 GPU로 보내는 줄.
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    features: wgpu::Features,
    /// GPU의 시계를 읽는 데 쓰는 것들. 읽을 수 없는 GPU에서는 없다.
    timer: Option<Timer>,
}

/// GPU가 계산을 시작할 때와 끝낼 때의 시각을 받아 두는 자리.
struct Timer {
    queries: wgpu::QuerySet,
    /// GPU가 시각을 적어 주는 버퍼.
    times: wgpu::Buffer,
    /// 그것을 CPU가 읽으려고 옮겨 적는 버퍼.
    readable: wgpu::Buffer,
}

impl Gpu {
    /// 이 기계에서 가장 성능이 좋은 GPU를 연다. 없으면 `None`.
    pub fn new() -> Option<Self> {
        Self::open(true)
    }

    /// WebGPU의 표준에 있는 기능만 써서 GPU를 연다. 브라우저에서 돌 때와
    /// 같은 조건이다. 표준 밖의 기능이 무엇을 주는지 견줄 때 쓴다.
    pub fn standard() -> Option<Self> {
        Self::open(false)
    }

    fn open(extensions: bool) -> Option<Self> {
        let instance = wgpu::Instance::default();
        // 1. 어댑터: 이 기계에 있는 GPU 가운데 하나.
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        // 화면에 그리는 장치가 아니라 CPU가 GPU를 흉내 내는 것이면 쓰지 않는다.
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return None;
        }

        // 2. 장치: 그 GPU와 맺은 연결. 쓰고 싶은 기능과 한도를 여기서 말한다.
        //    WebGPU의 기본 한도는 버퍼 하나를 256MB로 묶는다. 가중치 행렬이
        //    그보다 크므로 이 GPU가 허락하는 만큼으로 올린다.
        let mut wanted = wgpu::Features::TIMESTAMP_QUERY;
        if extensions {
            // 둘 다 wgpu가 브라우저 밖에서만 내놓는 기능이다.
            wanted |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        let features = adapter.features() & wanted;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("badak"),
            required_features: features,
            required_limits: adapter.limits(),
            // SAFETY: 실험 기능은 23.6절의 행렬 연산 하나만 쓴다. 그 셰이더는
            // 이 크레이트 안에 있고, 결과를 CPU의 계산과 대조하는 테스트가 있다.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .ok()?;
        let timer = features
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Timer {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                times: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readable: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            });
        Some(Self {
            device,
            queue,
            info: adapter.get_info(),
            features,
            timer,
        })
    }
    // ANCHOR_END: gpu

    /// GPU의 이름.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// 어댑터가 알려 준 정보.
    pub fn info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// 이 장치의 한도.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// 계산에 쓰는 버퍼를 CPU가 곧바로 읽을 수 있는가(23.7절).
    pub fn can_map_storage(&self) -> bool {
        self.features
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
    }

    /// 작업 그룹이 함께 하는 행렬 연산(23.6절)을 쓸 수 있는가.
    pub fn can_cooperate(&self) -> bool {
        self.features
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    }

    /// GPU가 일한 시간을 GPU의 시계로 잴 수 있는가.
    pub fn can_time(&self) -> bool {
        self.timer.is_some()
    }

    /// `wgpu`의 장치. 이 크레이트가 감싸지 않은 일을 할 때 쓴다.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// `wgpu`의 줄.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    // ANCHOR: kernel
    /// WGSL로 쓴 컴퓨트 셰이더를 GPU의 기계어로 옮겨 커널로 만든다.
    ///
    /// `entry`는 셰이더에서 `@compute`가 붙은 함수의 이름이다.
    pub fn kernel(&self, label: &str, source: &str, entry: &str) -> Kernel {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                // 버퍼를 어떤 자리에 묶는지는 셰이더의 선언에서 읽어 낸다.
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        Kernel { pipeline }
    }
    // ANCHOR_END: kernel

    // ANCHOR: buffers
    /// 값을 담은 버퍼를 만든다. 셰이더가 읽는 입력과 가중치에 쓴다.
    pub fn upload<T: Plain>(&self, data: &[T]) -> wgpu::Buffer {
        let bytes = as_bytes(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            // 버퍼의 크기는 4의 배수여야 한다.
            size: (bytes.len() as u64).next_multiple_of(4),
            usage: self.storage_usage(),
            // 만들자마자 CPU가 쓸 수 있게 열어 둔다.
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("방금 열어 둔 버퍼다")
            .slice(..bytes.len())
            .copy_from_slice(bytes);
        buffer.unmap();
        // CPU가 열 수 없는 버퍼는 wgpu가 임시 버퍼에 받아 두었다가 다음 명령을
        // 보낼 때 옮겨 적는다. 빈 명령을 보내 지금 옮기게 한다. 그러지 않으면
        // 가중치를 다 올릴 때까지 임시 버퍼가 쌓여 메모리를 두 배로 쓴다.
        self.queue.submit([]);
        self.wait();
        buffer
    }

    /// 0으로 채운 버퍼를 만든다. 셰이더가 결과를 적는 데 쓴다.
    pub fn zeros<T: Plain>(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((len * size_of::<T>()) as u64).next_multiple_of(4),
            usage: self.storage_usage(),
            mapped_at_creation: false,
        })
    }

    /// CPU가 값을 적어 넣을 버퍼를 만든다. 순전파가 선형 층의 입력을 담는 데
    /// 쓴다. 쓸 수 있는 GPU에서는 CPU가 곧바로 열어 적을 수 있게 만든다.
    pub fn input<T: Plain>(&self, len: usize) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if self.can_map_storage() {
            usage |= wgpu::BufferUsages::MAP_WRITE;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((len * size_of::<T>()) as u64).next_multiple_of(4),
            usage,
            mapped_at_creation: false,
        })
    }

    /// 셰이더에 넘기는 작은 값(형상 같은 것)을 담은 버퍼를 만든다.
    pub fn uniform(&self, values: &[u32]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (values.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("방금 열어 둔 버퍼다")
            .copy_from_slice(as_bytes(values));
        buffer.unmap();
        buffer
    }

    fn storage_usage(&self) -> wgpu::BufferUsages {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if self.can_map_storage() {
            // 읽기만 열 수 있게 한다. 쓰기까지 열 수 있게 하면 wgpu가 이 버퍼를
            // "쓰기 전용으로 빠른" 메모리에 두는데, 그런 메모리는 읽기가 서너
            // 배 느리다(23.7절).
            usage | wgpu::BufferUsages::MAP_READ
        } else {
            usage
        }
    }
    // ANCHOR_END: buffers

    // ANCHOR: run
    /// 커널을 실행하라는 명령을 GPU에 보낸다. 끝나기를 기다리지 않는다.
    ///
    /// `buffers`는 셰이더의 `@binding(0)`, `@binding(1)`, ... 에 차례로 묶인다.
    /// `groups`는 작업 그룹을 축마다 몇 개 띄울지다.
    pub fn run(&self, kernel: &Kernel, buffers: &[&wgpu::Buffer], groups: [u32; 3]) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, kernel, buffers, groups);
        self.queue.submit([encoder.finish()]);
    }

    /// 빈 명령 묶음을 만든다.
    pub fn begin(&self) -> Encoder {
        self.device.create_command_encoder(&Default::default())
    }

    /// 명령 묶음을 GPU에 보낸다. 끝나기를 기다리지 않는다.
    pub fn submit(&self, encoder: Encoder) {
        self.queue.submit([encoder.finish()]);
    }

    /// 명령 묶음에 커널 실행 하나를 적는다. 여러 커널을 한 번에 보낼 때 쓴다.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) {
        // 1. 묶음: 셰이더의 자리마다 어느 버퍼를 줄지.
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        // 2. 명령: 어느 커널을, 어느 버퍼로, 몇 그룹 실행할지.
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }

    /// 지금까지 보낸 명령이 모두 끝날 때까지 기다린다.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU를 기다리는 데 실패했습니다");
    }
    // ANCHOR_END: run

    /// 커널을 실행하고 끝날 때까지 기다린다. 걸린 시간을 돌려준다.
    pub fn run_and_wait(
        &self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) -> Duration {
        let start = Instant::now();
        self.run(kernel, buffers, groups);
        self.wait();
        start.elapsed()
    }

    // ANCHOR: run_timed
    /// 커널을 실행하고 끝날 때까지 기다린다. 돌려주는 것은 (부른 쪽에서 잰
    /// 시간, GPU가 스스로 잰 계산 시간)이다. GPU의 시계를 읽을 수 없으면 뒤의
    /// 것은 `None`이다.
    pub fn run_timed(
        &self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) -> (Duration, Option<Duration>) {
        let Some(timer) = &self.timer else {
            return (self.run_and_wait(kernel, buffers, groups), None);
        };
        let (queries, times, readable) = (&timer.queries, &timer.times, &timer.readable);

        let start = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
        let wall = start.elapsed();

        // 시각을 꺼내는 명령은 따로 보낸다. 이 책의 기계에서는 같은 명령 묶음
        // 안에서 꺼내면 지난번 실행의 시각이 나왔다.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(queries, 0..2, times, 0);
        encoder.copy_buffer_to_buffer(times, 0, readable, 0, 16);
        self.queue.submit([encoder.finish()]);

        let mut ticks = [0_u32; 4];
        self.copy_out(readable, 16, &mut ticks);
        let at = |index: usize| u64::from(ticks[index]) | u64::from(ticks[index + 1]) << 32;
        // 스케일 하나가 몇 나노초인지는 GPU마다 다르다.
        let nanoseconds =
            at(2).saturating_sub(at(0)) as f64 * f64::from(self.queue.get_timestamp_period());
        (wall, Some(Duration::from_nanos(nanoseconds as u64)))
    }
    // ANCHOR_END: run_timed

    // ANCHOR: read
    /// 버퍼의 값을 CPU로 읽어 온다. 앞서 보낸 명령이 끝나기를 기다린다.
    pub fn read<T: Plain>(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
        let mut values = vec![T::default(); len];
        self.read_into(buffer, &mut values);
        values
    }

    /// 버퍼의 값을 `values`에 읽어 온다.
    pub fn read_into<T: Plain>(&self, buffer: &wgpu::Buffer, values: &mut [T]) {
        let bytes = size_of_val(values) as u64;
        if self.can_map_storage() {
            // 계산에 쓴 버퍼를 곧바로 연다. CPU와 GPU가 메모리를 함께 쓰는
            // 기계에서는 옮겨 적는 일 없이 그 메모리를 읽는다.
            self.copy_out(buffer, bytes, values);
        } else {
            // WebGPU의 기본 규칙에서는 계산에 쓰는 버퍼를 CPU가 열 수 없다.
            // 읽기 전용 버퍼로 한 번 옮겨 적고 그것을 연다.
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes.next_multiple_of(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes.next_multiple_of(4));
            self.queue.submit([encoder.finish()]);
            self.copy_out(&staging, bytes, values);
        }
    }

    fn copy_out<T: Plain>(&self, buffer: &wgpu::Buffer, bytes: u64, values: &mut [T]) {
        let slice = buffer.slice(..bytes.next_multiple_of(4));
        // 열어 달라고 부탁하고, GPU가 앞선 일을 끝내고 열어 줄 때까지 기다린다.
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("버퍼를 여는 데 실패했습니다");
        });
        self.wait();
        let view = slice.get_mapped_range().expect("열릴 때까지 기다렸다");
        as_bytes_mut(values).copy_from_slice(&view[..bytes as usize]);
        drop(view);
        buffer.unmap();
    }
    // ANCHOR_END: read

    // ANCHOR: write
    /// CPU의 값을 이미 있는 버퍼의 `offset`번째 값부터 적는다.
    pub fn write<T: Plain>(&self, buffer: &wgpu::Buffer, offset: usize, data: &[T]) {
        let bytes = as_bytes(data);
        let start = (offset * size_of::<T>()) as u64;
        if buffer.usage().contains(wgpu::BufferUsages::MAP_WRITE) && bytes.len().is_multiple_of(4) {
            // CPU가 열 수 있는 버퍼다. 그 메모리에 곧바로 적는다.
            let slice = buffer.slice(start..start + bytes.len() as u64);
            slice.map_async(wgpu::MapMode::Write, |result| {
                result.expect("버퍼를 여는 데 실패했습니다");
            });
            self.wait();
            slice
                .get_mapped_range_mut()
                .expect("열릴 때까지 기다렸다")
                .copy_from_slice(bytes);
            buffer.unmap();
        } else {
            // 표준의 길: wgpu가 값을 임시 버퍼에 옮겨 적어 두었다가, 다음 명령을
            // 보낼 때 GPU가 그것을 제자리로 옮겨 적는다.
            self.queue.write_buffer(buffer, start, bytes);
        }
    }
    // ANCHOR_END: write
}

/// GPU에서 실행할 수 있게 준비된 컴퓨트 셰이더.
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
}
