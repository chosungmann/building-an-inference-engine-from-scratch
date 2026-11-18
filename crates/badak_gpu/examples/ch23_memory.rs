//! 23장: CPU와 GPU 사이에서 값을 옮기는 값을 잰다.
//!
//! 실행: `cargo run --release -p badak_gpu --example ch23_memory`
//!
//! 이 책의 기계는 CPU와 GPU가 메모리를 함께 쓴다. 그런데도 WebGPU의 규칙은
//! 값을 옮겨 적게 한다. 표준의 규칙만 쓸 때와, 계산에 쓰는 버퍼를 CPU가
//! 곧바로 열 수 있게 하는(표준 밖의) 기능을 쓸 때를 견준다.

use std::time::Instant;

use badak_bench::stats::Summary;
use badak_gpu::Gpu;

fn milliseconds(repetitions: usize, mut work: impl FnMut()) -> f64 {
    work();
    let times: Vec<f64> = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    Summary::new(&times).median
}

fn main() {
    let (Some(direct), Some(standard)) = (Gpu::new(), Gpu::standard()) else {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return;
    };
    println!("GPU: {}", direct.name());
    println!(
        "계산에 쓰는 버퍼를 CPU가 곧바로 열기: 표준 밖의 기능을 켜면 {}, 표준만 쓰면 {}",
        if direct.can_map_storage() {
            "됨"
        } else {
            "안 됨"
        },
        if standard.can_map_storage() {
            "됨"
        } else {
            "안 됨"
        }
    );
    println!();
    println!(
        "{:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>16}",
        "크기",
        "CPU에서 복사",
        "새 버퍼로",
        "적기 (표준)",
        "적기 (곧바로)",
        "읽기 (표준)",
        "읽기 (곧바로)",
        "쓰기도 열면"
    );
    for megabytes in [1, 8, 64, 512] {
        let len = megabytes * (1 << 20) / 4;
        let values: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let mut target = vec![0.0_f32; len];
        let repetitions = if megabytes >= 64 { 7 } else { 30 };
        // 견줄 기준: 같은 양을 CPU의 메모리 안에서 옮겨 적는 시간.
        let copy = milliseconds(repetitions, || {
            target.copy_from_slice(&values);
            std::hint::black_box(&mut target);
        });
        let upload = milliseconds(repetitions, || {
            std::hint::black_box(direct.upload(&values));
            direct.wait();
        });
        let on_standard = standard.upload(&values);
        let on_direct = direct.upload(&values);
        let through_copy = milliseconds(repetitions, || {
            standard.read_into(&on_standard, &mut target)
        });
        let mapped = milliseconds(repetitions, || direct.read_into(&on_direct, &mut target));
        // 이미 있는 버퍼에 적는다. 표준에서는 임시 버퍼를 거치고, 표준 밖의
        // 기능이 있으면 그 메모리에 곧바로 적는다.
        let standard_input = standard.input::<f32>(len);
        let write = milliseconds(repetitions, || {
            standard.write(&standard_input, 0, &values);
            standard.queue().submit([]);
            standard.wait();
        });
        let direct_input = direct.input::<f32>(len);
        let write_mapped = milliseconds(repetitions, || direct.write(&direct_input, 0, &values));
        // 읽기와 쓰기를 모두 열 수 있게 만든 버퍼에서 읽는다(23.7절의 함정).
        let both = direct.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
        direct.write(&both, 0, &values);
        let mapped_both = milliseconds(repetitions, || direct.read_into(&both, &mut target));
        println!(
            "{:>8}MB {:>12.2}ms {:>12.2}ms {:>12.2}ms {:>12.2}ms {:>12.2}ms {:>12.2}ms {:>14.2}ms",
            megabytes, copy, upload, write, write_mapped, through_copy, mapped, mapped_both
        );
    }
}
