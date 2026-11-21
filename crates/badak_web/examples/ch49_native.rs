//! 49장: 브라우저에서 시킨 GPU의 일을 맥에서 그대로 시켜, 견줄 숫자를 적는다.
//!
//! 실행: `cargo run --release -p badak_web --example ch49_native -- [모델 ...]`
//!
//! 부르는 것은 브라우저의 측정 페이지(`web/bench_gpu.html`)가 부르는 것과
//! 같은 문의 함수들이고, 적는 줄도 같은 꼴이다. 다른 것은 GPU에 닿는
//! 길뿐이다. 여기에서는 `wgpu`가 Metal을 부른다.
//!
//! 표준 WebGPU의 기능만 쓴다. `BADAK_GPU=full`을 주면 23장의 표준 밖 기능
//! 둘도 쓴다.

use std::ffi::CString;
use std::time::Instant;

use badak_graph::models_dir;
use badak_web::gpu::*;
use badak_web::*;

const VARIANTS: [&str; 4] = ["simple", "wide", "exact", "block"];
const SHAPES: [(usize, usize, usize); 3] = [(64, 1536, 576), (256, 2048, 2048), (256, 8192, 2048)];

fn microseconds(seconds: f64) -> String {
    if seconds < 0.0 {
        "none".to_string()
    } else {
        format!("{:.0}", seconds * 1e6)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if badak_web_gpu_open() == 0 {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return Ok(());
    }
    let mut name = [0_u8; 256];
    // SAFETY: `name`은 256바이트를 쓸 수 있는 자리다.
    let len = unsafe { badak_web_gpu_name(name.as_mut_ptr(), name.len()) }.min(name.len());
    println!("gpu-name {}", String::from_utf8_lossy(&name[..len]));

    let linear =
        |variant: usize, (count, rows, cols): (usize, usize, usize), repeats: usize, warm: f64| {
            let (mut out, mut digest) = ([0.0_f64; 4], 0_u64);
            // SAFETY: `out`은 `f64` 넷, `digest`는 `u64` 하나를 쓸 수 있는 자리다.
            let status = unsafe {
                badak_web_gpu_linear(
                    variant as i32,
                    count,
                    rows,
                    cols,
                    repeats,
                    warm,
                    out.as_mut_ptr(),
                    &mut digest,
                )
            };
            assert_eq!(status, 0);
            (out, digest)
        };
    let (tiny, _) = linear(3, (4, 8, 32), 20, 0.0);
    println!(
        "gpu-round-trip wall-us {} gpu-us {} read-us {}",
        microseconds(tiny[0]),
        microseconds(tiny[1]),
        microseconds(tiny[2])
    );
    for shape in SHAPES {
        let (count, rows, cols) = shape;
        std::thread::sleep(std::time::Duration::from_secs(3));
        let mut digest = 0_u64;
        // SAFETY: `digest`는 `u64` 하나를 쓸 수 있는 자리다.
        let seconds = unsafe { badak_web_bench_linear(rows, cols, count, 10, &mut digest) };
        println!(
            "cpu-linear {count}x{rows}x{cols} us {} digest {digest:016x}",
            microseconds(seconds)
        );
        // GPU는 한가하면 느리게 돈다. 쉬었다가 바로 잰 것(식은 GPU)을 하나
        // 적고, 나머지는 0.5초 동안 데운 뒤에 잰다.
        std::thread::sleep(std::time::Duration::from_secs(3));
        let (cold, _) = linear(3, shape, 20, 0.0);
        println!(
            "gpu-linear-cold block {count}x{rows}x{cols} wall-us {} gpu-us {} read-us {}",
            microseconds(cold[0]),
            microseconds(cold[1]),
            microseconds(cold[2])
        );
        for (variant, label) in VARIANTS.iter().enumerate() {
            std::thread::sleep(std::time::Duration::from_secs(3));
            let (out, digest) = linear(variant, shape, 20, 0.5);
            println!(
                "gpu-linear {label} {count}x{rows}x{cols} wall-us {} gpu-us {} read-us {} error {:.1e} digest {digest:016x}",
                microseconds(out[0]),
                microseconds(out[1]),
                microseconds(out[2]),
                out[3]
            );
        }
    }

    for model in std::env::args().skip(1) {
        let directory = CString::new(models_dir().join(&model).to_str().ok_or("길")?)?;
        for (label, mode, min_count) in [
            ("cpu", 0, 1),
            ("gpu-layer", 1, 1),
            ("gpu-mlp", 2, 1),
            ("gpu-mlp-32", 2, 32),
        ] {
            std::thread::sleep(std::time::Duration::from_secs(3));
            // SAFETY: 문의 약속대로 부른다.
            unsafe {
                let started = Instant::now();
                let llm =
                    badak_web_llm_open(directory.as_ptr().cast(), directory.as_bytes().len(), 0);
                assert!(!llm.is_null());
                let opened = started.elapsed().as_secs_f64() * 1e3;
                let started = Instant::now();
                let megabytes = if mode > 0 {
                    badak_web_llm_offload(llm, mode, min_count)
                } else {
                    0.0
                };
                assert!(megabytes >= 0.0);
                let uploaded = started.elapsed().as_secs_f64() * 1e3;
                let mut speed = [0.0_f64; 2];
                assert_eq!(badak_web_llm_speed(llm, 256, 32, 4, speed.as_mut_ptr()), 0);
                let mut s = [0.0_f64; 5];
                if mode > 0 {
                    badak_web_gpu_statistics(s.as_mut_ptr());
                }
                println!(
                    "llm-speed {model} {label} prompt {:.0} generation {:.1} open-ms {opened:.0} upload-mb {megabytes:.0} upload-ms {uploaded:.0} gpu-calls {} cpu-calls {} write-ms {:.0} compute-ms {:.0} read-ms {:.0}",
                    speed[0],
                    speed[1],
                    s[3],
                    s[4],
                    s[0] * 1e3,
                    s[1] * 1e3,
                    s[2] * 1e3
                );
                badak_web_llm_close(llm);
            }
        }
    }
    Ok(())
}
