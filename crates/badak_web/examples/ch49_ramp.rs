//! 49장: GPU는 일이 이어져야 빨라진다. 같은 커널을 몇 번 되풀이해 재느냐에
//! 따라 "가장 빨랐던 한 번"이 어떻게 달라지는지 본다.
//!
//! 실행: `cargo run --release -p badak_web --example ch49_ramp`
//!
//! 되풀이마다 앞에서 3초를 쉰다. 데우기는 하지 않는다.

use badak_web::gpu::*;

fn main() {
    if badak_web_gpu_open() == 0 {
        println!("이 기계에는 쓸 수 있는 GPU가 없습니다.");
        return;
    }
    // 벡터 256개, 2048 → 2048, "값 열여섯" 커널.
    for repeats in [5, 20, 50, 100, 400, 20] {
        std::thread::sleep(std::time::Duration::from_secs(3));
        let (mut out, mut digest) = ([0.0_f64; 4], 0_u64);
        // SAFETY: `out`은 `f64` 넷, `digest`는 `u64` 하나를 쓸 수 있는 자리다.
        let status = unsafe {
            badak_web_gpu_linear(
                3,
                256,
                2048,
                2048,
                repeats,
                0.0,
                out.as_mut_ptr(),
                &mut digest,
            )
        };
        assert_eq!(status, 0);
        println!(
            "ramp repeats {repeats} wall-us {:.0} gpu-us {:.0} read-us {:.0}",
            out[0] * 1e6,
            out[1] * 1e6,
            out[2] * 1e6
        );
    }
}
