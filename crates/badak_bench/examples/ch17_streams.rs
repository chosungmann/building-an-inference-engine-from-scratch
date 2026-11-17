//! 17장: 스레드 하나가 메모리의 여러 곳을 동시에 읽으면 얼마나 읽는가.
//!
//! 실행: `cargo run --release -p badak_bench --example ch17_streams`
//!
//! 15.2절은 배열을 처음부터 끝까지 한 줄기로 읽었다. 여기서는 1GB 배열을
//! 여러 부분으로 나누고, 부분마다 한 조각씩 번갈아 읽는다. 읽는 양은 같고
//! 줄기의 수만 다르다.

use std::hint::black_box;
use std::time::Instant;

/// 한 번에 읽는 조각의 길이. 64개의 `u64`, 512바이트다.
const PIECE: usize = 64;

// ANCHOR: read
/// 배열을 `streams`개의 부분으로 나누어, 부분마다 한 조각씩 번갈아 읽는다.
fn read_streams(data: &[u64], streams: usize) -> u64 {
    let part = data.len() / streams / PIECE * PIECE;
    let mut sum = 0_u64;
    for start in (0..part).step_by(PIECE) {
        for stream in 0..streams {
            let piece = &data[stream * part + start..][..PIECE];
            sum = piece
                .iter()
                .fold(sum, |sum, &value| sum.wrapping_add(value));
        }
    }
    sum
}
// ANCHOR_END: read

fn main() {
    let data: Vec<u64> = (0..(1_usize << 27)).map(|i| i as u64).collect();
    println!("1GB 배열을 스레드 하나가 읽는 속도");
    println!("{:>8} {:>10}", "줄기", "GB/초");
    for streams in [1, 2, 3, 4, 6, 8, 12, 16] {
        let bytes = (data.len() / streams / PIECE * PIECE * streams * 8) as f64;
        let best = (0..5)
            .map(|_| {
                let start = Instant::now();
                black_box(read_streams(black_box(&data), streams));
                bytes / start.elapsed().as_secs_f64() / 1e9
            })
            .fold(0.0, f64::max);
        println!("{:>8} {:>10.1}", streams, best);
    }
}
