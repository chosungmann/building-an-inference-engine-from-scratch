//! 24장: 새 토큰 하나의 어텐션을 여러 방법으로 계산해 견준다.
//!
//! 실행: `cargo run --release -p badak_models --example ch24_attention`
//!
//! 블록 하나에서 새 토큰 하나가 앞의 토큰들을 보는 일만 떼어 낸다. 크기는
//! Llama 3.2 1B와 같다(질의 헤드 32, 키·값 헤드 8, 헤드의 길이 64). 키와
//! 값은 아무 값으로 채운다. 방법을 하나씩 바꿔 가며 시간을 잰다.

use std::time::Instant;

use badak_bench::stats::Summary;
use badak_core::linalg;
use badak_core::prob::softmax_in_place;
use badak_core::random::Pcg64;
use badak_core::simd::{self, Backend, Walk};
use badak_core::threads;

const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const SHARED: usize = HEADS / KV_HEADS;
const DIM: usize = 64;

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64::new(seed);
    (0..len)
        .map(|_| (rng.next_u64() >> 40) as f32 / (1 << 23) as f32 - 1.0)
        .collect()
}

/// 키와 값. 두 가지 레이아웃으로 들고 있다.
struct Cache {
    positions: usize,
    /// (토큰, 키·값 헤드, 길이). 22장까지의 레이아웃이다.
    mixed_keys: Vec<f32>,
    mixed_values: Vec<f32>,
    /// 헤드마다 (토큰, 길이).
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Cache {
    fn new(positions: usize) -> Self {
        let mixed_keys = values(positions * KV_HEADS * DIM, 1);
        let mixed_values = values(positions * KV_HEADS * DIM, 2);
        let split = |mixed: &[f32]| -> Vec<Vec<f32>> {
            (0..KV_HEADS)
                .map(|head| {
                    (0..positions)
                        .flat_map(|token| {
                            let start = (token * KV_HEADS + head) * DIM;
                            mixed[start..start + DIM].iter().copied()
                        })
                        .collect()
                })
                .collect()
        };
        Self {
            positions,
            keys: split(&mixed_keys),
            values: split(&mixed_values),
            mixed_keys,
            mixed_values,
        }
    }
}

// ANCHOR: before
/// 22장까지의 방법. 헤드마다, 앞의 토큰마다 2장의 내적을 한다.
fn before(cache: &Cache, q: &[f32], output: &mut [f32]) {
    let scale = 1.0 / (DIM as f32).sqrt();
    let vector = |token: usize, head: usize| {
        let start = (token * KV_HEADS + head) * DIM;
        start..start + DIM
    };
    let mut attention = vec![0.0_f32; cache.positions];
    for (head, mixed) in output.chunks_mut(DIM).enumerate() {
        let kv_head = head / SHARED;
        let query = &q[head * DIM..(head + 1) * DIM];
        for (earlier, score) in attention.iter_mut().enumerate() {
            *score = linalg::dot(query, &cache.mixed_keys[vector(earlier, kv_head)]) * scale;
        }
        softmax_in_place(&mut attention);
        for (earlier, &weight) in attention.iter().enumerate() {
            let value = &cache.mixed_values[vector(earlier, kv_head)];
            for (sum, value) in mixed.iter_mut().zip(value) {
                *sum += weight * value;
            }
        }
    }
}
// ANCHOR_END: before

/// 레이아웃만 바꾼다. 헤드의 키들이 나란히 놓여 있다. 내적은 2장의 것 그대로다.
fn contiguous(cache: &Cache, q: &[f32], output: &mut [f32]) {
    let scale = 1.0 / (DIM as f32).sqrt();
    let mut attention = vec![0.0_f32; cache.positions];
    for (head, mixed) in output.chunks_mut(DIM).enumerate() {
        let kv_head = head / SHARED;
        let query = &q[head * DIM..(head + 1) * DIM];
        let keys = cache.keys[kv_head].as_chunks::<DIM>().0;
        for (score, key) in attention.iter_mut().zip(keys) {
            *score = linalg::dot(query, key) * scale;
        }
        softmax_in_place(&mut attention);
        let values = cache.values[kv_head].as_chunks::<DIM>().0;
        for (&weight, value) in attention.iter().zip(values) {
            for (sum, value) in mixed.iter_mut().zip(value) {
                *sum += weight * value;
            }
        }
    }
}

// ANCHOR: unit
/// 키·값 헤드 하나와, 그것을 함께 쓰는 질의 헤드 `count`개를 계산한다.
/// 점수는 17장의 커널로 낸다. `attention`은 `(count, positions)`의 자리다.
fn unit(cache: &Cache, kv_head: usize, queries: &[f32], output: &mut [f32], attention: &mut [f32]) {
    let (positions, count) = (cache.positions, queries.len() / DIM);
    let scale = 1.0 / (DIM as f32).sqrt();
    let walk = Walk::for_vectors(count);
    let keys = &cache.keys[kv_head];
    simd::linear_with(
        Backend::detect(),
        walk,
        queries,
        count,
        keys,
        positions,
        DIM,
        attention,
    );
    for scores in attention.chunks_exact_mut(positions) {
        for score in scores.iter_mut() {
            *score *= scale;
        }
        softmax_in_place(scores);
    }
    for (position, value) in cache.values[kv_head]
        .as_chunks::<DIM>()
        .0
        .iter()
        .enumerate()
    {
        for (head, sums) in output.as_chunks_mut::<DIM>().0.iter_mut().enumerate() {
            let weight = attention[head * positions + position];
            for (sum, value) in sums.iter_mut().zip(value) {
                *sum += weight * value;
            }
        }
    }
}
// ANCHOR_END: unit

/// 17장의 커널로, 질의 헤드를 하나씩.
fn kernel_one(cache: &Cache, q: &[f32], output: &mut [f32]) {
    let mut attention = vec![0.0_f32; cache.positions];
    for (head, mixed) in output.chunks_mut(DIM).enumerate() {
        let query = &q[head * DIM..(head + 1) * DIM];
        unit(cache, head / SHARED, query, mixed, &mut attention);
    }
}

/// 17장의 커널로, 키·값 헤드를 함께 쓰는 질의 헤드 넷을 한꺼번에.
fn kernel_four(cache: &Cache, q: &[f32], output: &mut [f32]) {
    let mut attention = vec![0.0_f32; SHARED * cache.positions];
    for (kv_head, mixed) in output.chunks_mut(SHARED * DIM).enumerate() {
        let queries = &q[kv_head * SHARED * DIM..(kv_head + 1) * SHARED * DIM];
        unit(cache, kv_head, queries, mixed, &mut attention);
    }
}

/// 위와 같되 키·값 헤드 여덟을 스레드에 나눈다.
fn threaded(cache: &Cache, q: &[f32], output: &mut [f32], count: usize) {
    threads::global()
        .pool()
        .run_chunks(output, SHARED * DIM, count, &|kv_head, mixed| {
            let queries = &q[kv_head * SHARED * DIM..(kv_head + 1) * SHARED * DIM];
            let mut attention = vec![0.0_f32; SHARED * cache.positions];
            unit(cache, kv_head, queries, mixed, &mut attention);
        });
}

fn microseconds(mut work: impl FnMut()) -> f64 {
    work();
    let times: Vec<f64> = (0..21)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    Summary::new(&times).median
}

fn main() {
    println!("블록 하나에서 새 토큰 하나의 어텐션 (마이크로초, 되풀이 21번의 중앙값)");
    println!(
        "{:<44} {:>10} {:>10} {:>10}",
        "방법 \\ 앞의 토큰 수", 256, 1024, 4096
    );
    let caches: Vec<Cache> = [256, 1024, 4096].into_iter().map(Cache::new).collect();
    let q = values(HEADS * DIM, 3);
    type Method<'a> = (&'a str, Box<dyn Fn(&Cache, &[f32], &mut [f32])>);
    let methods: Vec<Method> = vec![
        ("22장까지", Box::new(before)),
        ("키를 헤드마다 나란히", Box::new(contiguous)),
        ("17장의 커널로, 질의를 하나씩", Box::new(kernel_one)),
        ("17장의 커널로, 질의 넷을 한꺼번에", Box::new(kernel_four)),
        (
            "헤드를 스레드 8개에 나눠",
            Box::new(|c, q, o| threaded(c, q, o, 8)),
        ),
    ];
    let mut reference: Vec<Vec<f32>> = Vec::new();
    for (index, (label, method)) in methods.iter().enumerate() {
        let mut row = Vec::new();
        for (column, cache) in caches.iter().enumerate() {
            let mut output = vec![0.0_f32; HEADS * DIM];
            row.push(microseconds(|| {
                output.fill(0.0);
                method(cache, &q, &mut output);
            }));
            // 첫 방법의 결과와 견준다. 내적의 순서가 달라 비트는 다를 수 있다.
            if index == 0 {
                reference.push(output);
            } else {
                for (got, want) in output.iter().zip(&reference[column]) {
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "{label}: {got} 대 {want}"
                    );
                }
            }
        }
        let width = 44 - label.chars().filter(|c| !c.is_ascii()).count();
        println!(
            "{:<width$} {:>10.1} {:>10.1} {:>10.1}",
            label, row[0], row[1], row[2]
        );
    }

    // 마지막 방법에서 시간이 어디에 쓰이는지. 스레드 하나로, 4,096토큰에서.
    let cache = &caches[2];
    let positions = cache.positions;
    let queries = &q[..SHARED * DIM];
    let mut attention = vec![0.0_f32; SHARED * positions];
    let walk = Walk::for_vectors(SHARED);
    let score = microseconds(|| {
        simd::linear_with(
            Backend::detect(),
            walk,
            queries,
            SHARED,
            &cache.keys[0],
            positions,
            DIM,
            &mut attention,
        );
        std::hint::black_box(&mut attention);
    });
    let mut scratch = attention.clone();
    let normalize = microseconds(|| {
        scratch.copy_from_slice(&attention);
        for scores in scratch.chunks_exact_mut(positions) {
            softmax_in_place(scores);
        }
    });
    let mut output = vec![0.0_f32; SHARED * DIM];
    let mix = microseconds(|| {
        for (position, value) in cache.values[0].as_chunks::<DIM>().0.iter().enumerate() {
            for (head, sums) in output.as_chunks_mut::<DIM>().0.iter_mut().enumerate() {
                let weight = scratch[head * positions + position];
                for (sum, value) in sums.iter_mut().zip(value) {
                    *sum += weight * value;
                }
            }
        }
        std::hint::black_box(&mut output);
    });
    println!();
    println!("키·값 헤드 하나(질의 넷, 앞의 토큰 4,096개)의 세 단계 (마이크로초, 스레드 하나)");
    println!("  점수: 질의와 키의 내적          {score:>8.1}");
    println!("  비중: 소프트맥스                {normalize:>8.1}");
    println!("  섞기: 값에 비중을 곱해 더하기   {mix:>8.1}");
}
