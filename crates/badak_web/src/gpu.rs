//! GPU 쪽의 문 (49장).
//!
//! 23장의 GPU 백엔드를 브라우저에서 부른다. 커널도, 선형 층을 GPU에 맡기는
//! 코드도 23장의 것 그대로다. 달라진 것은 그 아래다. 맥에서는 `wgpu`가
//! Metal을 부르고, 브라우저에서는 들여온 함수가 브라우저의 WebGPU를 부른다.
//!
//! 이 파일의 함수들은 맥에서도 그대로 부를 수 있다
//! (`examples/ch49_native.rs`). 같은 일을 두 곳에서 시켜 견주려는 것이다.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use badak_core::random::Pcg64;
use badak_core::simd;
use badak_ffi::Llm;
use badak_gpu::Gpu;
use badak_gpu::linear::{Linear, Variant};
use badak_gpu::offload::Offload;
use badak_models::llama::cache::KvCache;
use badak_models::tokenizer::TokenId;

use crate::fail;

static GPU: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();
/// 가장 최근에 모델을 맡긴 창구. 시간이 어디에 쓰였는지 물을 때 쓴다.
static OFFLOAD: Mutex<Option<Arc<Offload>>> = Mutex::new(None);

fn gpu() -> Option<Arc<Gpu>> {
    GPU.get_or_init(|| {
        // 표준 WebGPU의 기능만 쓴다. 브라우저와 맥이 같은 조건이 된다.
        // 맥에서 `BADAK_GPU=full`을 주면 표준 밖의 기능도 쓴다(23장).
        let full = std::env::var("BADAK_GPU").as_deref() == Ok("full");
        if full { Gpu::new() } else { Gpu::standard() }.map(Arc::new)
    })
    .clone()
}

/// GPU를 연다. 쓸 수 있으면 1을 돌려준다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_web_gpu_open() -> i32 {
    i32::from(gpu().is_some())
}

/// GPU의 이름을 `pointer`에 채우고 전체 길이를 돌려준다.
///
/// # Safety
///
/// `pointer`부터 `capacity`바이트를 쓸 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_gpu_name(pointer: *mut u8, capacity: usize) -> usize {
    let Some(gpu) = gpu() else { return 0 };
    let name = gpu.name().as_bytes();
    // SAFETY: 부르는 쪽의 약속.
    unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), pointer, name.len().min(capacity)) };
    name.len()
}

// ANCHOR: linear
/// 선형 층 하나를 GPU에서 여러 번 계산한다.
///
/// `variant`는 커널의 종류다(0: 단순, 1: 넷씩 네 줄기, 2: CPU와 같은 순서,
/// 3: 값 열여섯). 재기 전에 `warm`초 동안 같은 커널을 돌려 GPU를 데운다.
/// GPU는 한가하면 느리게 돌다가 일이 이어지면 빨라지기 때문이다.
/// `out`의 네 칸에 차례로 적는다: 명령을 보내고 끝나기를
/// 기다린 시간(초, 가장 빨랐던 것), GPU가 스스로 잰 계산 시간(초, 잴 수
/// 없으면 -1), 결과를 읽어 오는 시간(초), CPU의 결과와 견준 가장 큰 상대
/// 오차. `digest`에는 결과의 지문을 둔다. 실패하면 -1을 돌려준다.
///
/// # Safety
///
/// `out`은 `f64` 넷, `digest`는 `u64` 하나를 쓸 수 있는 자리여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_gpu_linear(
    variant: i32,
    count: usize,
    rows: usize,
    cols: usize,
    repeats: usize,
    warm: f64,
    out: *mut f64,
    digest: *mut u64,
) -> i32 {
    let Some(gpu) = gpu() else {
        return fail::<()>("쓸 수 있는 GPU가 없습니다").map_or(-1, |()| -1);
    };
    let variant = match variant {
        0 => Variant::Simple,
        1 => Variant::Wide,
        2 => Variant::Exact,
        3 => Variant::Block,
        other => return fail::<()>(format!("모르는 커널 {other}")).map_or(-1, |()| -1),
    };
    if !variant.supports(count, rows, cols) {
        return fail::<()>("이 커널은 이 형상을 계산하지 못합니다").map_or(-1, |()| -1);
    }
    // 48장의 `badak_web_bench_linear`와 같은 값이다. 지문을 견줄 수 있다.
    let values = |count: usize, seed: usize| -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 37 + seed * 101) % 29) as f32 * 0.037 - 0.5)
            .collect()
    };
    let (x, w) = (values(count * cols, 3), values(rows * cols, 4));
    let mut expected = vec![0.0_f32; count * rows];
    simd::linear(&x, count, &w, rows, cols, &mut expected);

    let (x_buffer, w_buffer) = (gpu.upload(&x), gpu.upload(&w));
    let y_buffer = gpu.zeros::<f32>(count * rows);
    let shape = gpu.uniform(&[count as u32, rows as u32, cols as u32, 0]);
    let linear = Linear::new(&gpu, variant);
    let groups = linear.groups(count, rows);
    let buffers = [&shape, &x_buffer, &w_buffer, &y_buffer];
    let mut y = vec![0.0_f32; count * rows];
    let (mut wall, mut inside, mut read) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    // 데우기. 적어도 한 번은 돌린다. 셰이더가 이때 GPU의 기계어로 옮겨진다.
    let started = Instant::now();
    loop {
        gpu.run_and_wait(linear.kernel(), &buffers, groups);
        if started.elapsed().as_secs_f64() >= warm {
            break;
        }
    }
    for repeat in 0..repeats.max(1) + 1 {
        let (outside, timed) = gpu.run_timed(linear.kernel(), &buffers, groups);
        let started = Instant::now();
        gpu.read_into(&y_buffer, &mut y);
        if repeat > 0 {
            wall = wall.min(outside.as_secs_f64());
            inside = inside.min(timed.map_or(f64::INFINITY, |timed| timed.as_secs_f64()));
            read = read.min(started.elapsed().as_secs_f64());
        }
    }
    let error = y
        .iter()
        .zip(&expected)
        .map(|(&got, &want)| f64::from((got - want).abs()) / f64::from(want.abs().max(1.0)))
        .fold(0.0_f64, f64::max);
    let fingerprint = y.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01b3)
    });
    // SAFETY: 부르는 쪽의 약속.
    unsafe {
        *out = wall;
        *out.add(1) = if inside.is_finite() { inside } else { -1.0 };
        *out.add(2) = read;
        *out.add(3) = error;
        *digest = fingerprint;
    }
    0
}
// ANCHOR_END: linear

// ANCHOR: offload
/// 언어 모델의 선형 층을 GPU에 맡긴다. 23장의 `load_on_gpu`와 같은 일이다.
///
/// `mode`가 1이면 블록의 선형 층 일곱 개를 하나씩 맡기고, 2면 MLP는 통째로
/// 맡긴다. 입력 벡터가 `min_count`개보다 적은 일은 CPU에 남는다. 임베딩
/// 표는 CPU에 둔다. 성공하면 GPU에 올린 바이트 수를 MB로 돌려주고,
/// 실패하면 -1이다.
///
/// # Safety
///
/// `llm`은 [`badak_web_llm_open`](crate::badak_web_llm_open)이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_llm_offload(llm: *mut Llm, mode: i32, min_count: usize) -> f64 {
    let Some(gpu) = gpu() else {
        return fail::<()>("쓸 수 있는 GPU가 없습니다").map_or(-1.0, |()| -1.0);
    };
    let offload = Offload::new(gpu);
    offload.set_min_count(min_count.max(1));
    // SAFETY: 부르는 쪽의 약속.
    let model = unsafe { &mut *llm }.model_mut();
    let whole_mlp = mode == 2;
    let mut bytes = 0_usize;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        if whole_mlp {
            model.override_mlps(|_, gate, up, down| {
                bytes += (gate.len() + up.len() + down.len()) * size_of::<f32>();
                Ok(offload.mlp(gate, up, down)?)
            })?;
        }
        model.replace_matrices(|kind, _, matrix| {
            let in_mlp = matches!(kind, "gate_proj" | "up_proj" | "down_proj");
            if kind == "embedding" || (whole_mlp && in_mlp) {
                return Ok(matrix.clone());
            }
            bytes += matrix.len() * size_of::<f32>();
            Ok(offload.matrix(matrix.clone())?)
        })?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            *OFFLOAD.lock().unwrap_or_else(PoisonError::into_inner) = Some(offload);
            bytes as f64 / 1e6
        }
        Err(error) => fail::<()>(error).map_or(-1.0, |()| -1.0),
    }
}
// ANCHOR_END: offload

/// 프롬프트를 한꺼번에 처리하는 속도와 토큰을 하나씩 처리하는 속도를 잰다.
/// 23장의 `speed`와 같은 방법이고 같은 토큰을 쓴다.
///
/// `out`의 두 칸에 프롬프트 처리와 토큰 생성의 속도(토큰/초, 되풀이의
/// 중앙값)를 적는다. 첫 번째 되풀이는 준비 운동으로 치고 버린다.
///
/// # Safety
///
/// `llm`은 [`badak_web_llm_open`](crate::badak_web_llm_open)이 준 포인터여야
/// 하고, `out`은 `f64` 둘을 쓸 수 있는 자리여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_llm_speed(
    llm: *const Llm,
    prompt_tokens: usize,
    new_tokens: usize,
    repeats: usize,
    out: *mut f64,
) -> i32 {
    // SAFETY: 부르는 쪽의 약속.
    let model = unsafe { &*llm }.model();
    let mut rng = Pcg64::new(23);
    let vocab = model.config.vocab_size as u64;
    let tokens: Vec<TokenId> = (0..prompt_tokens + new_tokens)
        .map(|_| (rng.next_u64() % vocab) as TokenId)
        .collect();
    let (mut prompt, mut generation) = (Vec::new(), Vec::new());
    for repetition in 0..repeats.max(1) + 1 {
        let mut cache = KvCache::new(&model.config);
        cache.reserve(tokens.len());
        let start = Instant::now();
        if let Err(error) = model.forward_cached(&tokens[..prompt_tokens], &mut cache) {
            return fail::<()>(error).map_or(-1, |()| -1);
        }
        let prompt_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for token in &tokens[prompt_tokens..] {
            if let Err(error) = model.forward_cached(std::slice::from_ref(token), &mut cache) {
                return fail::<()>(error).map_or(-1, |()| -1);
            }
        }
        let generation_seconds = start.elapsed().as_secs_f64();
        if repetition > 0 {
            prompt.push(prompt_tokens as f64 / prompt_seconds);
            generation.push(new_tokens as f64 / generation_seconds);
        }
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    // SAFETY: 부르는 쪽의 약속.
    unsafe {
        *out = median(&mut prompt);
        *out.add(1) = median(&mut generation);
    }
    0
}

/// 가장 최근에 맡긴 모델에서 GPU의 일에 든 시간을 `out`의 다섯 칸에 적고
/// 기록을 0으로 되돌린다: 입력을 적는 시간, 계산을 기다린 시간, 결과를
/// 읽어 온 시간(초), GPU에서 계산한 선형 층의 수, CPU에 남은 선형 층의 수.
///
/// # Safety
///
/// `out`은 `f64` 다섯을 쓸 수 있는 자리여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_gpu_statistics(out: *mut f64) {
    let offload = OFFLOAD.lock().unwrap_or_else(PoisonError::into_inner);
    let statistics = offload
        .as_ref()
        .map(|offload| offload.statistics(true))
        .unwrap_or_default();
    // SAFETY: 부르는 쪽의 약속.
    unsafe {
        *out = statistics.upload.as_secs_f64();
        *out.add(1) = statistics.compute.as_secs_f64();
        *out.add(2) = statistics.download.as_secs_f64();
        *out.add(3) = statistics.calls as f64;
        *out.add(4) = statistics.on_cpu as f64;
    }
}
