//! `badak`을 브라우저에서 부르는 문 (48장).
//!
//! 44장의 문은 Swift와 Java가 불렀다. 이 문은 JavaScript가 부른다. 엔진은
//! WebAssembly 모듈 하나가 되고, 이 파일의 함수들이 그 모듈이 밖으로 내놓는
//! 것의 전부다.
//!
//! 폰의 문과 다른 점이 둘 있다.
//!
//! - **JavaScript는 모듈의 메모리를 통째로 본다.** 모듈의 메모리는 큰 바이트
//!   배열 하나이고, 포인터는 그 배열의 위치다. 글이나 배열을 주고받으려면
//!   JavaScript가 그 배열에 직접 쓰고 읽는다. 쓸 자리는 모듈에게 달라고
//!   한다([`badak_web_alloc`]).
//! - **거꾸로 부르기는 "들여오기"로 한다.** 모듈은 만들어질 때 밖의 함수를
//!   들여올 수 있다. 토큰이 나올 때마다 부를 함수를 그렇게 받는다.

pub mod gpu;

#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use badak_core::simd::{self, Backend};
use badak_ffi::Llm;
use badak_formats::tflite::Model;
use badak_graph::{Graph, Interpreter, Registry};

/// 마지막 오류의 설명. 스레드마다 두지 않고 하나만 둔다. 이 문의 함수들은
/// 한 스레드(엔진을 올린 워커)만 부른다.
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

pub(crate) fn fail<T>(error: impl std::fmt::Display) -> Option<T> {
    *LAST_ERROR.lock().unwrap_or_else(PoisonError::into_inner) = error.to_string();
    None
}

// ANCHOR: imports
// 모듈이 밖에서 들여오는 함수. JavaScript가 모듈을 만들 때 건네준다.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "badak")]
unsafe extern "C" {
    /// 글 조각 하나가 완성되었다. 0을 돌려주면 계속, 아니면 그만.
    fn piece(pointer: *const u8, length: usize) -> i32;
}

/// 맥에서 돌릴 때의 자리 채움. 조각과 그것이 온 때를 모아 둔다.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn piece(pointer: *const u8, length: usize) -> i32 {
    // SAFETY: 부르는 쪽이 넘긴 `length`바이트의 글이다.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    PIECES.with(|pieces| pieces.borrow_mut().push((Instant::now(), bytes.to_vec())));
    0
}
// ANCHOR_END: imports

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static PIECES: RefCell<Vec<(Instant, Vec<u8>)>> = const { RefCell::new(Vec::new()) };
}

/// 맥에서 돌릴 때, 지금까지 모인 조각들을 꺼내 간다.
#[cfg(not(target_arch = "wasm32"))]
pub fn take_pieces() -> Vec<(Instant, Vec<u8>)> {
    PIECES.with(|pieces| std::mem::take(&mut *pieces.borrow_mut()))
}

// ANCHOR: alloc
/// 모듈의 메모리에서 `length`바이트의 자리를 얻는다. JavaScript가 글이나
/// 배열을 넘기려고 부른다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_web_alloc(length: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(length.max(1));
    let pointer = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    pointer
}

/// [`badak_web_alloc`]으로 얻은 자리를 돌려준다.
///
/// # Safety
///
/// `pointer`와 `length`는 `badak_web_alloc`이 준 그대로여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_free(pointer: *mut u8, length: usize) {
    // SAFETY: 부르는 쪽의 약속.
    drop(unsafe { Vec::from_raw_parts(pointer, 0, length.max(1)) });
}
// ANCHOR_END: alloc

/// 메모리의 한 구간을 글로 읽는다.
///
/// # Safety
///
/// `pointer`부터 `length`바이트를 읽을 수 있어야 한다.
unsafe fn text<'a>(pointer: *const u8, length: usize) -> Option<&'a str> {
    // SAFETY: 부르는 쪽의 약속.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    std::str::from_utf8(bytes)
        .ok()
        .or_else(|| fail("UTF-8이 아닌 글입니다"))
}

/// 마지막 오류의 설명을 `pointer`에 채우고 전체 길이를 돌려준다.
///
/// # Safety
///
/// `pointer`부터 `capacity`바이트를 쓸 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_error(pointer: *mut u8, capacity: usize) -> usize {
    let last = LAST_ERROR.lock().unwrap_or_else(PoisonError::into_inner);
    let count = last.len().min(capacity);
    // SAFETY: 부르는 쪽의 약속.
    unsafe { std::ptr::copy_nonoverlapping(last.as_ptr(), pointer, count) };
    last.len()
}

// ANCHOR: backend
/// 이 모듈이 어떤 커널로 빌드되었는지. 0은 기준 구현, 1은 128비트 벡터 명령,
/// 2는 거기에 `relaxed-simd`를 더한 것이다.
///
/// 44장의 폰에서는 실행할 때 프로세서에 물었다. WebAssembly에는 묻는 방법이
/// 없다. 모르는 명령이 하나라도 든 모듈은 브라우저가 아예 올리지 않는다.
/// 그래서 빌드를 여러 벌 만들어 두고, JavaScript가 브라우저에 맞는 것을
/// 골라 올린다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_web_backend() -> i32 {
    match Backend::detect() {
        Backend::Wasm if cfg!(target_feature = "relaxed-simd") => 2,
        Backend::Wasm => 1,
        _ => 0,
    }
}
// ANCHOR_END: backend

/// 엔진이 쓸 스레드의 수를 정한다. 엔진을 처음 쓰기 전에 불러야 한다.
/// 정해졌으면 1을 돌려준다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_web_threads(threads: usize) -> i32 {
    i32::from(badak_core::threads::configure(threads.max(1)))
}

// ANCHOR: bench
/// 선형 층 하나를 여러 번 계산해 가장 빨랐던 한 번의 시간(초)을 돌려준다. `digest`에는
/// 결과 전체를 숫자 하나로 줄인 지문을 둔다. 지문이 같으면 비트가 같다.
///
/// # Safety
///
/// `digest`는 `u64` 하나를 쓸 수 있는 자리를 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_bench_linear(
    rows: usize,
    cols: usize,
    count: usize,
    repeats: usize,
    digest: *mut u64,
) -> f64 {
    let values = |count: usize, seed: usize| -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 37 + seed * 101) % 29) as f32 * 0.037 - 0.5)
            .collect()
    };
    let (x, w) = (values(count * cols, 3), values(rows * cols, 4));
    let mut y = vec![0.0_f32; count * rows];
    // 브라우저의 시계는 무디다(5µs나 100µs 단위). 한 번이 짧으면 여러 번을
    // 묶어 잰다. 한 묶음이 5ms쯤 되게 한다.
    let started = Instant::now();
    simd::linear(&x, count, &w, rows, cols, &mut y);
    let once = started.elapsed().as_secs_f64().max(1e-7);
    let batch = ((0.005 / once) as usize).clamp(1, 10_000);
    let mut best = f64::INFINITY;
    for _ in 0..repeats.max(1) {
        let started = Instant::now();
        for _ in 0..batch {
            simd::linear(&x, count, &w, rows, cols, &mut y);
        }
        best = best.min(started.elapsed().as_secs_f64() / batch as f64);
    }
    // 17장의 `ch17_backend`와 같은 지문(FNV-1a).
    let fingerprint = y.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01b3)
    });
    // SAFETY: 부르는 쪽의 약속.
    unsafe { *digest = fingerprint };
    best
}
// ANCHOR_END: bench

/// 그래프 하나를 돌릴 준비. 26장의 해석기다.
pub struct Network {
    interpreter: Interpreter,
}

/// `.tflite` 파일을 열어 해석기를 만든다. 실패하면 빈 포인터다.
///
/// # Safety
///
/// `path`부터 `length`바이트의 글을 읽을 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_open(path: *const u8, length: usize) -> *mut Network {
    // SAFETY: 부르는 쪽의 약속.
    let Some(path) = (unsafe { text(path, length) }) else {
        return std::ptr::null_mut();
    };
    let open = || -> Result<Network, Box<dyn std::error::Error>> {
        let graph = Graph::from_tflite(&Model::open(path)?)?;
        Ok(Network {
            interpreter: Interpreter::new(&graph, &Registry::builtin())?,
        })
    };
    match open() {
        Ok(network) => Box::into_raw(Box::new(network)),
        Err(error) => fail::<()>(error).map_or(std::ptr::null_mut(), |()| std::ptr::null_mut()),
    }
}

/// 입력을 쓸 자리. JavaScript가 여기에 값을 직접 쓴다.
///
/// # Safety
///
/// `network`는 [`badak_web_graph_open`]이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_input(network: *mut Network) -> *mut f32 {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { (*network).interpreter.input_mut(0).as_mut_ptr() }
}

/// 입력의 값의 수.
///
/// # Safety
///
/// `network`는 [`badak_web_graph_open`]이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_input_len(network: *const Network) -> usize {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { (*network).interpreter.input_shape(0).iter().product() }
}

/// 한 번 실행한다. 성공하면 0이다.
///
/// # Safety
///
/// `network`는 [`badak_web_graph_open`]이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_run(network: *mut Network) -> i32 {
    // SAFETY: 부르는 쪽의 약속.
    match unsafe { (*network).interpreter.run() } {
        Ok(()) => 0,
        Err(error) => fail::<()>(error).map_or(1, |()| 1),
    }
}

/// 출력이 놓인 자리. JavaScript가 여기에서 값을 직접 읽는다.
///
/// # Safety
///
/// `network`는 [`badak_web_graph_open`]이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_output(network: *const Network) -> *const f32 {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { (*network).interpreter.output(0).as_ptr() }
}

/// 출력의 값의 수.
///
/// # Safety
///
/// `network`는 [`badak_web_graph_open`]이 준 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_graph_output_len(network: *const Network) -> usize {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { (*network).interpreter.output_shape(0).iter().product() }
}

/// 모델 폴더를 열어 언어 모델을 만든다. `weights`는 44장의 문과 같다
/// (0: 파일 그대로, 1: `Q8_0`, 2: `Q4_0`). 실패하면 빈 포인터다.
///
/// # Safety
///
/// `directory`부터 `length`바이트의 글을 읽을 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_llm_open(
    directory: *const u8,
    length: usize,
    weights: i32,
) -> *mut Llm {
    // SAFETY: 부르는 쪽의 약속.
    let Some(directory) = (unsafe { text(directory, length) }) else {
        return std::ptr::null_mut();
    };
    match Llm::open(directory, weights) {
        Ok(llm) => Box::into_raw(Box::new(llm)),
        Err(error) => fail::<()>(error).map_or(std::ptr::null_mut(), |()| std::ptr::null_mut()),
    }
}

// ANCHOR: chat
/// 질문 하나에 답한다. 글이 완성되는 대로 들여온 함수 `piece`를 부른다.
/// 쓴 토큰의 수를 돌려주고, 실패하면 -1이다. `system_length`가 0이면
/// 지시문을 넣지 않는다.
///
/// # Safety
///
/// `llm`은 [`badak_web_llm_open`]이 준 포인터여야 하고, 두 글은 적힌 길이만큼
/// 읽을 수 있어야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_llm_chat(
    llm: *const Llm,
    system: *const u8,
    system_length: usize,
    question: *const u8,
    question_length: usize,
    max_tokens: usize,
) -> i32 {
    // SAFETY: 부르는 쪽의 약속.
    let (Some(system), Some(question)) = (unsafe { text(system, system_length) }, unsafe {
        text(question, question_length)
    }) else {
        return -1;
    };
    let system = (!system.is_empty()).then_some(system);
    // SAFETY: 부르는 쪽의 약속. 조각은 이 호출 동안만 살아 있다.
    let written = unsafe { &*llm }.chat(system, question, max_tokens, |text| unsafe {
        piece(text.as_ptr(), text.len()) == 0
    });
    match written {
        Ok(count) => count as i32,
        Err(error) => fail::<()>(error).map_or(-1, |()| -1),
    }
}
// ANCHOR_END: chat

/// 언어 모델을 지운다.
///
/// # Safety
///
/// `llm`은 [`badak_web_llm_open`]이 준 뒤 아직 지우지 않은 포인터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_web_llm_close(llm: *mut Llm) {
    if !llm.is_null() {
        // SAFETY: 부르는 쪽의 약속.
        drop(unsafe { Box::from_raw(llm) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_digest_matches_the_reference() {
        // 48장의 표에 있는 지문. 기준 구현, NEON, AVX2가 같은 값을 낸다.
        let mut digest = 0_u64;
        // SAFETY: `digest`는 `u64` 하나를 쓸 수 있는 자리다.
        let seconds = unsafe { badak_web_bench_linear(13, 203, 9, 2, &mut digest) };
        assert!(seconds > 0.0);
        assert_eq!(digest, 0x7c9c_f3b9_bac3_45e0);
    }

    #[test]
    fn memory_round_trip() {
        let pointer = badak_web_alloc(16);
        assert!(!pointer.is_null());
        // SAFETY: 방금 얻은 16바이트다.
        unsafe {
            std::slice::from_raw_parts_mut(pointer, 16).fill(7);
            badak_web_free(pointer, 16);
        }
    }

    #[test]
    fn missing_model_reports_an_error() {
        let path = "없는/모델.tflite";
        let mut buffer = [0_u8; 256];
        // SAFETY: 문의 약속대로 부른다.
        unsafe {
            assert!(badak_web_graph_open(path.as_ptr(), path.len()).is_null());
            let length = badak_web_error(buffer.as_mut_ptr(), buffer.len());
            assert!(length > 0);
        }
    }

    #[test]
    fn recognizes_the_first_test_digit() {
        let images = std::fs::read(badak_graph::models_dir().join("mnist/t10k-images-idx3-ubyte"))
            .expect("MNIST");
        let path = badak_graph::models_dir().join("mnist_tflite/mlp.tflite");
        let path = path.to_str().expect("UTF-8");
        // SAFETY: 문의 약속대로 부른다. 포인터들은 해석기가 살아 있는 동안 쓴다.
        unsafe {
            let network = badak_web_graph_open(path.as_ptr(), path.len());
            assert!(!network.is_null());
            let input = std::slice::from_raw_parts_mut(
                badak_web_graph_input(network),
                badak_web_graph_input_len(network),
            );
            for (value, &pixel) in input.iter_mut().zip(&images[16..16 + 784]) {
                *value = f32::from(pixel) / 255.0;
            }
            assert_eq!(badak_web_graph_run(network), 0);
            let scores = std::slice::from_raw_parts(
                badak_web_graph_output(network),
                badak_web_graph_output_len(network),
            );
            let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]));
            assert_eq!(best, Some(7));
            drop(Box::from_raw(network));
        }
    }
}
