//! `badak`을 다른 언어에서 부르는 문 (44장).
//!
//! 폰의 앱은 Swift나 Kotlin, Java로 쓴다. 이 언어들이 함께 아는 말은 C의
//! 함수 호출 규약뿐이다. 이 크레이트는 엔진의 일부를 C 함수로 내놓는다.
//! 함수의 선언은 `include/badak.h`에 손으로 적었다.
//!
//! 문에서 지키는 약속은 넷이다.
//!
//! 1. **패닉은 문을 넘지 않는다.** 모든 함수가 [`guard`] 안에서 돈다.
//! 2. **오류는 숫자로 나간다.** 설명은 [`badak_last_error`]로 따로 읽는다.
//! 3. **메모리는 만든 쪽이 지운다.** 엔진이 만든 것은 `badak_*_close`로만
//!    지우고, 글과 배열은 부르는 쪽이 준 자리에 채운다.
//! 4. **넘겨받은 포인터는 쓰기 전에 본다.** 빈 포인터는 오류로 돌려준다.

use std::cell::RefCell;
#[cfg(unix)]
use std::ffi::c_int;
use std::ffi::{CStr, c_char, c_void};
#[cfg(unix)]
use std::fs::File;
use std::panic::{AssertUnwindSafe, catch_unwind};

use badak_formats::store::{Manifest, Store};

pub mod classifier;
#[cfg(any(target_os = "android", test))]
pub mod jni;
pub mod llm;

pub use classifier::{Classifier, Error, Score, Status};
pub use llm::Llm;

thread_local! {
    /// 이 스레드에서 마지막으로 난 오류의 설명.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

// ANCHOR: guard
/// 문 안쪽의 일을 하고 그 결과를 숫자로 바꾼다. 패닉도 숫자가 된다.
///
/// Rust의 패닉이 C의 함수 밖으로 풀려 나가면 프로세스가 그 자리에서
/// 죽는다. 앱이 통째로 죽는 것이다. 여기에서 잡아 오류로 돌려준다.
fn guard(work: impl FnOnce() -> Result<(), Error>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "알 수 없는 패닉".to_owned());
        Err(Error {
            status: Status::Panic,
            message: format!("엔진 안에서 패닉이 났습니다: {message}"),
        })
    });
    match outcome {
        Ok(()) => Status::Ok as i32,
        Err(error) => {
            LAST_ERROR.with(|last| *last.borrow_mut() = error.message);
            error.status as i32
        }
    }
}
// ANCHOR_END: guard

/// C의 글을 Rust의 글로 읽는다.
///
/// # Safety
///
/// `text`는 비어 있거나, 0으로 끝나는 바이트열을 가리켜야 한다.
unsafe fn text<'a>(what: &str, text: *const c_char) -> Result<&'a str, Error> {
    if text.is_null() {
        return Err(Error::argument(format!("{what}이 빈 포인터입니다")));
    }
    // SAFETY: 비어 있지 않음을 보았고, 0으로 끝나는 것은 부르는 쪽의 약속이다.
    unsafe { CStr::from_ptr(text) }
        .to_str()
        .map_err(|_| Error::argument(format!("{what}이 UTF-8이 아닙니다")))
}

/// 글을 부르는 쪽이 준 자리에 채운다. 끝에 0을 붙인다. 자리가 모자라면
/// 들어가는 만큼만 채운다(글자의 가운데에서 자르지 않는다). 0을 뺀 글
/// 전체의 길이를 돌려주므로, 부르는 쪽은 잘렸는지 알 수 있다.
///
/// # Safety
///
/// `buffer`는 비어 있거나, 쓸 수 있는 `capacity`바이트를 가리켜야 한다.
unsafe fn fill(source: &str, buffer: *mut c_char, capacity: usize) -> usize {
    if !buffer.is_null() && capacity > 0 {
        let mut count = source.len().min(capacity - 1);
        while !source.is_char_boundary(count) {
            count -= 1;
        }
        // SAFETY: `count < capacity`이고 두 자리는 겹치지 않는다.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), buffer.cast::<u8>(), count);
            *buffer.add(count) = 0;
        }
    }
    source.len()
}

/// 엔진의 버전. 0으로 끝나는 글이고, 지우지 않는다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// 이 스레드에서 마지막으로 난 오류의 설명을 `buffer`에 채운다. 설명
/// 전체의 길이(끝의 0은 뺀 바이트 수)를 돌려준다.
///
/// # Safety
///
/// `buffer`는 비어 있거나, 쓸 수 있는 `capacity`바이트를 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_last_error(buffer: *mut c_char, capacity: usize) -> usize {
    // SAFETY: 부르는 쪽의 약속을 그대로 넘긴다.
    LAST_ERROR.with(|last| unsafe { fill(&last.borrow(), buffer, capacity) })
}

// ANCHOR: cpu
/// 이 프로세서에서 엔진이 무엇을 쓰는지 한 줄로 적는다.
///
/// "빌드"는 컴파일러가 늘 있다고 믿고 쓴 명령이고, "실행"은 엔진이 켜질 때
/// 프로세서에 물어서 알아낸 것이다.
pub fn cpu_features() -> String {
    let built = [
        ("neon", cfg!(target_feature = "neon")),
        ("dotprod", cfg!(target_feature = "dotprod")),
        ("fp16", cfg!(target_feature = "fp16")),
        ("avx2", cfg!(target_feature = "avx2")),
    ];
    #[cfg(target_arch = "aarch64")]
    let found = [
        ("neon", std::arch::is_aarch64_feature_detected!("neon")),
        (
            "dotprod",
            std::arch::is_aarch64_feature_detected!("dotprod"),
        ),
        ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
    ];
    #[cfg(target_arch = "x86_64")]
    let found = [
        ("avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("fma", std::arch::is_x86_feature_detected!("fma")),
    ];
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let found: [(&str, bool); 0] = [];
    let names = |features: &[(&str, bool)]| {
        let names: Vec<&str> = features
            .iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| *name)
            .collect();
        if names.is_empty() {
            "-".to_owned()
        } else {
            names.join("+")
        }
    };
    format!(
        "{} 빌드 {} 실행 {} 구현 {:?} 스레드 {}",
        std::env::consts::ARCH,
        names(&built),
        names(&found),
        badak_core::simd::Backend::detect(),
        std::thread::available_parallelism().map_or(1, usize::from)
    )
}
// ANCHOR_END: cpu

/// [`cpu_features`]의 글을 `buffer`에 채운다. 글 전체의 길이를 돌려준다.
///
/// # Safety
///
/// `buffer`는 비어 있거나, 쓸 수 있는 `capacity`바이트를 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_cpu_features(buffer: *mut c_char, capacity: usize) -> usize {
    // SAFETY: 부르는 쪽의 약속을 그대로 넘긴다.
    unsafe { fill(&cpu_features(), buffer, capacity) }
}

/// 엔진이 쓸 스레드의 수를 정한다. 모델을 처음 열기 전에 불러야 한다.
/// 이미 정해졌으면 `BADAK_INVALID_ARGUMENT`다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_set_threads(threads: usize) -> i32 {
    guard(|| {
        if threads == 0 || !badak_core::threads::configure(threads) {
            return Err(Error::argument(
                "스레드의 수는 1 이상이어야 하고, 모델을 열기 전에만 정할 수 있습니다",
            ));
        }
        Ok(())
    })
}

// ANCHOR: open
/// 모델 파일을 열어 분류기를 만든다. 성공하면 `*out`에 분류기를 둔다.
///
/// # Safety
///
/// `path`는 0으로 끝나는 글을, `out`은 포인터 하나를 쓸 수 있는 자리를
/// 가리켜야 한다(둘 다 비어 있으면 오류로 돌려준다).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_open(
    path: *const c_char,
    out: *mut *mut Classifier,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Error::argument("out이 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let path = unsafe { text("path", path) }?;
        let classifier = Box::new(Classifier::open(path)?);
        // SAFETY: `out`이 비어 있지 않음을 보았다. 상자를 날 포인터로 바꿔
        // 넘기고, `badak_classifier_close`가 다시 상자로 되돌려 지운다.
        unsafe { *out = Box::into_raw(classifier) };
        Ok(())
    })
}
// ANCHOR_END: open

/// 열려 있는 파일의 한 구간에 든 모델로 분류기를 만든다. Android의 앱
/// 번들 안에 든 모델을 꺼내지 않고 여는 데 쓴다. 파일 디스크립터는 빌려
/// 쓸 뿐이고 닫지 않는다.
///
/// # Safety
///
/// `fd`는 열려 있는 파일 디스크립터여야 하고, `out`은 포인터 하나를 쓸 수
/// 있는 자리를 가리켜야 한다.
#[cfg(unix)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_open_fd(
    fd: c_int,
    offset: i64,
    length: i64,
    out: *mut *mut Classifier,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Error::argument("out이 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let classifier = Box::new(unsafe { open_fd(fd, offset, length) }?);
        // SAFETY: `badak_classifier_open`과 같다.
        unsafe { *out = Box::into_raw(classifier) };
        Ok(())
    })
}

// ANCHOR: open_fd
/// 빌린 파일 디스크립터의 한 구간으로 분류기를 만든다.
///
/// # Safety
///
/// `fd`는 이 함수가 도는 동안 열려 있는 파일 디스크립터여야 한다.
#[cfg(unix)]
pub(crate) unsafe fn open_fd(fd: c_int, offset: i64, length: i64) -> Result<Classifier, Error> {
    use std::mem::ManuallyDrop;
    use std::os::fd::FromRawFd;

    let (Ok(offset), Ok(length)) = (u64::try_from(offset), usize::try_from(length)) else {
        return Err(Error::argument("offset과 length는 0 이상이어야 합니다"));
    };
    if fd < 0 {
        return Err(Error::argument("fd가 열린 파일이 아닙니다"));
    }
    // 디스크립터의 주인은 부르는 쪽이다. `File`이 사라질 때 닫지 않도록
    // 감싸 둔다. 매핑은 디스크립터가 닫힌 뒤에도 남는다.
    // SAFETY: 열려 있다는 것은 부르는 쪽의 약속이다.
    let file = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
    Classifier::open_range(&file, offset, length)
}
// ANCHOR_END: open_fd

/// 분류기를 빌린다.
///
/// # Safety
///
/// `classifier`는 비어 있거나, `badak_classifier_open*`이 준 뒤 아직 닫지
/// 않은 포인터여야 하고, 다른 스레드가 함께 쓰고 있지 않아야 한다.
unsafe fn borrow<'a>(classifier: *mut Classifier) -> Result<&'a mut Classifier, Error> {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { classifier.as_mut() }.ok_or_else(|| Error::argument("분류기가 빈 포인터입니다"))
}

/// 입력의 값의 수. 분류기가 비어 있으면 0이다.
///
/// # Safety
///
/// `classifier`는 비어 있거나 아직 닫지 않은 분류기여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_input_len(classifier: *const Classifier) -> usize {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { classifier.as_ref() }.map_or(0, Classifier::input_len)
}

/// 출력의 값의 수. 분류기가 비어 있으면 0이다.
///
/// # Safety
///
/// `classifier`는 비어 있거나 아직 닫지 않은 분류기여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_output_len(classifier: *const Classifier) -> usize {
    // SAFETY: 부르는 쪽의 약속.
    unsafe { classifier.as_ref() }.map_or(0, Classifier::output_len)
}

/// 숫자로 바꿔 둔 입력을 넣고 점수를 받는다.
///
/// # Safety
///
/// `input`은 읽을 수 있는 `f32` `input_len`개를, `output`은 쓸 수 있는
/// `f32` `output_len`개를 가리켜야 한다. 둘은 겹치지 않아야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_run(
    classifier: *mut Classifier,
    input: *const f32,
    input_len: usize,
    output: *mut f32,
    output_len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: 부르는 쪽의 약속.
        let classifier = unsafe { borrow(classifier) }?;
        if input.is_null() || output.is_null() {
            return Err(Error::argument("input이나 output이 빈 포인터입니다"));
        }
        // SAFETY: 비어 있지 않음을 보았고, 길이와 겹치지 않음은 부르는 쪽의
        // 약속이다.
        let (input, output) = unsafe {
            (
                std::slice::from_raw_parts(input, input_len),
                std::slice::from_raw_parts_mut(output, output_len),
            )
        };
        classifier.run(input, output)
    })
}

// ANCHOR: classify
/// JPEG 파일의 바이트를 받아, 확률이 높은 부류부터 `top`에 채운다.
/// `*written`에 채운 수를 둔다.
///
/// # Safety
///
/// `bytes`는 읽을 수 있는 `len`바이트를, `top`은 쓸 수 있는 `badak_score`
/// `capacity`개를, `written`은 `size_t` 하나를 쓸 수 있는 자리를 가리켜야
/// 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_classify_jpeg(
    classifier: *mut Classifier,
    bytes: *const u8,
    len: usize,
    top: *mut Score,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: 부르는 쪽의 약속.
        let classifier = unsafe { borrow(classifier) }?;
        if bytes.is_null() || top.is_null() || written.is_null() {
            return Err(Error::argument(
                "bytes, top, written 가운데 빈 포인터가 있습니다",
            ));
        }
        // SAFETY: 비어 있지 않음을 보았고, 길이는 부르는 쪽의 약속이다.
        let (bytes, top) = unsafe {
            (
                std::slice::from_raw_parts(bytes, len),
                std::slice::from_raw_parts_mut(top, capacity),
            )
        };
        let count = classifier.classify_jpeg(bytes, top)?;
        // SAFETY: 비어 있지 않음을 보았다.
        unsafe { *written = count };
        Ok(())
    })
}
// ANCHOR_END: classify

/// 분류기를 지운다. 빈 포인터를 주면 아무 일도 하지 않는다.
///
/// # Safety
///
/// `classifier`는 비어 있거나, `badak_classifier_open*`이 준 뒤 아직 닫지
/// 않은 포인터여야 한다. 닫은 뒤에는 그 포인터를 다시 쓰면 안 된다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_classifier_close(classifier: *mut Classifier) {
    if !classifier.is_null() {
        // SAFETY: `Box::into_raw`로 내보낸 포인터를 한 번만 되돌린다.
        let classifier = unsafe { Box::from_raw(classifier) };
        // 지우다가 난 패닉도 문을 넘지 않게 한다.
        let _ = catch_unwind(AssertUnwindSafe(move || drop(classifier)));
    }
}

/// 받은 모델 파일을 확인하고 `directory` 아래에 놓는다. `manifest`는 JSON으로
/// 적힌 명세다. 파일이 명세와 다르면 `BADAK_MISMATCH`이고 쓰던 모델은
/// 그대로다.
///
/// # Safety
///
/// 세 포인터는 0으로 끝나는 글을 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_model_install(
    directory: *const c_char,
    manifest: *const c_char,
    downloaded: *const c_char,
) -> i32 {
    guard(|| {
        // SAFETY: 부르는 쪽의 약속.
        let (directory, manifest, downloaded) = unsafe {
            (
                text("directory", directory)?,
                text("manifest", manifest)?,
                text("downloaded", downloaded)?,
            )
        };
        let manifest = Manifest::parse(manifest)?;
        Store::new(directory).install(&manifest, downloaded)?;
        Ok(())
    })
}

/// 지금 쓰는 버전의 모델 파일의 길을 `buffer`에 채운다. `*length`에 길 전체의
/// 길이(끝의 0은 뺀 바이트 수)를 둔다. 놓인 모델이 없으면 길이가 0이다.
///
/// # Safety
///
/// `directory`와 `name`은 0으로 끝나는 글을, `buffer`는 쓸 수 있는
/// `capacity`바이트를, `length`는 `size_t` 하나를 쓸 수 있는 자리를
/// 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_model_current(
    directory: *const c_char,
    name: *const c_char,
    buffer: *mut c_char,
    capacity: usize,
    length: *mut usize,
) -> i32 {
    guard(|| {
        if length.is_null() {
            return Err(Error::argument("length가 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let (directory, name) = unsafe { (text("directory", directory)?, text("name", name)?) };
        let path = Store::new(directory)
            .current(name)?
            .map(|(_, path)| path.to_string_lossy().into_owned())
            .unwrap_or_default();
        // SAFETY: 부르는 쪽의 약속. `length`는 비어 있지 않음을 보았다.
        unsafe { *length = fill(&path, buffer, capacity) };
        Ok(())
    })
}

/// 앞에 쓰던 버전으로 되돌린다. `*rolled`에 되돌렸으면 1, 되돌릴 버전이 없으면
/// 0을 둔다.
///
/// # Safety
///
/// `directory`와 `name`은 0으로 끝나는 글을, `rolled`는 `int32_t` 하나를 쓸
/// 수 있는 자리를 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_model_rollback(
    directory: *const c_char,
    name: *const c_char,
    rolled: *mut i32,
) -> i32 {
    guard(|| {
        if rolled.is_null() {
            return Err(Error::argument("rolled가 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let (directory, name) = unsafe { (text("directory", directory)?, text("name", name)?) };
        let done = Store::new(directory).rollback(name)?;
        // SAFETY: 비어 있지 않음을 보았다.
        unsafe { *rolled = i32::from(done) };
        Ok(())
    })
}

// ANCHOR: llm_callback
/// 글 조각을 받는 함수. `piece`는 `length`바이트의 UTF-8이고 끝에 0이 없다.
/// 0을 돌려주면 계속 쓰고, 다른 값을 돌려주면 그만 쓴다.
///
/// 엔진이 **부르는 쪽의 함수를 거꾸로 부른다.** `user`는 부르는 쪽이 준
/// 포인터를 그대로 돌려주는 것이고, 엔진은 그것이 무엇인지 모른다.
pub type PieceCallback =
    unsafe extern "C" fn(user: *mut c_void, piece: *const c_char, length: usize) -> i32;
// ANCHOR_END: llm_callback

/// 모델 폴더를 열어 언어 모델을 만든다. `weights`는 0(파일 그대로), 1(`Q8_0`),
/// 2(`Q4_0`) 가운데 하나다.
///
/// # Safety
///
/// `directory`는 0으로 끝나는 글을, `out`은 포인터 하나를 쓸 수 있는 자리를
/// 가리켜야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_llm_open(
    directory: *const c_char,
    weights: i32,
    out: *mut *mut Llm,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Error::argument("out이 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let directory = unsafe { text("directory", directory) }?;
        let llm = Box::new(Llm::open(directory, weights)?);
        // SAFETY: `out`이 비어 있지 않음을 보았다. `badak_llm_close`가 지운다.
        unsafe { *out = Box::into_raw(llm) };
        Ok(())
    })
}

// ANCHOR: llm_chat
/// 질문 하나에 답한다. 글이 완성되는 대로 `callback`을 부르고, 쓴 토큰의
/// 수를 `*written`에 둔다. 이 함수는 답을 다 쓸 때까지 돌아오지 않는다.
/// `callback`은 이 함수를 부른 스레드에서 불린다. `system`은 대화의 맨 앞에
/// 놓는 지시문이고, 비어 있으면 넣지 않는다.
///
/// # Safety
///
/// `llm`은 아직 닫지 않은 모델을, `question`은 0으로 끝나는 글을, `written`은
/// `size_t` 하나를 쓸 수 있는 자리를 가리켜야 한다. `system`은 비어 있거나
/// 0으로 끝나는 글을 가리켜야 한다. `callback`은 이 호출 동안 부를 수 있는
/// 함수여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_llm_chat(
    llm: *mut Llm,
    system: *const c_char,
    question: *const c_char,
    max_tokens: usize,
    callback: Option<PieceCallback>,
    user: *mut c_void,
    written: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: 부르는 쪽의 약속.
        let llm =
            unsafe { llm.as_ref() }.ok_or_else(|| Error::argument("모델이 빈 포인터입니다"))?;
        if written.is_null() {
            return Err(Error::argument("written이 빈 포인터입니다"));
        }
        // SAFETY: 부르는 쪽의 약속.
        let question = unsafe { text("question", question) }?;
        let system = if system.is_null() {
            None
        } else {
            // SAFETY: 비어 있지 않음을 보았고, 0으로 끝나는 것은 약속이다.
            Some(unsafe { text("system", system) }?)
        };
        let count = llm.chat(system, question, max_tokens, |piece| match callback {
            // SAFETY: 조각은 이 호출 동안만 살아 있고, 받는 쪽은 그동안만
            // 읽는다는 것이 약속이다.
            Some(callback) => unsafe { callback(user, piece.as_ptr().cast(), piece.len()) == 0 },
            None => true,
        })?;
        // SAFETY: 비어 있지 않음을 보았다.
        unsafe { *written = count };
        Ok(())
    })
}
// ANCHOR_END: llm_chat

/// 언어 모델을 지운다. 빈 포인터를 주면 아무 일도 하지 않는다.
///
/// # Safety
///
/// `llm`은 비어 있거나, `badak_llm_open`이 준 뒤 아직 닫지 않은 포인터여야
/// 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn badak_llm_close(llm: *mut Llm) {
    if !llm.is_null() {
        // SAFETY: `Box::into_raw`로 내보낸 포인터를 한 번만 되돌린다.
        let llm = unsafe { Box::from_raw(llm) };
        let _ = catch_unwind(AssertUnwindSafe(move || drop(llm)));
    }
}

/// 일부러 패닉을 낸다. 패닉이 문을 넘지 않는지 시험하는 데만 쓴다.
#[doc(hidden)]
#[unsafe(no_mangle)]
pub extern "C" fn badak_test_panic() -> i32 {
    guard(|| panic!("시험으로 낸 패닉"))
}
