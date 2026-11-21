//! Android의 Java와 Kotlin이 부르는 함수들 (44장).
//!
//! Java는 C 함수를 바로 부르지 못한다. JNI(Java Native Interface)의 규칙을
//! 따르는 함수만 부른다. 규칙은 둘이다. 함수의 이름이 `Java_패키지_클래스_
//! 메서드`의 꼴이고, 첫 두 인자가 `JNIEnv*`와 클래스다.
//!
//! `JNIEnv`는 **함수 포인터의 표를 가리키는 포인터**다. Java의 글이나 배열은
//! 속을 볼 수 없는 손잡이로 오고, 그 속을 읽고 쓰려면 이 표의 함수를 불러야
//! 한다. 표의 순서는 JNI 명세가 정했고 바뀌지 않는다. 필요한 칸의 번호만
//! 여기에 적었다. 번호는 NDK의 `jni.h`에서 세어 확인했다.

use std::ffi::{CString, c_char, c_void};

use crate::classifier::{Classifier, Error, Score, Status};
use crate::{LAST_ERROR, guard};

// ANCHOR: env
/// 표의 칸의 수 (JNI 1.6).
const SLOTS: usize = 233;

/// `JNIEnv`. 표를 가리키는 포인터를 가리킨다.
pub type Env = *mut *const [*const c_void; SLOTS];
/// Java 객체의 손잡이.
pub type Object = *mut c_void;

const DELETE_LOCAL_REF: usize = 23;
const GET_OBJECT_CLASS: usize = 31;
const GET_METHOD_ID: usize = 33;
const CALL_BOOLEAN_METHOD_A: usize = 39;
const NEW_STRING_UTF: usize = 167;
const GET_STRING_UTF_CHARS: usize = 169;
const RELEASE_STRING_UTF_CHARS: usize = 170;
const GET_ARRAY_LENGTH: usize = 171;
const NEW_BYTE_ARRAY: usize = 176;
const GET_BYTE_ARRAY_REGION: usize = 200;
const GET_FLOAT_ARRAY_REGION: usize = 205;
const SET_BYTE_ARRAY_REGION: usize = 208;
const SET_INT_ARRAY_REGION: usize = 211;
const SET_FLOAT_ARRAY_REGION: usize = 213;
const EXCEPTION_CHECK: usize = 228;

/// 표의 `slot`번 칸을 함수 포인터로 꺼낸다.
///
/// # Safety
///
/// `env`는 살아 있는 `JNIEnv`여야 하고, `F`는 그 칸의 함수의 타입이어야 한다.
unsafe fn function<F: Copy>(env: Env, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<*const c_void>()) };
    // SAFETY: 부르는 쪽의 약속. 함수 포인터와 데이터 포인터의 크기가 같음은
    // 위에서 컴파일할 때 확인했다.
    unsafe { std::mem::transmute_copy(&(**env)[slot]) }
}
// ANCHOR_END: env

type RegionGetter<T> = unsafe extern "C" fn(Env, Object, i32, i32, *mut T);
type RegionSetter<T> = unsafe extern "C" fn(Env, Object, i32, i32, *const T);

/// Java 배열의 길이.
unsafe fn array_len(env: Env, array: Object) -> Result<usize, Error> {
    if array.is_null() {
        return Err(Error::argument("배열이 null입니다"));
    }
    // SAFETY: 171번 칸은 `jsize GetArrayLength(JNIEnv*, jarray)`다.
    let length = unsafe {
        function::<unsafe extern "C" fn(Env, Object) -> i32>(env, GET_ARRAY_LENGTH)(env, array)
    };
    Ok(length.max(0) as usize)
}

// ANCHOR: copy_in
/// Java 배열의 내용을 Rust의 벡터로 복사해 온다.
unsafe fn read_array<T: Copy + Default>(
    env: Env,
    array: Object,
    slot: usize,
) -> Result<Vec<T>, Error> {
    // SAFETY: 부르는 쪽의 약속을 그대로 넘긴다.
    let length = unsafe { array_len(env, array) }?;
    let mut values = vec![T::default(); length];
    // SAFETY: `slot`은 `T`에 맞는 `Get<T>ArrayRegion`이고, 받는 자리의
    // 길이가 `length`다.
    unsafe {
        function::<RegionGetter<T>>(env, slot)(env, array, 0, length as i32, values.as_mut_ptr());
    }
    Ok(values)
}
// ANCHOR_END: copy_in

/// Rust의 값들을 Java 배열의 앞쪽에 써 넣는다. 배열이 더 짧으면 오류다.
unsafe fn write_array<T>(env: Env, array: Object, slot: usize, values: &[T]) -> Result<(), Error> {
    // SAFETY: 부르는 쪽의 약속을 그대로 넘긴다.
    if unsafe { array_len(env, array) }? < values.len() {
        return Err(Error::argument("결과를 받을 배열이 짧습니다"));
    }
    // SAFETY: `slot`은 `T`에 맞는 `Set<T>ArrayRegion`이고, 배열에 자리가
    // 있음을 위에서 보았다.
    unsafe {
        function::<RegionSetter<T>>(env, slot)(env, array, 0, values.len() as i32, values.as_ptr());
    }
    Ok(())
}

/// Java의 글을 Rust의 글로 복사해 온다.
unsafe fn read_string(env: Env, string: Object) -> Result<String, Error> {
    if string.is_null() {
        return Err(Error::argument("글이 null입니다"));
    }
    type Get = unsafe extern "C" fn(Env, Object, *mut u8) -> *const c_char;
    type Release = unsafe extern "C" fn(Env, Object, *const c_char);
    // SAFETY: 169번과 170번 칸의 타입이다. 받은 글은 바로 복사하고 돌려준다.
    unsafe {
        let chars = function::<Get>(env, GET_STRING_UTF_CHARS)(env, string, std::ptr::null_mut());
        if chars.is_null() {
            return Err(Error::argument("글을 읽지 못했습니다"));
        }
        let text = std::ffi::CStr::from_ptr(chars)
            .to_string_lossy()
            .into_owned();
        function::<Release>(env, RELEASE_STRING_UTF_CHARS)(env, string, chars);
        Ok(text)
    }
}

/// Rust의 글로 Java의 글을 만든다.
///
/// JNI가 받는 것은 UTF-8이 아니라 "고친 UTF-8"이다. 0과, 네 바이트로 적는
/// 글자(이모지 등)를 적는 법이 다르다. 한글을 포함해 세 바이트까지의
/// 글자는 같다. 다른 것들은 물음표로 바꿔 넘긴다.
unsafe fn new_string(env: Env, text: &str) -> Object {
    let plain: String = text
        .chars()
        .map(|character| match character {
            '\0' => ' ',
            character if character.len_utf8() > 3 => '?',
            character => character,
        })
        .collect();
    let text = CString::new(plain).expect("0을 지웠다");
    // SAFETY: 167번 칸은 `jstring NewStringUTF(JNIEnv*, const char*)`다.
    unsafe {
        function::<unsafe extern "C" fn(Env, *const c_char) -> Object>(env, NEW_STRING_UTF)(
            env,
            text.as_ptr(),
        )
    }
}

/// `book.badak.Badak.version()`.
///
/// # Safety
///
/// JVM만 부른다. `env`는 살아 있는 `JNIEnv`여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_version(env: Env, _class: Object) -> Object {
    // SAFETY: JVM이 준 `env`다.
    unsafe { new_string(env, env!("CARGO_PKG_VERSION")) }
}

/// `book.badak.Badak.cpuFeatures()`.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_cpuFeatures(env: Env, _class: Object) -> Object {
    // SAFETY: JVM이 준 `env`다.
    unsafe { new_string(env, &crate::cpu_features()) }
}

/// `book.badak.Badak.lastError()`. 이 스레드에서 마지막으로 난 오류의 설명.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_lastError(env: Env, _class: Object) -> Object {
    // SAFETY: JVM이 준 `env`다.
    LAST_ERROR.with(|last| unsafe { new_string(env, &last.borrow()) })
}

// ANCHOR: jni_open
/// `book.badak.Badak.open(int fd, long offset, long length)`. 분류기를
/// 가리키는 주소를 `long`으로 돌려준다. 실패하면 0이다.
///
/// # Safety
///
/// JVM만 부른다. `fd`는 열려 있는 파일 디스크립터여야 한다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_open(
    _env: Env,
    _class: Object,
    fd: i32,
    offset: i64,
    length: i64,
) -> i64 {
    let mut handle = 0_i64;
    guard(|| {
        // SAFETY: 부르는 쪽의 약속.
        let classifier = unsafe { crate::open_fd(fd, offset, length) }?;
        handle = Box::into_raw(Box::new(classifier)) as i64;
        Ok(())
    });
    handle
}
// ANCHOR_END: jni_open

/// `long`으로 온 주소를 분류기로 빌린다.
unsafe fn borrow<'a>(handle: i64) -> Result<&'a mut Classifier, Error> {
    // SAFETY: Java 쪽이 `open`이 준 값을 `close`하기 전까지만 넘긴다.
    unsafe { (handle as *mut Classifier).as_mut() }
        .ok_or_else(|| Error::argument("닫힌 분류기입니다"))
}

// ANCHOR: jni_classify
/// `book.badak.Badak.classifyJpeg(long, byte[], int[], float[])`. 확률이 높은
/// 부류부터 두 배열에 채우고 채운 수를 돌려준다. 실패하면 오류의 종류에
/// 음수 부호를 붙여 돌려준다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_classifyJpeg(
    env: Env,
    _class: Object,
    handle: i64,
    jpeg: Object,
    indices: Object,
    probabilities: Object,
) -> i32 {
    let mut count = 0;
    let status = guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이들이다.
        unsafe {
            let classifier = borrow(handle)?;
            let bytes: Vec<i8> = read_array(env, jpeg, GET_BYTE_ARRAY_REGION)?;
            // Java의 byte는 부호가 있다. 같은 비트를 u8로 읽는다.
            let bytes = std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), bytes.len());
            let capacity = array_len(env, indices)?.min(array_len(env, probabilities)?);
            let mut top = vec![Score::default(); capacity];
            count = classifier.classify_jpeg(bytes, &mut top)?;
            let index: Vec<i32> = top[..count]
                .iter()
                .map(|score| score.index as i32)
                .collect();
            let probability: Vec<f32> =
                top[..count].iter().map(|score| score.probability).collect();
            write_array(env, indices, SET_INT_ARRAY_REGION, &index)?;
            write_array(env, probabilities, SET_FLOAT_ARRAY_REGION, &probability)
        }
    });
    if status == Status::Ok as i32 {
        count as i32
    } else {
        -status
    }
}
// ANCHOR_END: jni_classify

/// `book.badak.Badak.run(long, float[], float[])`. 오류의 종류를 돌려준다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_run(
    env: Env,
    _class: Object,
    handle: i64,
    input: Object,
    output: Object,
) -> i32 {
    guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이들이다.
        unsafe {
            let classifier = borrow(handle)?;
            let input: Vec<f32> = read_array(env, input, GET_FLOAT_ARRAY_REGION)?;
            let mut scores = vec![0.0_f32; classifier.output_len()];
            classifier.run(&input, &mut scores)?;
            write_array(env, output, SET_FLOAT_ARRAY_REGION, &scores)
        }
    })
}

/// `book.badak.Badak.close(long)`.
///
/// # Safety
///
/// JVM만 부른다. 같은 값을 두 번 닫으면 안 된다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_close(_env: Env, _class: Object, handle: i64) {
    // SAFETY: `open`이 준 값을 한 번만 닫는 것은 Java 쪽의 약속이다.
    unsafe { crate::badak_classifier_close(handle as *mut Classifier) };
}

/// `book.badak.Badak.install(String directory, String manifest, String
/// downloaded)`. 오류의 종류를 돌려준다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_install(
    env: Env,
    _class: Object,
    directory: Object,
    manifest: Object,
    downloaded: Object,
) -> i32 {
    use badak_formats::store::{Manifest, Store};
    guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이들이다.
        let (directory, manifest, downloaded) = unsafe {
            (
                read_string(env, directory)?,
                read_string(env, manifest)?,
                read_string(env, downloaded)?,
            )
        };
        Store::new(directory).install(&Manifest::parse(&manifest)?, downloaded)?;
        Ok(())
    })
}

/// `book.badak.Badak.current(String directory, String name)`. 지금 쓰는 판의
/// 모델 파일의 길. 놓인 모델이 없거나 오류가 나면 `null`이다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_current(
    env: Env,
    _class: Object,
    directory: Object,
    name: Object,
) -> Object {
    use badak_formats::store::Store;
    let mut path = None;
    guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이들이다.
        let (directory, name) = unsafe { (read_string(env, directory)?, read_string(env, name)?) };
        path = Store::new(directory).current(&name)?.map(|(_, path)| path);
        Ok(())
    });
    match path {
        // SAFETY: JVM이 준 `env`다.
        Some(path) => unsafe { new_string(env, &path.to_string_lossy()) },
        None => std::ptr::null_mut(),
    }
}

/// `book.badak.Badak.llmOpen(String directory, int weights)`. 모델을 가리키는
/// 주소를 돌려준다. 실패하면 0이다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_llmOpen(
    env: Env,
    _class: Object,
    directory: Object,
    weights: i32,
) -> i64 {
    let mut handle = 0_i64;
    guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이다.
        let directory = unsafe { read_string(env, directory) }?;
        handle = Box::into_raw(Box::new(crate::Llm::open(directory, weights)?)) as i64;
        Ok(())
    });
    handle
}

/// Java 메서드에 넘기는 인자 하나. C의 `jvalue`다. 여기서는 객체만 넘긴다.
#[repr(C)]
union JValue {
    object: Object,
    long: i64,
}

// ANCHOR: jni_callback
/// `book.badak.Badak.llmChat(long, String, String, int, PieceListener)`. 글이
/// 완성되는 대로 `listener.onPiece(byte[])`를 부른다. 쓴 토큰의 수를
/// 돌려주고, 실패하면 오류의 종류에 음수 부호를 붙여 돌려준다.
///
/// 44장의 함수들은 Java가 엔진을 불렀다. 여기서는 엔진이 **Java의 메서드를
/// 거꾸로 부른다.** 메서드를 부르려면 세 가지를 차례로 물어야 한다. 이
/// 객체의 클래스가 무엇인가, 그 클래스에서 이 이름과 이 모양의 메서드는
/// 어느 것인가, 그리고 그것을 부른다.
///
/// # Safety
///
/// JVM만 부른다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_llmChat(
    env: Env,
    _class: Object,
    handle: i64,
    system: Object,
    question: Object,
    max_tokens: i32,
    listener: Object,
) -> i32 {
    type GetClass = unsafe extern "C" fn(Env, Object) -> Object;
    type GetMethod = unsafe extern "C" fn(Env, Object, *const c_char, *const c_char) -> *mut c_void;
    type Call = unsafe extern "C" fn(Env, Object, *mut c_void, *const JValue) -> u8;
    type NewArray = unsafe extern "C" fn(Env, i32) -> Object;
    type Delete = unsafe extern "C" fn(Env, Object);
    type Check = unsafe extern "C" fn(Env) -> u8;

    let mut written = 0;
    let status = guard(|| {
        // SAFETY: JVM이 준 `env`와 손잡이들이다. 칸마다의 타입은 `jni.h`의
        // 선언을 옮긴 것이다.
        unsafe {
            let llm = (handle as *const crate::Llm)
                .as_ref()
                .ok_or_else(|| Error::argument("닫힌 모델입니다"))?;
            let question = read_string(env, question)?;
            // 지시문은 null일 수 있다.
            let system = if system.is_null() {
                None
            } else {
                Some(read_string(env, system)?)
            };
            if listener.is_null() {
                return Err(Error::argument("listener가 null입니다"));
            }
            // "([B)Z"는 메서드의 모양이다. byte[] 하나를 받아 boolean을 낸다.
            let class = function::<GetClass>(env, GET_OBJECT_CLASS)(env, listener);
            let method = function::<GetMethod>(env, GET_METHOD_ID)(
                env,
                class,
                c"onPiece".as_ptr(),
                c"([B)Z".as_ptr(),
            );
            if method.is_null() {
                return Err(Error::argument("listener에 onPiece(byte[])가 없습니다"));
            }
            written = llm.chat(
                system.as_deref(),
                &question,
                max_tokens.max(0) as usize,
                |piece| {
                    // 조각을 Java의 byte[]로 옮긴다. 글(String)로 넘기지 않는
                    // 까닭은 JNI의 글이 UTF-8과 조금 다르기 때문이다(44장).
                    let array = function::<NewArray>(env, NEW_BYTE_ARRAY)(env, piece.len() as i32);
                    if array.is_null() {
                        return false;
                    }
                    function::<RegionSetter<u8>>(env, SET_BYTE_ARRAY_REGION)(
                        env,
                        array,
                        0,
                        piece.len() as i32,
                        piece.as_ptr(),
                    );
                    let arguments = [JValue { object: array }];
                    let go = function::<Call>(env, CALL_BOOLEAN_METHOD_A)(
                        env,
                        listener,
                        method,
                        arguments.as_ptr(),
                    );
                    // 만든 배열의 손잡이를 돌려준다. 돌려주지 않으면 이 함수가
                    // 끝날 때까지 쌓이고, 답이 길면 손잡이의 표가 넘친다.
                    function::<Delete>(env, DELETE_LOCAL_REF)(env, array);
                    // Java 쪽에서 예외가 났으면 그만 쓴다. 예외는 이 함수가
                    // 돌아간 뒤에 Java에서 던져진다.
                    go != 0 && function::<Check>(env, EXCEPTION_CHECK)(env) == 0
                },
            )?;
            function::<Delete>(env, DELETE_LOCAL_REF)(env, class);
            Ok(())
        }
    });
    if status == Status::Ok as i32 {
        written as i32
    } else {
        -status
    }
}
// ANCHOR_END: jni_callback

/// `book.badak.Badak.llmClose(long)`.
///
/// # Safety
///
/// JVM만 부른다. 같은 값을 두 번 닫으면 안 된다.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_book_badak_Badak_llmClose(_env: Env, _class: Object, handle: i64) {
    // SAFETY: `llmOpen`이 준 값을 한 번만 닫는 것은 Java 쪽의 약속이다.
    unsafe { crate::badak_llm_close(handle as *mut crate::Llm) };
}

#[cfg(test)]
pub(crate) mod fake {
    //! 시험에 쓰는 가짜 JVM. 표에서 쓰는 열다섯 칸만 채웠다. 나머지 칸은 비어 있어서,
    //! 적어 두지 않은 칸을 부르면 시험이 그 자리에서 죽는다.

    use super::*;

    /// 조각을 듣는 함수.
    pub type Hear = Box<dyn FnMut(&[u8]) -> bool>;

    /// 가짜 Java 객체.
    pub enum Thing {
        Bytes(Vec<i8>),
        Ints(Vec<i32>),
        Floats(Vec<f32>),
        Text(CString),
        /// `onPiece(byte[])`를 가진 객체. 받은 조각으로 이 함수를 부른다.
        Listener(Hear),
    }

    impl std::fmt::Debug for Thing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                Self::Bytes(_) => "byte[]",
                Self::Ints(_) => "int[]",
                Self::Floats(_) => "float[]",
                Self::Text(_) => "String",
                Self::Listener(_) => "PieceListener",
            })
        }
    }

    thread_local! {
        /// `NewByteArray`가 만들어 아직 돌려받지 못한 배열의 수.
        pub static LIVE_ARRAYS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    impl Thing {
        pub fn handle(&mut self) -> Object {
            std::ptr::from_mut(self).cast()
        }
    }

    unsafe fn thing<'a>(object: Object) -> &'a mut Thing {
        // SAFETY: 시험이 `Thing::handle`로 만든 손잡이만 넘긴다.
        unsafe { &mut *object.cast::<Thing>() }
    }

    unsafe extern "C" fn array_length(_: Env, array: Object) -> i32 {
        // SAFETY: 위와 같다.
        match unsafe { thing(array) } {
            Thing::Bytes(values) => values.len() as i32,
            Thing::Ints(values) => values.len() as i32,
            Thing::Floats(values) => values.len() as i32,
            other => panic!("배열이 아니다: {other:?}"),
        }
    }

    unsafe extern "C" fn object_class(_: Env, object: Object) -> Object {
        // 가짜 클래스는 객체 자신이다.
        object
    }

    unsafe extern "C" fn method_id(
        _: Env,
        _: Object,
        name: *const c_char,
        shape: *const c_char,
    ) -> *mut c_void {
        // SAFETY: 0으로 끝나는 글 둘을 받는다.
        let (name, shape) = unsafe {
            (
                std::ffi::CStr::from_ptr(name),
                std::ffi::CStr::from_ptr(shape),
            )
        };
        if name == c"onPiece" && shape == c"([B)Z" {
            std::ptr::dangling_mut::<u64>().cast()
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn new_bytes(_: Env, length: i32) -> Object {
        LIVE_ARRAYS.with(|live| live.set(live.get() + 1));
        Box::into_raw(Box::new(Thing::Bytes(vec![0; length as usize]))).cast()
    }

    unsafe extern "C" fn set_bytes(_: Env, array: Object, start: i32, len: i32, from: *const u8) {
        // SAFETY: 읽을 자리의 길이는 부른 쪽이 `len`으로 약속했다.
        unsafe {
            let Thing::Bytes(values) = thing(array) else {
                panic!("byte[]가 아니다")
            };
            let source = std::slice::from_raw_parts(from.cast::<i8>(), len as usize);
            values[start as usize..(start + len) as usize].copy_from_slice(source);
        }
    }

    unsafe extern "C" fn call_boolean(
        _: Env,
        object: Object,
        _: *mut c_void,
        arguments: *const JValue,
    ) -> u8 {
        // SAFETY: 인자는 `byte[]` 하나다.
        unsafe {
            let Thing::Bytes(bytes) = thing((*arguments).object) else {
                panic!("byte[]가 아니다")
            };
            let bytes: Vec<u8> = bytes.iter().map(|&byte| byte as u8).collect();
            let Thing::Listener(listener) = thing(object) else {
                panic!("listener가 아니다")
            };
            u8::from(listener(&bytes))
        }
    }

    unsafe extern "C" fn delete_local(_: Env, object: Object) {
        // SAFETY: `new_bytes`가 상자로 만든 배열만 지운다. 클래스(객체 자신)는
        // 시험이 가진 것이므로 두고, 세지도 않는다.
        unsafe {
            if matches!(thing(object), Thing::Bytes(_)) {
                drop(Box::from_raw(object.cast::<Thing>()));
                LIVE_ARRAYS.with(|live| live.set(live.get() - 1));
            }
        }
    }

    unsafe extern "C" fn exception_check(_: Env) -> u8 {
        0
    }

    unsafe extern "C" fn get_bytes(_: Env, array: Object, start: i32, len: i32, out: *mut i8) {
        // SAFETY: 받는 자리의 길이는 부른 쪽이 `len`으로 약속했다.
        unsafe {
            let Thing::Bytes(values) = thing(array) else {
                panic!("byte[]가 아니다")
            };
            let range = start as usize..(start + len) as usize;
            std::slice::from_raw_parts_mut(out, len as usize).copy_from_slice(&values[range]);
        }
    }

    unsafe extern "C" fn get_floats(_: Env, array: Object, start: i32, len: i32, out: *mut f32) {
        // SAFETY: 위와 같다.
        unsafe {
            let Thing::Floats(values) = thing(array) else {
                panic!("float[]가 아니다")
            };
            let range = start as usize..(start + len) as usize;
            std::slice::from_raw_parts_mut(out, len as usize).copy_from_slice(&values[range]);
        }
    }

    unsafe extern "C" fn set_ints(_: Env, array: Object, start: i32, len: i32, from: *const i32) {
        // SAFETY: 위와 같다.
        unsafe {
            let Thing::Ints(values) = thing(array) else {
                panic!("int[]가 아니다")
            };
            let range = start as usize..(start + len) as usize;
            values[range].copy_from_slice(std::slice::from_raw_parts(from, len as usize));
        }
    }

    unsafe extern "C" fn set_floats(_: Env, array: Object, start: i32, len: i32, from: *const f32) {
        // SAFETY: 위와 같다.
        unsafe {
            let Thing::Floats(values) = thing(array) else {
                panic!("float[]가 아니다")
            };
            let range = start as usize..(start + len) as usize;
            values[range].copy_from_slice(std::slice::from_raw_parts(from, len as usize));
        }
    }

    unsafe extern "C" fn new_string_utf(_: Env, text: *const c_char) -> Object {
        // SAFETY: 0으로 끝나는 글을 받는다. 만든 것은 시험이 `take_text`로
        // 되찾아 지운다.
        let text = unsafe { std::ffi::CStr::from_ptr(text) }.to_owned();
        Box::into_raw(Box::new(Thing::Text(text))).cast()
    }

    unsafe extern "C" fn get_string(_: Env, string: Object, _: *mut u8) -> *const c_char {
        // SAFETY: 위와 같다.
        match unsafe { thing(string) } {
            Thing::Text(text) => text.as_ptr(),
            _ => panic!("글이 아니다"),
        }
    }

    unsafe extern "C" fn release_string(_: Env, _: Object, _: *const c_char) {}

    /// 가짜 표.
    pub fn table() -> Box<[*const c_void; SLOTS]> {
        let mut table = Box::new([std::ptr::null::<c_void>(); SLOTS]);
        table[DELETE_LOCAL_REF] = delete_local as *const c_void;
        table[GET_OBJECT_CLASS] = object_class as *const c_void;
        table[GET_METHOD_ID] = method_id as *const c_void;
        table[CALL_BOOLEAN_METHOD_A] = call_boolean as *const c_void;
        table[NEW_BYTE_ARRAY] = new_bytes as *const c_void;
        table[SET_BYTE_ARRAY_REGION] = set_bytes as *const c_void;
        table[EXCEPTION_CHECK] = exception_check as *const c_void;
        table[NEW_STRING_UTF] = new_string_utf as *const c_void;
        table[GET_STRING_UTF_CHARS] = get_string as *const c_void;
        table[RELEASE_STRING_UTF_CHARS] = release_string as *const c_void;
        table[GET_ARRAY_LENGTH] = array_length as *const c_void;
        table[GET_BYTE_ARRAY_REGION] = get_bytes as *const c_void;
        table[GET_FLOAT_ARRAY_REGION] = get_floats as *const c_void;
        table[SET_INT_ARRAY_REGION] = set_ints as *const c_void;
        table[SET_FLOAT_ARRAY_REGION] = set_floats as *const c_void;
        table
    }

    /// `NewStringUTF`가 만든 글을 되찾는다.
    pub fn take_text(object: Object) -> String {
        // SAFETY: `new_string_utf`가 상자로 만든 것이다.
        match *unsafe { Box::from_raw(object.cast::<Thing>()) } {
            Thing::Text(text) => text.into_string().unwrap(),
            other => panic!("글이 아니다: {other:?}"),
        }
    }

    pub fn text(text: &str) -> Thing {
        Thing::Text(CString::new(text).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Thing, table, take_text, text};
    use super::*;
    use badak_graph::models_dir;

    #[test]
    fn classifies_through_the_jni_functions() {
        use std::os::fd::AsRawFd;

        let table = table();
        let mut pointer: *const [*const c_void; SLOTS] = &raw const *table;
        let env: Env = &raw mut pointer;
        let class = std::ptr::null_mut();
        let model =
            std::fs::File::open(models_dir().join("vision_tflite/mobilenetv2.tflite")).unwrap();
        let length = model.metadata().unwrap().len() as i64;
        let jpeg = std::fs::read(models_dir().join("images/cup.jpg")).unwrap();
        let mut jpeg = Thing::Bytes(jpeg.into_iter().map(|byte| byte as i8).collect());
        let mut indices = Thing::Ints(vec![-1; 3]);
        let mut probabilities = Thing::Floats(vec![0.0; 3]);
        // SAFETY: 가짜 표가 채운 칸만 쓰인다.
        unsafe {
            assert_eq!(
                take_text(Java_book_badak_Badak_version(env, class)),
                env!("CARGO_PKG_VERSION")
            );
            assert_eq!(take_text(new_string(env, "한글 a\0b 🙂")), "한글 a b ?");
            // 틀린 구간은 0을 돌려주고 설명을 남긴다.
            assert_eq!(
                Java_book_badak_Badak_open(env, class, model.as_raw_fd(), 0, 100),
                0
            );
            assert!(!take_text(Java_book_badak_Badak_lastError(env, class)).is_empty());

            let handle = Java_book_badak_Badak_open(env, class, model.as_raw_fd(), 0, length);
            assert_ne!(handle, 0);
            let count = Java_book_badak_Badak_classifyJpeg(
                env,
                class,
                handle,
                jpeg.handle(),
                indices.handle(),
                probabilities.handle(),
            );
            assert_eq!(count, 3);
            let (Thing::Ints(index), Thing::Floats(probability)) = (&indices, &probabilities)
            else {
                unreachable!()
            };
            assert_eq!(index[0], 968);
            assert!(probability[0] > 0.9 && probability[2] > 0.0);

            // 숫자로 바꾼 입력을 넣는 길.
            let image = std::fs::read(models_dir().join("images/cup.jpg")).unwrap();
            let image = badak_image::Image::decode_jpeg(&image).unwrap();
            let mut input = Thing::Floats(badak_image::Preprocess::IMAGENET.apply(&image));
            let mut output = Thing::Floats(vec![0.0; 1000]);
            assert_eq!(
                Java_book_badak_Badak_run(env, class, handle, input.handle(), output.handle()),
                0
            );
            let Thing::Floats(scores) = &output else {
                unreachable!()
            };
            let best = (0..1000)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .unwrap();
            assert_eq!(best, 968);
            // 결과를 받을 배열이 짧으면 오류다. 배열 밖에 쓰지 않는다.
            let mut short = Thing::Floats(vec![0.0; 10]);
            let status =
                Java_book_badak_Badak_run(env, class, handle, input.handle(), short.handle());
            assert_eq!(status, Status::InvalidArgument as i32);
            // null과 닫힌 분류기.
            let status =
                Java_book_badak_Badak_classifyJpeg(env, class, handle, class, class, class);
            assert_eq!(status, -(Status::InvalidArgument as i32));
            Java_book_badak_Badak_close(env, class, handle);
            let status = Java_book_badak_Badak_run(env, class, 0, input.handle(), output.handle());
            assert_eq!(status, Status::InvalidArgument as i32);
        }
    }

    #[test]
    fn installs_through_the_jni_functions() {
        use badak_formats::store::Manifest;

        let table = table();
        let mut pointer: *const [*const c_void; SLOTS] = &raw const *table;
        let env: Env = &raw mut pointer;
        let class = std::ptr::null_mut();
        let directory =
            std::env::temp_dir().join(format!("badak_jni_store_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let arrived = directory.join("arrived.bin");
        std::fs::write(&arrived, "모델").unwrap();
        let manifest = Manifest::describe("toy", "1", &arrived).unwrap();
        let mut store = text(directory.join("models").to_str().unwrap());
        let mut name = text("toy");
        let mut manifest = text(&manifest.to_json());
        let mut downloaded = text(arrived.to_str().unwrap());
        // SAFETY: 가짜 표가 채운 칸만 쓰인다.
        unsafe {
            assert!(
                Java_book_badak_Badak_current(env, class, store.handle(), name.handle()).is_null()
            );
            let status = Java_book_badak_Badak_install(
                env,
                class,
                store.handle(),
                manifest.handle(),
                downloaded.handle(),
            );
            assert_eq!(status, 0);
            let path = take_text(Java_book_badak_Badak_current(
                env,
                class,
                store.handle(),
                name.handle(),
            ));
            assert_eq!(std::fs::read_to_string(path).unwrap(), "모델");
            // 같은 파일을 다시 놓으려 하면 받은 파일이 이미 없다.
            let status = Java_book_badak_Badak_install(
                env,
                class,
                store.handle(),
                manifest.handle(),
                downloaded.handle(),
            );
            assert_eq!(status, Status::Io as i32);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn the_engine_calls_back_into_java_for_every_piece() {
        use super::fake::LIVE_ARRAYS;

        let directory = models_dir().join("smollm2_135m");
        if !directory.join("model.safetensors").exists() {
            eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
            return;
        }
        let table = table();
        let mut pointer: *const [*const c_void; SLOTS] = &raw const *table;
        let env: Env = &raw mut pointer;
        let class = std::ptr::null_mut();
        let mut path = text(directory.to_str().unwrap());
        let mut question = text("What is the capital of France?");
        let heard = std::rc::Rc::new(std::cell::RefCell::new(Vec::<u8>::new()));
        let sink = std::rc::Rc::clone(&heard);
        let mut listener = Thing::Listener(Box::new(move |piece: &[u8]| {
            // 조각마다 온전한 UTF-8이어야 한다.
            assert!(std::str::from_utf8(piece).is_ok());
            sink.borrow_mut().extend_from_slice(piece);
            true
        }));
        let mut calls = 0;
        let mut stopper = Thing::Listener(Box::new(move |_: &[u8]| {
            calls += 1;
            calls < 3
        }));
        // SAFETY: 가짜 표가 채운 칸만 쓰인다.
        unsafe {
            assert_eq!(
                Java_book_badak_Badak_llmOpen(env, class, path.handle(), 9),
                0
            );
            let handle = Java_book_badak_Badak_llmOpen(env, class, path.handle(), 0);
            assert_ne!(handle, 0);
            let written = Java_book_badak_Badak_llmChat(
                env,
                class,
                handle,
                class,
                question.handle(),
                24,
                listener.handle(),
            );
            assert!(written > 0 && written <= 24, "{written}");
            let answer = String::from_utf8(heard.borrow().clone()).unwrap();
            assert!(answer.contains("Paris"), "{answer}");
            // 만든 배열은 모두 돌려받았다.
            assert_eq!(LIVE_ARRAYS.with(std::cell::Cell::get), 0);
            // 듣는 쪽이 거짓을 돌려주면 거기서 멈춘다.
            let written = Java_book_badak_Badak_llmChat(
                env,
                class,
                handle,
                class,
                question.handle(),
                24,
                stopper.handle(),
            );
            assert_eq!(written, 3);
            // null과 닫힌 모델.
            let status = Java_book_badak_Badak_llmChat(
                env,
                class,
                handle,
                class,
                question.handle(),
                4,
                class,
            );
            assert_eq!(status, -(Status::InvalidArgument as i32));
            Java_book_badak_Badak_llmClose(env, class, handle);
            let status = Java_book_badak_Badak_llmChat(
                env,
                class,
                0,
                class,
                question.handle(),
                4,
                listener.handle(),
            );
            assert_eq!(status, -(Status::InvalidArgument as i32));
        }
    }
}
