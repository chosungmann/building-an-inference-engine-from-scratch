# 부록 A. 이 책이 쓰는 Rust

이 책은 Rust를 가르치는 책이 아닙니다. 다른 언어로 오래 일한 사람이
코드를 따라 읽을 수 있을 만큼만 Rust를 씁니다. 이 부록은 그 "만큼"이
무엇인지를 모은 것입니다. 엔진의 코드에 나오는 Rust의 기능을 하나씩 들고,
그것이 무엇을 하는지와 이 책의 어디에서 쓰이는지를 적습니다.

Rust를 처음 본다면 이 부록을 먼저 훑고 [4장](../part1/ch04_tensor.md)으로 가면 됩니다. Rust를 아는
독자는 A.7절(`unsafe`를 쓰는 규칙)과 A.13절(쓰지 않는 것)만 보아도
됩니다.

확인한 버전은 Rust 1.100, 에디션 2024입니다.

## A.1 Cargo와 작업 공간

Rust의 빌드 도구는 Cargo입니다. 이 책의 저장소는 **작업
공간**<sup>workspace</sup> 하나이고, 그 안에 크레이트(crate. 다른 언어의
패키지나 라이브러리) 열둘이 있습니다.

| 크레이트 | 하는 일 | 처음 나오는 곳 |
|---|---|---|
| `badak_core` | 텐서, 연산자, 커널, 스레드 풀, 양자화 | [4장](../part1/ch04_tensor.md) |
| `badak_formats` | 모델 파일 읽기(safetensors, GGUF, tflite, ONNX) | [6장](../part1/ch06_model_files.md) |
| `badak_models` | 코드로 쓴 모델들(LLM, ViT, Whisper, 확산 모델 등) | [7장](../part1/ch07_mnist.md) |
| `badak_bench` | 재는 도구와 측정용 예제 | [14장](../part3/ch14_measurement.md) |
| `badak_gpu` | GPU의 선형 층 | [23장](../part3/ch23_gpu.md) |
| `badak_graph` | 그래프 해석기와 최적화 | [26장](../part4/ch26_graph_interpreter.md) |
| `badak_image` | 사진 읽기와 전처리 | [27장](../part4/ch27_convolution.md) |
| `badak_audio` | 소리 읽기와 스펙트로그램 | [33장](../part4/ch33_speech_recognition.md) |
| `badak_server`, `badak_cli` | 추론 서버와 명령 | [39장](../part5/ch39_inference_server.md) |
| `badak_ffi` | 폰의 앱이 부르는 C 함수들 | [44장](../part6/ch44_on_the_phone.md) |
| `badak_web` | 브라우저가 부르는 함수들 | [48장](../part6/ch48_browser.md) |

책에서 되풀이해 쓰는 명령은 넷입니다.

```bash
cargo build --release                  # 모두 빌드한다
cargo test --workspace                 # 시험을 모두 돌린다
cargo run --release -p badak_models --example ch11_generate   # 예제 하나를 돌린다
cargo clippy --workspace --all-targets -- -D warnings          # 린트
```

`--release`를 빼면 최적화 없이 빌드되어 수십 배 느립니다. 속도를 재는
명령에는 늘 붙습니다. `-p`는 크레이트를, `--example`은 그 크레이트의
`examples/` 아래의 프로그램 하나를 고릅니다. 이 책의 실험은 대부분
예제이고, 이름이 장 번호로 시작합니다(`ch11_generate`, `ch52_compile`).

테스트는 `#[test]`가 붙은 함수입니다. `cargo test`가 찾아서 돌립니다. 이
책의 테스트는 실제 모델의 순전파를 돌리기 때문에 테스트용 빌드도
최적화합니다(`Cargo.toml`의 `[profile.test] opt-level = 2`).

**밖에서 가져온 크레이트는 둘뿐입니다.** GPU를 쓰는
`wgpu`([23장](../part3/ch23_gpu.md))와 JPEG를 푸는
`jpeg-decoder`([27장](../part4/ch27_convolution.md))입니다. 둘 다 추론의
계산 밖, 경계에 있습니다. 나머지는 표준 라이브러리만 씁니다.

## A.2 값, 소유권, 빌림

Rust의 값에는 **주인이 하나** 있습니다. 주인이 사라지면 값도 사라집니다.
쓰레기 수집기가 없고 `free`를 부르지도 않습니다.

```rust
let weights: Vec<f32> = vec![0.0; 1024];   // weights가 주인이다
let moved = weights;                        // 주인이 moved로 옮겨 갔다
// 여기서 weights를 쓰면 컴파일되지 않는다
```

값을 넘겨주지 않고 잠깐 보여 주는 것이 **빌림**입니다. `&T`는 읽기만 하는
빌림이고 여럿이 함께 가질 수 있습니다. `&mut T`는 고칠 수 있는 빌림이고
한 번에 하나만 있을 수 있습니다. 컴파일러가 이것을 확인합니다.

이 책의 커널이 받는 인자가 거의 모두 빌림입니다.

```rust
fn linear(x: &[f32], w: &[f32], out: &mut [f32]) { /* ... */ }
```

`&[f32]`는 **슬라이스**입니다. "어딘가에 이어져 있는 `f32` 여러 개"를
가리키는 포인터와 길이입니다. 그것이 `Vec`의 것이든, 파일을 메모리에
비춘 것이든, 아레나의 한 구간이든 커널은 모릅니다. 이 책에서 "값을 베끼지
않는다"는 말은 대개 슬라이스로 넘긴다는 뜻입니다.

**수명**<sup>lifetime</sup>(`'a`)은 빌림이 언제까지 유효한지를
컴파일러에게 알리는 표시입니다. 엔진의 코드에는 드물게 나옵니다.
`'static`은 "프로그램이 끝날 때까지"입니다.

주인이 여럿이어야 할 때는 `Arc<T>`를 씁니다. 가리키는 곳의 수를 세다가
0이 되면 값을 없앱니다. 텐서가 그렇게 만들어져 있습니다.
뷰([4장](../part1/ch04_tensor.md))는 같은 저장소를 가리키는 `Arc`를 하나
더 가진 텐서입니다.

```rust
{{#include ../../../crates/badak_core/src/tensor.rs:handle}}
```

`Arc::get_mut`은 "가리키는 것이 나 하나인가"를 알려 줍니다. [40장](../part5/ch40_throughput.md)의 KV
캐시 페이지가 이것으로 "쓸 때 베끼기"를 합니다.

## A.3 `enum`, `match`, `Option`, `Result`

Rust의 `enum`은 "이것 아니면 저것"이고, 경우마다 다른 값을 담을 수
있습니다. 위의 `Handle`이 그 예입니다. `match`로 경우를 나눕니다. 경우를
하나라도 빠뜨리면 컴파일되지 않습니다.

`Option<T>`는 값이 있거나(`Some(값)`) 없는(`None`) 것입니다. 널 포인터가
없습니다. `Result<T, E>`는 성공(`Ok(값)`)이거나 실패(`Err(오류)`)입니다.
예외가 없습니다. 실패할 수 있는 함수는 `Result`를 돌려줍니다.

```rust
{{#include ../../../crates/badak_core/src/error.rs:error}}
```

물음표는 "실패이면 여기서 돌아간다"입니다.

```rust
let tensor = weights.tensor("fc1.weight")?;   // Err이면 이 함수가 그 Err로 끝난다
```

`let ... else`는 "이 형태가 아니면 else로"입니다. 책의 코드에 자주
나옵니다.

```rust
let &[rows, cols] = &shape[..] else {
    return Err(unsupported("가중치가 행렬이 아닌 선형 층"));
};
```

예제 프로그램의 `main`은 `Result<(), Box<dyn std::error::Error>>`를
돌려줍니다. "어떤 오류든"이라는 뜻입니다. 라이브러리 쪽은 크레이트마다
자기 `Error`를 둡니다.

`unwrap()`과 `expect()`는 "실패이면 프로그램을 멈춘다"입니다. 엔진의
라이브러리 코드에서는 피하고, 테스트와 예제에서 씁니다.

## A.4 트레이트와 제네릭

**트레이트**<sup>trait</sup>는 "이런 함수를 가진 타입"이라는 약속입니다.
다른 언어의 인터페이스에 가깝습니다.

```rust
{{#include ../../../crates/badak_core/src/tensor.rs:element}}
```

텐서의 원소가 될 수 있는 타입을 `Element`로 묶었습니다. `const DTYPE`처럼
트레이트는 함수만이 아니라 상수도 가질 수 있습니다.

**제네릭**은 타입을 인자로 받는 함수와 타입입니다.
`fn to_vec<T: Element>(&self) -> Vec<T>`는 "`Element`인 어떤 `T`에
대해서든"입니다. 컴파일러가 쓰이는 `T`마다 함수를 따로 만듭니다(단형화).
그래서 실행할 때 드는 비용이 없습니다.
[20장](../part3/ch20_weight_quantization.md)의 커널이 이것으로 가중치의
형식(`f32`, BF16, `Q8_0`, `Q4_0`)마다 따로 컴파일됩니다.

**상수 제네릭**은 숫자를 인자로 받습니다.
[17장](../part3/ch17_simd.md)의 내적 커널이 "입력 `M`개와 가중치
`N`행"을 타입에 적어 둡니다.

```rust
pub fn dots<W: Weight, const M: usize, const N: usize>(
    xs: [&[f32]; M],
    ws: [&[W]; N],
) -> [[f32; N]; M]
```

`[f32; 8]`은 길이가 8로 정해진 배열입니다. 길이가 타입의 일부여서
컴파일러가 반복문을 풀고 경계 검사를 없앨 수 있습니다. [52장](../part7/ch52_ml_compilers.md)의 컴파일러가
낸 소스가 이것에 기댑니다.

**`dyn Trait`**은 반대입니다. 타입을 실행할 때 정합니다. 함수 포인터의
표를 거쳐 부릅니다. [26장](../part4/ch26_graph_interpreter.md)의
레지스트리가 연산자마다 커널을 `Box<dyn Kernel>`로 갖고 있고, 해석기가
마디마다 그것을 부릅니다. [52장](../part7/ch52_ml_compilers.md)에서 잰
"마디 하나에 11ns"의 일부가 이 호출입니다.

`impl Trait for Type`은 타입에 트레이트를 구현합니다. `#[derive(Debug,
Clone)]`은 흔한 트레이트의 구현을 컴파일러에게 만들어 달라는 것입니다.

## A.5 반복자와 슬라이스의 도구

반복문은 대개 반복자로 씁니다.

```rust
let dot: f32 = x.iter().zip(w).map(|(a, b)| a * b).sum();
```

`|a, b| ...`는 클로저(이름 없는 함수)입니다. 반복자는 최적화하면 손으로
쓴 반복문과 같은 기계어가 됩니다.

커널에서 자주 쓰는 것들입니다.

| 도구 | 하는 일 | 쓰이는 곳 |
|---|---|---|
| `chunks_exact(n)` | 슬라이스를 `n`개씩 끊는다. 남는 것은 `remainder()` | [16장](../part3/ch16_matmul.md)부터 줄곧 |
| `as_chunks::<N>()` | `&[T]`를 `&[[T; N]]`과 나머지로. 길이가 타입에 들어간다 | 사진의 픽셀을 셋씩 읽을 때 등 |
| `zip`, `enumerate` | 둘을 함께 돈다, 번호를 붙인다 | 어디서나 |
| `split_at_mut(k)` | 고칠 수 있는 슬라이스를 겹치지 않는 둘로 | 스레드에 출력을 나눠 줄 때([18장](../part3/ch18_multithreading.md)) |
| `total_cmp` | `f32`를 NaN까지 순서 지어 견준다 | [12장](../part2/ch12_sampling.md)의 샘플링 |
| `to_bits`, `from_bits` | `f32`와 `u32` 사이를 비트 그대로 | "비트가 같다"를 확인할 때 |

슬라이스의 `x[i]`는 **경계를 검사합니다.** 벗어나면 프로그램이
멈춥니다(패닉). 안쪽 반복문에서 이 검사가 속도를 떨어뜨릴 수 있어서,
커널은 검사가 사라지는 형태로 씁니다. 길이를 한 번 확인해
두거나(`assert!`), 고정 길이 배열로 바꾸거나, 반복자를 씁니다.
[16장](../part3/ch16_matmul.md)에 그 과정이 있습니다.

## A.6 `f32`의 덧셈은 순서를 바꿀 수 없다

Rust의 기능이라기보다 Rust가 지키는 규칙입니다. 부동소수점의 덧셈은
순서를 바꾸면 마지막 비트가 달라질 수 있습니다. Rust의 컴파일러는 **답이
달라지는 최적화를 하지 않습니다.** C 컴파일러의 `-ffast-math` 같은
스위치가 안정판에 없습니다.

그래서 `sum += x[i] * w[i]`라는 반복문은 컴파일러가 스스로 벡터 명령으로
바꾸지 못합니다. 사람이 합을 여러 변수에 나누어 담아 "이 순서로 더해도
된다"고 코드로 알려야 합니다. [16장](../part3/ch16_matmul.md)과
[17장](../part3/ch17_simd.md)이 그 내용이고,
[52장](../part7/ch52_ml_compilers.md)의 커널 찾기에서 합을 하나에만
담았을 때 7배 느렸던 것이 이것입니다.

이 규칙에는 장점도 있습니다. 같은 코드는 어느 기계에서나 같은 비트를
냅니다. 이 책이 "NEON과 AVX2가 같은 비트를 낸다", "폰에서 맥과 같은
비트를 낸다"를 테스트할 수 있는 것이 그 덕입니다.

## A.7 `unsafe`

Rust의 컴파일러는 메모리를 잘못 쓰는 코드를 거절합니다. 컴파일러가
확인할 수 없는 일을 해야 할 때 `unsafe`를 씁니다. "여기는 내가
책임진다"는 표시입니다. `unsafe` 안에서만 할 수 있는 일은 몇 가지입니다.

- 날 포인터(`*const T`, `*mut T`)가 가리키는 곳을 읽고 쓰기
- `unsafe`로 표시된 함수 부르기(C 함수, 프로세서의 명령 등)
- 가변 전역 변수 쓰기

엔진의 어디에서 쓰는지입니다.

| 쓰는 곳 | 까닭 | 장 |
|---|---|---|
| 벡터 명령으로 값을 읽고 쓰기 | 내장 함수가 포인터를 받는다 | 17, 20, 28, 29, 31 |
| 파일을 메모리에 비추기 | 운영체제의 `mmap`은 C 함수다 | 6, 22 |
| 바이트를 `f32`의 슬라이스로 보기 | 정렬과 길이를 사람이 보증한다 | 4, 6 |
| 스레드 풀이 빌린 클로저를 넘기기 | 수명을 지운 포인터로 건넨다 | 18 |
| 여러 스레드가 출력의 다른 구간을 쓰기 | 겹치지 않음을 사람이 보증한다 | 18, 28 |
| 인라인 어셈블리(SME) | 프로세서의 상태를 직접 바꾼다 | 31 |
| C, JNI, 브라우저와 만나는 문 | 밖에서 온 포인터를 믿어야 한다 | 44, 47, 48, 49 |

이 책은 `unsafe`에 규칙 둘을 둡니다.

**하나.** `unsafe` 블록마다 바로 위에 `// SAFETY:`로 시작하는 주석을
적습니다. "왜 이것이 안전한가"를 적습니다. 엔진에 그런 주석이 200개
넘게 있습니다.

```rust
// SAFETY: `offset + 4 <= full <= len`이고 모든 벡터의 길이가
// `len`임을 위에서 확인했다. 읽는 네 값은 슬라이스 안에 있다.
let x = unsafe { vld1q_f32(xs.as_ptr().add(offset)) };
```

**둘.** `unsafe fn` 안에서도 위험한 일은 `unsafe { }`로 다시 감쌉니다
(`Cargo.toml`의 `unsafe_op_in_unsafe_fn = "deny"`). 어느 줄이 위험한지가
보이게 하려는 것입니다.

`unsafe`가 틀렸는지 찾아 주는 도구가 Miri입니다. Rust 코드를 해석해
실행하며 잘못된 메모리 접근을 잡습니다.
[22장](../part3/ch22_memory.md)의 아레나와
[26장](../part4/ch26_graph_interpreter.md)의 해석기를 그것으로 돌려
봤습니다. 어셈블리는 따라가지
못합니다([31장](../part4/ch31_mobile_engines.md)의 SME 테스트는 Miri에서
빼 두었습니다).

## A.8 벡터 명령: `std::arch`

프로세서의 벡터 명령은 `std::arch`의 **내장 함수**<sup>intrinsic</sup>로
씁니다. 함수 하나가 명령 하나입니다.

```rust
use std::arch::aarch64::*;          // NEON (ARM)
use std::arch::x86_64::*;           // AVX2 (x86-64)
use std::arch::wasm32::*;           // WebAssembly SIMD (48장)
```

```rust
{{#include ../../../crates/badak_core/src/simd/avx2.rs:dots}}
```

눈여겨볼 것이 둘입니다.

**`#[target_feature(enable = "...")]`.** "이 함수 안에서는 이 명령들을 써도
된다"고 컴파일러에게 알립니다. 이 표시가 없는 함수에서 부르려면
`unsafe`가 필요합니다. 그 명령이 없는 프로세서에서 부르면 프로그램이
죽기 때문입니다.

**실행할 때 고르기.** x86-64에서는 프로세서마다 가진 명령이 달라서
실행할 때 확인합니다. ARM의 NEON은 64비트 ARM에 늘 있어서 확인하지
않습니다.

```rust
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    // AVX2 커널을 쓴다
}
```

`#[cfg(target_arch = "aarch64")]`는 **컴파일할 때** 고르는 것입니다. 그
줄이 붙은 코드는 다른 프로세서용 빌드에 아예 들어가지 않습니다. [17장](../part3/ch17_simd.md)의
`Backend::detect`가 둘을 함께 씁니다.

## A.9 인라인 어셈블리

`std::arch`에 내장 함수가 없는 명령은 `asm!`으로 어셈블리를 직접 적습니다.
이 책에서는 한 곳, [31장](../part4/ch31_mobile_engines.md)의 SME 커널입니다. Rust 1.100의 표준 라이브러리에
SME의 내장 함수가 없기 때문입니다.

`asm!`은 컴파일러에게 **무엇을 망가뜨리는지** 알려야 합니다. 쓰는
레지스터를 빠짐없이 적지 않으면 컴파일러가 거기에 둔 값이 조용히
깨집니다. [31장](../part4/ch31_mobile_engines.md)에서 스트리밍 모드에 드나들 때 벡터 레지스터가 모두
지워지는 것을 그렇게 다뤘습니다.

## A.10 C와 만나는 곳

**C 함수를 부른다.** 선언하고 부릅니다. 에디션 2024에서는 선언 블록에도
`unsafe`를 적습니다.

```rust
{{#include ../../../crates/badak_formats/src/mmap.rs:mmap_ffi}}
```

`libc` 크레이트를 쓰면 이 선언을 대신해 주지만, 표준 라이브러리만 쓴다는
원칙 때문에 직접 적었습니다. 측정에 쓰는 macOS의 함수들([14장](../part3/ch14_measurement.md))도 그렇게
불렀습니다.

**C가 우리를 부른다.** [44장](../part6/ch44_on_the_phone.md)에서 폰의 앱이 엔진을 부르는 문입니다.

```rust
/// 엔진의 판. 0으로 끝나는 글이고, 지우지 않는다.
#[unsafe(no_mangle)]
pub extern "C" fn badak_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}
```

`extern "C"`는 C의 호출 규약을, `no_mangle`은 이름을 그대로 내보내라는
뜻입니다. 이 문에서 지켜야 할 것(패닉이 경계를 넘지 않게 하기, 포인터의
주인을 정하기)이 [44장](../part6/ch44_on_the_phone.md)의 절반입니다.

`#[repr(C)]`는 구조체의 필드를 C와 같은 순서로 놓게 합니다. C에 넘기는
구조체에 붙입니다. `#[repr(align(128))]`은 정렬을 정합니다. [18장](../part3/ch18_multithreading.md)에서 두
스레드의 값이 같은 캐시 줄에 놓이지 않게 하는 데 썼습니다.

## A.11 스레드

표준 라이브러리의 도구만 씁니다.

| 도구 | 하는 일 | 쓰이는 곳 |
|---|---|---|
| `std::thread::spawn` | 스레드를 만든다 | [18장](../part3/ch18_multithreading.md)의 풀, [39장](../part5/ch39_inference_server.md)의 서버 |
| `std::thread::scope` | 범위가 끝날 때 모두 합류하는 스레드. 지역 변수를 빌릴 수 있다 | 15, [18장](../part3/ch18_multithreading.md)의 실험 |
| `Arc<T>` | 여러 스레드가 함께 쥐는 값 | 어디서나 |
| `Mutex<T>`, `Condvar` | 잠금과, 조건이 될 때까지 재우기 | [18장](../part3/ch18_multithreading.md)의 풀, [40장](../part5/ch40_throughput.md)의 스케줄러 |
| `AtomicUsize` 등 | 잠금 없이 읽고 쓰는 정수 | [18장](../part3/ch18_multithreading.md)(일 나눠 갖기) |
| `OnceLock<T>` | 한 번만 만들어 두는 값 | 전역 스레드 풀, [31장](../part4/ch31_mobile_engines.md)의 늘어놓은 필터 |
| `thread_local!` | 스레드마다 따로 있는 값 | 스레드마다의 임시 버퍼, 마지막 오류([44장](../part6/ch44_on_the_phone.md)) |
| `std::sync::mpsc` | 스레드 사이의 통로 | [40장](../part5/ch40_throughput.md)의 스케줄러 |

Rust는 **데이터 경쟁을 컴파일할 때 막습니다.** 두 스레드가 같은 값을
잠금 없이 고치는 코드는 컴파일되지 않습니다. `Send`와 `Sync`라는 두
트레이트가 그것을 표시합니다. `Send`는 다른 스레드로 넘겨도 되는
타입이고, `Sync`는 여러 스레드가 함께 읽어도 되는 타입입니다. 대부분
컴파일러가 알아서 붙입니다.

[18장](../part3/ch18_multithreading.md)의 스레드 풀은 이 규칙을 한
곳에서 우회합니다. 빌린 클로저를 일꾼에게 넘기려고 수명을 지운 포인터를
쓰고, "풀의 `run`이 돌아오기 전까지만 쓴다"는 것을 사람이 보증합니다.

```rust
{{#include ../../../crates/badak_core/src/threads.rs:job}}
```

`rayon` 같은 크레이트가 이 일을 해 주지만 쓰지 않았습니다. 풀을 직접
만드는 것이 [18장](../part3/ch18_multithreading.md)의 내용이었습니다.

## A.12 조건부 컴파일

한 소스로 맥, 리눅스, 폰, 브라우저용을 모두 만듭니다. `cfg`가 가릅니다.

```rust
#[cfg(target_arch = "aarch64")]      // 64비트 ARM일 때만
#[cfg(target_os = "macos")]          // macOS일 때만
#[cfg(target_arch = "wasm32")]       // WebAssembly일 때만
#[cfg(test)]                         // 시험을 빌드할 때만
#[cfg(not(miri))]                    // Miri가 아닐 때만
```

`Cargo.toml`에서도 같은 표기로 의존성을 가릅니다. `badak_gpu`는
브라우저용 빌드에서 `wgpu`를 빼고 브라우저의 WebGPU를 직접
부릅니다([49장](../part6/ch49_browser_gpu.md)).

다른 기계용으로 빌드할 때는 타깃을 줍니다.

```bash
rustup target add aarch64-linux-android wasm32-wasip1-threads
cargo build --release -p badak_ffi --target aarch64-linux-android
```

## A.13 이 책이 쓰지 않는 것

Rust에서 널리 쓰이지만 엔진에는 없는 것들입니다. 다른 Rust 코드를 읽다가
만나면 이 책에서 배우지 않은 것입니다.

- **`async`와 `await`.** 서버([39장](../part5/ch39_inference_server.md))도 스레드와 막히는 소켓으로 썼습니다.
- **외부 크레이트.** `serde`(직렬화), `rayon`(스레드), `ndarray`(텐서),
  `libc`, `tokio`, `anyhow`. JSON 읽기도 직접 썼습니다([6장](../part1/ch06_model_files.md)).
- **절차적 매크로.** `macro_rules!`로 만든 작은 매크로가 몇 개
  있을 뿐입니다.
- **안정판에 없는 기능.** `std::simd`(프로세서와 무관한 벡터 타입),
  `f16` 타입, SME의 내장 함수. 16비트 실수는 타입을 직접 만들었습니다
  ([19장](../part3/ch19_quantization_theory.md)).
- **복잡한 수명 표시.** 엔진의 구조체는 대개 값을 소유하거나 `Arc`로
  쥡니다.

## A.14 더 배우려면

이 부록은 책의 코드를 읽는 데 필요한 만큼만 적었습니다. Rust를 제대로
배우려면 공식 자료가 가장 좋습니다. 아래 셋은 이름만 적어 둡니다. 저는
이 부록을 쓰며 이 자료들을 다시 열어 대조하지 않았습니다.

- *The Rust Programming Language* (<https://doc.rust-lang.org/book/>).
  소유권, 트레이트, 오류 처리. 이 부록의 A.2~A.5에 해당합니다.
- *The Rustonomicon* (<https://doc.rust-lang.org/nomicon/>). `unsafe`를
  쓸 때 지켜야 하는 것들.
- 표준 라이브러리 문서의 `std::arch`
  (<https://doc.rust-lang.org/std/arch/>). 내장 함수의 목록과 실행 시
  확인하는 방법.
