# 부록 D. x86-64와 Linux에서 따라 하기

이 책의 숫자는 모두 Apple Silicon 맥(M4 Max) 한 대에서 쟀습니다. 코드는
Linux와 x86-64에서도 돌게 썼습니다. 이 부록은 다른 기계에서 따라 할 때
**무엇이 그대로이고 무엇이 달라지는지**를 적습니다.

먼저 정직하게 적어 둡니다. **저자에게는 x86 기계가 없습니다.** 아래의
내용은 둘로 나뉩니다.

- **확인한 것.** GitHub의 Linux 러너(x86-64)에서 책의 테스트와 대조가
  모두 통과합니다. 커밋마다 돌립니다.
- **확인하지 않은 것.** x86에서의 속도, Linux에서의 GPU, Windows입니다.
  한 번도 재거나 돌려 보지 않았습니다. 그런 곳에는 그렇다고 적었습니다.

## D.1 확인되어 있는 것

저장소의 CI(`.github/workflows/ci.yml`)가 `ubuntu-latest`(x86-64)에서
다음을 합니다.

1. `cargo fmt --check`, `cargo clippy -- -D warnings`
2. 폰용 타깃(`aarch64-linux-android`, `aarch64-apple-ios`)으로 `cargo check`
3. 시험에 필요한 모델을 받는다(`scripts/download_*.sh`)
4. `cargo test --workspace`
5. 장마다의 대조 예제를 돌린다. [7장](../part1/ch07_mnist.md)(PyTorch), 26~30장(LiteRT, ONNX
   Runtime), 32~37장(PyTorch, transformers), 39~41장(서버, 묶음, 생성
   제어), [48장](../part6/ch48_browser.md)(WebAssembly 빌드)

그래서 **답이 맞다**는 것은 x86-64의 Linux에서도 확인되어 있습니다. [17장](../part3/ch17_simd.md)의
AVX2 커널은 NEON 커널과 **같은 비트**를 냅니다. CI가 두 커널의 결과의
지문을 로그에 남기고, 둘이 같습니다.

Llama 3.2가 필요한 테스트는 CI에서 건너뜁니다(승인이 필요한 모델이어서
받지 않습니다). SmolLM2-135M으로 같은 코드를 테스트합니다.

## D.2 준비

Rust는 맥과 같습니다. [rustup](https://rustup.rs)으로 최신 안정판을
깝니다. 그 밖에 필요한 것은 장에 따라 다릅니다.

| 도구 | 어디에 쓰나 |
|---|---|
| `curl`, `shasum`, `tar`, `gunzip`, `unzip` | 모델을 받는 스크립트. `shasum`은 Perl과 함께 오는 도구이고 대부분의 배포판에 있다 |
| C++ 컴파일러, CMake | 비교 엔진을 빌드할 때([부록 B](b_engines.md)) |
| Python 3 | 정답을 다시 만들거나 Python 엔진과 견줄 때 |
| Node.js | 48~50장(브라우저용 빌드를 Node.js로 먼저 시험한다) |

처음 해 볼 것은 이것입니다.

```bash
cargo test --workspace                       # 모델이 없는 시험은 "건너뜀"을 적고 지나간다
sh scripts/download_mnist.sh && sh scripts/download_ch07.sh
cargo run --release -p badak_models --example ch07_verify
cargo run --release -p badak_bench --example ch17_backend     # 어느 커널이 골라졌는지 적는다
```

마지막 명령이 `Avx2`를 적으면 [17장](../part3/ch17_simd.md)의 커널이 쓰이는 것입니다. `Portable`을
적으면 그 프로세서에 AVX2나 FMA가 없는 것이고, 벡터 명령 없이 돕니다.
느리지만 답은 같습니다.

## D.3 달라지는 것

| | 맥 (책의 기준) | x86-64, Linux |
|---|---|---|
| 벡터 커널 | NEON. 레지스터 하나에 `f32` 넷 | AVX2와 FMA. 레지스터 하나에 여덟. 둘 중 하나라도 없으면 벡터 명령 없이 |
| "빠른 길"([31장](../part4/ch31_mobile_engines.md)부터) | SME 커널 | **없다.** SME는 ARM의 명령이다 |
| 16비트 실수 | 한 명령으로 넷씩 바꾼다 | 하나씩 바꾼다(17.6절, [20장](../part3/ch20_weight_quantization.md)) |
| 캐시 줄 | 128바이트 | 대개 64바이트 |
| 메모리의 페이지 | 16KB | 대개 4KB |
| 코어 | 성능 코어 12개와 효율 코어 4개 | 프로세서마다 다르다. 하이퍼스레딩이 있으면 논리 코어가 물리 코어의 두 배 |
| GPU ([23장](../part3/ch23_gpu.md)) | `wgpu`가 Metal을 쓴다 | `wgpu`가 Vulkan을 쓴다. **돌려 보지 않았다** |
| 하드웨어 카운터 ([14장](../part3/ch14_measurement.md)) | 명령과 사이클의 수를 읽는다 | 읽지 않는다(`None`). `perf stat`을 쓴다 |
| Core ML, 운영체제의 모델, iOS 앱 | 45, 47, [44장](../part6/ch44_on_the_phone.md) | 없다. 예제는 안내만 적고 끝난다 |

하나씩 봅니다.

### 벡터 커널

[17장](../part3/ch17_simd.md)의 커널은 실행할 때 프로세서를 확인해 고릅니다. AVX2와 FMA가 둘 다
있어야 AVX2 커널을 씁니다. 2013년 무렵 이후의 x86 프로세서는 대부분
둘 다 있습니다.

**AVX-512는 쓰지 않습니다.** 그 명령이 있는 프로세서에서도 AVX2 커널이
돕니다. 실전 엔진은 AVX-512와 그 뒤에 나온 명령(VNNI 등)을 씁니다.
x86에서 견주면 그 차이가 격차의 한 갈래가 될 것입니다. 재 보지
않았습니다.

[20장](../part3/ch20_weight_quantization.md)의 양자화 커널(`Q8_0`,
`Q4_0`)에도 AVX2 판이 있고 NEON 판과 같은 비트를 냅니다.
[28장](../part4/ch28_conv_kernels.md)의 합성곱은
[17장](../part3/ch17_simd.md)의 내적 커널로 만들었으므로 AVX2로 돕니다.
**[29장](../part4/ch29_integer_inference.md)과
[31장](../part4/ch31_mobile_engines.md)의 정수 커널은 NEON 판뿐입니다.**
다른 프로세서에서는 벡터 명령 없이 값을 하나씩 계산합니다. 답은
같고(비트까지) 느립니다.

속도를 다듬은 것은 어디서나 NEON 쪽입니다. x86 쪽은 **맞게 돌게만**
만들었고, 합을 몇 개로 나눌지나 행을 몇 개씩 읽을지 같은 것을 x86에서 재
가며 고르지 않았습니다. [52장](../part7/ch52_ml_compilers.md)의 실험이 보여 준 대로 그런 선택은 기계마다
다릅니다.

### SME가 없다

[31장](../part4/ch31_mobile_engines.md)부터의 "빠른 길"은 SME 커널을 씁니다. SME는 Arm의 명령이고, 우리
코드는 macOS에서만 그것이 있는지 확인합니다. **다른 곳에서는 빠른 길이
보통의 커널로 돕니다.** 그래서

- 31~37장의 "빠른 길"과 "기본"의 차이가 사라집니다.
- [53장](../part7/ch53_scorecard.md)의 성적표에서 4부의 줄들은 이 맥의 SME에 기댄 숫자입니다. x86에서는
  NEON 줄에 해당하는 숫자가 나옵니다.
- 상대 엔진들도 달라집니다. [31장](../part4/ch31_mobile_engines.md)에서 SME로 계산하던 XNNPACK, ONNX
  Runtime, MNN은 x86에서 AVX2나 AVX-512 커널을 씁니다.

Linux를 돌리는 ARM 기계에 SME가 있더라도 우리 엔진은 그것을 찾지
못합니다(확인하는 코드가 macOS의 `sysctl`뿐입니다). [31장](../part4/ch31_mobile_engines.md)의 남긴 것입니다.

### 스레드의 수

책의 벤치마크는 대개 "스레드 12개"로 잽니다. 이 맥의 성능 코어가
12개이기 때문입니다. 자기 기계에서는 **물리 코어의 수**로 바꿔야 합니다.
하이퍼스레딩으로 생긴 논리 코어까지 쓰면 행렬 곱은 대개 빨라지지
않습니다. 같은 코어의 벡터 장치를 둘이 나눠 쓰기 때문입니다. 이것도 재
보지 않은 일반론입니다.

성능 코어와 효율 코어가 섞인 프로세서(인텔의 최근 것들)에서는
[18장](../part3/ch18_multithreading.md)에서 본 문제가 같은 형태로
나타납니다. 일을 똑같이 나누면 느린 코어가 전체를 늦춥니다.
[18장](../part3/ch18_multithreading.md)의 풀은 일을 잘게 나눠 먼저 끝난
스레드가 더 가져가게 했으므로 그대로 돕니다.

[18장](../part3/ch18_multithreading.md)에서 구조체를 128바이트 경계에 맞춘 것은 이 맥의 캐시 줄이
128바이트이기 때문입니다. 64바이트인 기계에서도 문제는 없습니다. 조금
더 넓게 띄운 것일 뿐입니다.

### 재는 도구

[14장](../part3/ch14_measurement.md)과 [22장](../part3/ch22_memory.md)의 도구 가운데 몇 가지가 macOS의 것입니다.

| 책에서 쓴 것 (macOS) | Linux에서 |
|---|---|
| `/usr/bin/time -l` (가장 많이 쥔 메모리, 풋프린트) | `/usr/bin/time -v`. "Maximum resident set size"가 나온다. [22장](../part3/ch22_memory.md)의 "풋프린트"에 해당하는 값은 없다 |
| `sample` ([14장](../part3/ch14_measurement.md)의 프로파일러) | `perf record`, `perf report` |
| 명령과 사이클의 수 (`proc_pid_rusage`) | `perf stat -e instructions,cycles` |
| `taskpolicy -b` (효율 코어에서 돌리기) | `taskset`으로 코어를 고른다 |
| `sysctl hw.…` (코어의 수, 캐시 줄) | `lscpu`, `getconf LEVEL1_DCACHE_LINESIZE` |
| `footprint`, `vm_stat` | `/proc/<pid>/status`, `/proc/meminfo` |

오른쪽 열은 흔히 쓰이는 대응을 적은 것이고, 이 책의 스크립트를 그것으로
바꿔 돌려 보지는 않았습니다.

`badak_bench`의 측정 코드는 Linux에서도 돕니다. 시간은 재고, 카운터의
칸은 비워 둡니다.

다음 스크립트는 macOS의 명령에 기대므로 Linux에서 그대로 돌지 않습니다.
`measure_ch20_memory.sh`, `measure_ch22_memory.sh`,
`measure_ch23_memory.sh`, `measure_ch25_memory.sh`, `profile_sample.sh`,
`bench_ch43.sh`, `bench_ch44.sh`, `bench_ch47.sh`, `bench_ch51.sh`,
`build_coreml.sh`. 나머지 벤치마크 스크립트는 POSIX 셸과 Cargo만 씁니다.

### 메모리

[22장](../part3/ch22_memory.md)의 메모리 매핑은 Linux에서 같은 코드로 돕니다. `mmap`에 넘기는
상수의 값이 두 운영체제에서 같습니다. 페이지의 크기가 달라도
됩니다(매핑을 64KB의 경계에 맞춰 시작합니다).

다만 [22장](../part3/ch22_memory.md)의 **숫자**는 macOS의 것입니다. "지켜야 하는 메모리"(풋프린트)는
macOS가 세는 방식의 이름입니다. Linux에서는 상주 메모리(RSS)를 보게 되고,
비춘 파일의 페이지가 거기에 들어갑니다. 같은 프로그램이 더 많은 메모리를
쓰는 것처럼 보일 수 있습니다. 운영체제가 그 페이지를 언제든 버릴 수 있다는
사실은 같습니다.

유닉스가 아닌 운영체제에서는 메모리 매핑 대신 파일을 통째로
읽습니다(`mmap.rs`의 `cfg(not(unix))`). 동작은 같고 시작이 느립니다.

### GPU

[23장](../part3/ch23_gpu.md)의 GPU 코드는 `wgpu`를 거칩니다. `wgpu`는 맥에서 Metal을, Linux에서
Vulkan을 씁니다. 우리가 쓴 셰이더(WGSL)는 그대로입니다. **Linux에서
돌려 보지 않았습니다.** GPU가 없거나 드라이버가 없으면 GPU를 여는 함수가
`None`을 돌려주고, 예제는 그 사실을 적고 끝납니다.

[23장](../part3/ch23_gpu.md)의 숫자는 "통합 메모리"를 가진 기계의 것입니다. CPU와 GPU가 같은
메모리를 씁니다. 따로 달린 그래픽 카드에서는 가중치를 GPU의 메모리로
**옮겨야** 하고, [23장](../part3/ch23_gpu.md)이 잰 "넘기는 값"이 달라집니다. [15장](../part3/ch15_bottleneck.md)의 계산(메모리
대역폭이 토큰 생성의 상한을 정한다)도 그래픽 카드의 메모리 대역폭으로
다시 해야 합니다. [42장](../part5/ch42_datacenter.md)이 제원 표로 그 계산을 해 보았습니다.

### 폰과 브라우저

- **Android용 빌드는 Linux에서도 될 것입니다.** NDK가 Linux용으로도
  나오고, [44장](../part6/ch44_on_the_phone.md)의 스크립트는 SDK의
  위치를 `ANDROID_HOME`으로 받습니다. Linux에서 돌려 보지는 않았습니다.
- **iOS용 빌드는 맥이 있어야 합니다.**
- **브라우저(48~50장).** WebAssembly 빌드는 어느 기계에서나 같은 파일이
  나옵니다. 브라우저에서 도는 것도 같습니다. 달라지는 것은 속도와
  WebGPU의 사정입니다. Linux의 브라우저에서 WebGPU가 켜져 있는지는
  브라우저와 드라이버에 따라 다르고, 확인하지 않았습니다.
  [48장](../part6/ch48_browser.md)의 `can_run.mjs`와 기능 확인 페이지가
  무엇이 되는지를 알려 줍니다.

### 비교 엔진

[부록 B](b_engines.md)의 방법은 Linux에서도 같습니다. 달라지는 것만 적습니다.

- **llama.cpp.** 기본 빌드가 Metal과 Accelerate를 쓰지 않습니다(없으니까요).
  GPU를 쓰려면 CUDA나 Vulkan을 켜는 CMake 옵션을 따로 줘야 합니다.
  `--plain` 빌드는 그대로입니다. [13장](../part2/ch13_first_showdown.md)과 [25장](../part3/ch25_rematch.md)의 "Accelerate"와 "Metal"의
  줄은 다른 것으로 바뀝니다.
- **Python 엔진들.** 같은 판의 패키지가 x86-64 Linux용으로도 있는지는
  확인하지 않았습니다. 대부분 있을 것입니다.
- **PyTorch([51장](../part7/ch51_why_not_pytorch.md)).** 이 맥의 PyTorch는 Accelerate로 행렬 곱을 했습니다.
  x86 Linux의 PyTorch는 다른 라이브러리(MKL이나 OpenBLAS)를 씁니다.
  [51장](../part7/ch51_why_not_pytorch.md)의 "프롬프트 처리는 같다"는 결과가 그대로일지는 재 봐야 압니다.

## D.4 숫자를 어떻게 읽을까

다른 기계에서 책의 벤치마크를 돌리면 숫자가 다릅니다. 무엇이 같고 무엇이
다를 것으로 보는지를 적습니다. **모두 재 보지 않은 예상입니다.**

같을 것으로 보는 것.

- **답.** 같은 비트이거나 장에 적힌 오차 안입니다. 이것은 확인되어
  있습니다.
- **경향.** 프롬프트 처리는 계산이, 토큰 생성은 메모리 대역폭이
  정한다([15장](../part3/ch15_bottleneck.md)). 양자화가 토큰 생성을
  빠르게 한다([20장](../part3/ch20_weight_quantization.md)). 스레드를
  늘리면 어느 지점에서 꺾인다([18장](../part3/ch18_multithreading.md)).
  묶으면 처리량이 늘고 지연이 는다([40장](../part5/ch40_throughput.md)).
- **[51장](../part7/ch51_why_not_pytorch.md)의 결론.** 연산을 부르는
  비용은 프로세서가 아니라 프레임워크의 구조에서 옵니다.

다를 것으로 보는 것.

- **절댓값 전부.**
- **엔진 사이의 순위.** [31장](../part4/ch31_mobile_engines.md)의 순위는
  SME가 갈랐습니다. x86에서는 AVX-512가 그 역할을 할 것이고, 우리 엔진은
  그것을 쓰지 않습니다.
- **스레드의 가장 좋은 수.**
- **GPU와 CPU의 비.** 통합 메모리가 아니면 크게 달라집니다.
- **[52장](../part7/ch52_ml_compilers.md)의 "가장 빠른 형태".** 기계마다
  다르다는 것이 그 실험의 요점이었습니다.

자기 기계의 숫자를 얻었다면 그것이 이 책에 없는 자료입니다. 측정 예제는
결과와 함께 기계의 정보(프로세서, 코어의 수, 운영체제)를 적도록 되어
있습니다. macOS가 아닌 곳에서는 그 정보의 일부가 비어 있으니 손으로 적어
둬야 합니다.

## D.5 Windows

돌려 보지 않았습니다. 엔진의 코드는 표준 라이브러리만 쓰므로 빌드는 될
것으로 보지만 확인하지 않았습니다. 스크립트는 POSIX 셸을 가정합니다.
Windows에서는 WSL 안에서 Linux처럼 따라 하는 것이 가장 덜 고생스러울
것입니다.
