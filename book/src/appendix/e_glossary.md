# 부록 E. 용어집

이 책이 쓰는 낱말을 한국어와 영어로 나란히 적습니다. 영어 자료와 소스
코드로 넘어갈 때 찾아보는 표입니다.

## E.1 낱말을 고른 원칙

이 책은 낱말을 두 가지로 적었습니다.

**굳은 낱말은 그대로 씁니다.** 텐서, 커널, 토큰, 어텐션, 캐시, 스레드처럼
한국어 개발 문서에서 이미 굳은 말은 소리 나는 대로 적었습니다. 처음 나올
때 영어를 괄호에 넣었습니다.

**풀어 쓸 수 있으면 풀어 썼습니다.** 뜻이 바로 떠오르는 한국어가 있으면
그것을 썼습니다. 그래서 이 책에는 다른 자료에서 보기 어려운 낱말이
있습니다. 영어 자료를 찾을 때는 오른쪽의 말로 찾아야 합니다.

| 이 책의 말 | 흔히 쓰는 말 | 영어 |
|---|---|---|
| 마디 | 노드 | node |
| 묶음 | 배치 | batch |
| 가림막 | 마스크 | mask |
| 덧대기 | 패딩 | padding |
| 보폭 | 스트라이드 (합성곱에서) | stride |
| 눈금 | 스케일 (양자화에서) | scale |
| 치우침 | 편향, 바이어스 | bias |
| 형상 | 셰이프, 모양 | shape |
| 씨앗 | 시드 | seed |
| 발자국, 지켜야 하는 메모리 | 메모리 풋프린트 | memory footprint |
| 비추다 (파일을 메모리에) | 메모리 매핑 | memory mapping |
| 낮추기 | 로어링 | lowering |
| 프롬프트 처리, 읽기 | 프리필 | prefill |
| 토큰 생성, 쓰기 | 디코드 | decode |
| 걸음 | 스텝 | step |
| 판 | 버전, 리비전 | version, revision |
| 꾸러미 | 패키지 | package |
| 문턱 | 임계값 | threshold |

"스트라이드"는 두 곳에서 다른 뜻으로 나옵니다. 텐서에서는 한 칸 갈 때
메모리에서 건너뛰는 원소의 수이고(2, [4장](../part1/ch04_tensor.md)) 그대로 "스트라이드"라고
적었습니다. 합성곱에서는 필터를 옮기는 칸의 수이고([27장](../part4/ch27_convolution.md)) "보폭"이라고
적었습니다. 영어로는 둘 다 stride입니다.

## E.2 가나다순

"처음 나오는 곳"은 그 낱말을 설명하는 장입니다.

{{#include generated/glossary.md}}

## E.3 영어에서 한국어로

영어 자료에서 본 낱말을 이 책에서 찾을 때 씁니다.

{{#include generated/glossary_english.md}}

## E.4 이름과 줄임말

낱말이 아니라 이름인 것들입니다. 번역하지 않고 그대로 적었습니다.

| 이름 | 무엇 | 처음 나오는 곳 |
|---|---|---|
| NEON | 64비트 ARM의 벡터 명령 | [17장](../part3/ch17_simd.md) |
| AVX2, FMA | x86-64의 벡터 명령 | [17장](../part3/ch17_simd.md) |
| SME | ARM의 행렬 명령. 이 책의 "빠른 길"이 쓴다 | [31장](../part4/ch31_mobile_engines.md) |
| SIMD | 명령 하나로 값 여럿을 계산하는 것 | [14장](../part3/ch14_measurement.md) |
| BF16, F16 | 16비트 실수의 두 형식 | [19장](../part3/ch19_quantization_theory.md) |
| `Q8_0`, `Q4_0`, `Q4_K_M` | llama.cpp의 양자화 형식들 | [20장](../part3/ch20_weight_quantization.md) |
| GGUF | llama.cpp의 모델 파일 형식 | [6장](../part1/ch06_model_files.md) |
| safetensors | Hugging Face의 가중치 파일 형식 | [6장](../part1/ch06_model_files.md) |
| `.tflite`, FlatBuffers | LiteRT의 모델 파일 형식과 그것이 쓰는 직렬화 | [26장](../part4/ch26_graph_interpreter.md) |
| ONNX, protobuf | 프레임워크 사이의 모델 파일 형식과 그것이 쓰는 직렬화 | [30장](../part4/ch30_graph_optimization.md) |
| NHWC, NCHW | 사진 텐서의 차원 순서 | [27장](../part4/ch27_convolution.md) |
| MLP, ViT, VAE, U-Net | 모델 구조의 이름들 | [3장](../part1/ch03_neural_networks.md), [32장](../part4/ch32_vit_detection.md), [35장](../part4/ch35_image_generation.md) |
| TTFT, TPS, ITL | 첫 토큰까지의 시간, 초당 토큰 수, 토큰 사이의 간격 | [1장](../part1/ch01_landscape.md), [14장](../part3/ch14_measurement.md) |
| NPU | 신경망 계산을 위한 처리 장치 | [45장](../part6/ch45_gpu_npu.md) |
| WASI, JSPI | WebAssembly가 바깥과 만나는 약속, 비동기 함수를 기다리게 해 주는 브라우저의 기능 | [48장](../part6/ch48_browser.md), [49장](../part6/ch49_browser_gpu.md) |
| LoRA | 가중치에 작은 행렬 둘의 곱을 더하는 미세 조정 방법 | [41장](../part5/ch41_control_and_speed.md) |
| IR | 중간 표현 | [52장](../part7/ch52_ml_compilers.md) |
