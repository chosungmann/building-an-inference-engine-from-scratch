# 부록 F. 참고 문헌 총람과 더 읽을거리

이 책의 근거는 장마다 끝의 "참고 문헌"에 각주로 달려 있습니다. 이 부록은
그것을 책 전체로 모아 봅니다.

- F.1 **더 읽을거리.** 주제마다 먼저 읽을 것을 몇 개씩 골랐습니다.
- F.2 **논문 총람.** 책이 인용한 논문 전부. 어느 장이 인용했는지와 함께.
- F.3 **읽은 소스 저장소.** 책이 소스를 읽거나 문서를 인용한 저장소 전부.
- F.4 **참고한 책.**
- F.5 **이 책의 근거를 읽는 법.**

F.2와 F.3의 표는 손으로 쓴 것이 아닙니다. `scripts/build_references.py`가
모든 장의 각주를 읽어 만듭니다. 장의 각주를 고치면 이 표도 따라
바뀝니다.

## F.1 더 읽을거리

장마다의 각주는 "이 문장의 근거"입니다. 여기에는 "이 주제를 더 알고
싶으면 무엇부터"를 적습니다. 모두 본문에서 인용한 것 가운데 골랐습니다.

먼저 밝혀 둡니다. **논문은 대부분 초록까지만 읽었습니다.** 어디까지
읽었는지는 그 논문을 인용한 장의 각주에 적혀 있습니다. 아래의 한 줄은
"무엇을 다루는 글인가"이지 "읽어 보니 어떻더라"가 아닙니다. 반대로 소스
저장소들은 실제로 열어 읽은 것이고, 각주에 파일과 줄 번호가 있습니다.

### 트랜스포머와 LLM의 구조 (2부)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Vaswani 외, "Attention Is All You Need" (arXiv:1706.03762) | 트랜스포머를 처음 내놓은 논문. [9장](../part2/ch09_transformer.md)의 구조가 여기서 나왔다 |
| Su 외, "RoFormer" (arXiv:2104.09864) | 회전 위치 임베딩(RoPE). [9장](../part2/ch09_transformer.md)에서 구현한 것 |
| Ainslie 외, "GQA" (arXiv:2305.13245) | 묶음 질의 어텐션. KV 캐시가 작아지는 까닭(9, [24장](../part3/ch24_attention.md)) |
| Holtzman 외, "The Curious Case of Neural Text Degeneration" (arXiv:1904.09751) | top-p 샘플링. 왜 가장 그럴듯한 글이 좋은 글이 아닌가([12장](../part2/ch12_sampling.md)) |
| llama.cpp의 소스 (`ggml-org/llama.cpp`) | 이 책의 2부, 3부와 나란히 읽을 실전 엔진. 13, 20, [25장](../part3/ch25_rematch.md)의 각주에 읽은 파일이 있다 |

### 성능 (3부)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Reddi 외, "MLPerf Inference Benchmark" (arXiv:1911.02549) | 추론을 재는 규칙을 업계가 어떻게 정했는가([14장](../part3/ch14_measurement.md)) |
| Jacob 외, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (arXiv:1712.05877) | 정수 양자화의 눈금과 영점. [19장](../part3/ch19_quantization_theory.md)과 [29장](../part4/ch29_integer_inference.md)의 식이 여기서 나왔다 |
| Nagel 외, "A White Paper on Neural Network Quantization" (arXiv:2106.08295) | 양자화 전반을 정리한 글([19장](../part3/ch19_quantization_theory.md)) |
| Dao 외, "FlashAttention" (arXiv:2205.14135) | 어텐션을 토막 내어 메모리를 오가지 않게 계산하는 법([24장](../part3/ch24_attention.md)) |
| Milakov와 Gimelshein, "Online normalizer calculation for softmax" (arXiv:1805.02867) | 소프트맥스를 한 번만 훑어 계산하는 법. [24장](../part3/ch24_attention.md)의 융합이 기대는 것 |

### 그래프와 모바일 엔진 (4부의 앞)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Pisarchyk와 Lee, "Efficient Memory Management for Deep Neural Net Inference" (arXiv:2001.03288) | 가운데 값의 자리를 겹쳐 쓰는 계획. [26장](../part4/ch26_graph_interpreter.md)의 메모리 계획 |
| Dukhan, "The Indirect Convolution Algorithm" (arXiv:1907.02129) | 창을 베끼지 않는 합성곱. XNNPACK이 쓰는 방법(28, [31장](../part4/ch31_mobile_engines.md)) |
| Lavin과 Gray, "Fast Algorithms for Convolutional Neural Networks" (arXiv:1509.09308) | 위노그라드 합성곱([28장](../part4/ch28_conv_kernels.md)) |
| LiteRT의 소스 (`google-ai-edge/LiteRT`)와 XNNPACK (`google/XNNPACK`) | 그래프 해석기의 실전판. 26, 29, [31장](../part4/ch31_mobile_engines.md)의 각주에 읽은 파일과 줄이 있다 |
| Jiang 외, "MNN" (arXiv:2002.12418) | 모바일 엔진 하나의 설계를 엔진을 만든 쪽이 설명한 글(1, [31장](../part4/ch31_mobile_engines.md)) |

### 분야별 모델 (4부의 뒤)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Dosovitskiy 외, "An Image is Worth 16x16 Words" (arXiv:2010.11929) | 비전 트랜스포머([32장](../part4/ch32_vit_detection.md)) |
| Radford 외, "Robust Speech Recognition via Large-Scale Weak Supervision" (arXiv:2212.04356) | Whisper([33장](../part4/ch33_speech_recognition.md)) |
| Reimers와 Gurevych, "Sentence-BERT" (arXiv:1908.10084) | 문장 임베딩을 만드는 법([34장](../part4/ch34_embedding_search.md)) |
| Malkov와 Yashunin, HNSW (arXiv:1603.09320) | 근사 검색. [34장](../part4/ch34_embedding_search.md)이 만들지 않고 남긴 것 |
| Rombach 외, "High-Resolution Image Synthesis with Latent Diffusion Models" (arXiv:2112.10752) | Stable Diffusion의 구조([35장](../part4/ch35_image_generation.md)) |
| Song 외, "Denoising Diffusion Implicit Models" (arXiv:2010.02502) | [35장](../part4/ch35_image_generation.md)에서 구현한 스케줄러 |
| Marafioti 외, "SmolVLM" (arXiv:2504.05299) | 작은 멀티모달 모델을 어떻게 줄였는가([36장](../part4/ch36_multimodal.md)) |

### 서빙 (5부)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Kwon 외, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arXiv:2309.06180) | vLLM. KV 캐시를 페이지로 나누는 생각([40장](../part5/ch40_throughput.md)) |
| Agrawal 외, "Sarathi-Serve" (arXiv:2403.02310) | 긴 프롬프트를 끊어 읽어 쓰는 요청을 막지 않는 법([40장](../part5/ch40_throughput.md)) |
| Leviathan 외, "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192) | 추측 디코딩([41장](../part5/ch41_control_and_speed.md)) |
| Willard와 Louf, "Efficient Guided Generation for Large Language Models" (arXiv:2307.09702) | 문법으로 출력을 제한하는 법([41장](../part5/ch41_control_and_speed.md)) |
| Hu 외, "LoRA" (arXiv:2106.09685) | 작은 어댑터로 모델을 고치는 법([41장](../part5/ch41_control_and_speed.md)) |
| Kiely, *Inference Engineering* (F.4절) | 데이터센터에서 추론을 운영하는 쪽의 관점 |

### 기기와 브라우저 (6부)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Lee 외, "On-Device Neural Net Inference with Mobile GPUs" (arXiv:1907.01989) | LiteRT의 GPU 델리게이트를 만든 쪽의 설명([45장](../part6/ch45_gpu_npu.md)) |
| Hinton 외, "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531) | 증류([46장](../part6/ch46_shrinking_models.md)) |
| Han 외, "Learning both Weights and Connections for Efficient Neural Networks" (arXiv:1506.02626) | 가지치기와 재학습([46장](../part6/ch46_shrinking_models.md)) |
| Ruan 외, "WebLLM" (arXiv:2412.15803) | 컴파일한 모델을 브라우저에서 돌리는 엔진(50, [52장](../part7/ch52_ml_compilers.md)) |
| ExecuTorch (`pytorch/executorch`)와 LiteRT-LM (`google-ai-edge/LiteRT-LM`) | 기기용 LLM 엔진의 실전판(47, [51장](../part7/ch51_why_not_pytorch.md)) |

### 컴파일러 (7부)

| 무엇 | 얻을 수 있는 것 |
|---|---|
| Chen 외, "TVM" (arXiv:1802.04799) | ML 컴파일러의 대표. 그래프의 층과 연산자의 층([52장](../part7/ch52_ml_compilers.md)) |
| Chen 외, "Learning to Optimize Tensor Programs" (arXiv:1805.08166) | 커널을 찾는 일(52.7절의 큰 판) |
| Lai 외, "Relax" (arXiv:2311.02103) | 동적 형상을 다루는 컴파일러의 표현 |
| Lattner 외, "MLIR" (arXiv:2002.11054) | 컴파일러를 짓는 재료 |
| Paszke 외, "PyTorch" (arXiv:1912.01703) | 즉시 실행이 무엇을 위해 무엇을 치르는가([51장](../part7/ch51_why_not_pytorch.md)) |

## F.2 논문 총람

책이 인용한 논문 가운데 arXiv 번호가 있는 것 전부입니다. 처음 인용한
장의 순서로 늘어놓았습니다. 제목과 저자와 연도는 arXiv가 알려 주는 값을
그대로 실었고(`reference/arxiv_titles.tsv`), 번호마다 각주에 적힌 제목과
arXiv의 제목이 맞는지 대조했습니다.

{{#include generated/papers.md}}

arXiv에 없는 것은 이 표에 없습니다. 예를 들어 MNIST의 논문([7장](../part1/ch07_mnist.md)),
GPT-2의 보고서([8장](../part2/ch08_tokenizer.md)), 위상 정렬의 논문([26장](../part4/ch26_graph_interpreter.md)), 수학 함수의 근사 계수가 실린
책([32장](../part4/ch32_vit_detection.md)), RFC들([39장](../part5/ch39_inference_server.md))입니다. 그 장의 각주에 있습니다.

## F.3 읽은 소스 저장소

각주에 나오는 GitHub의 저장소 전부입니다. 많이 인용한 순서입니다.
어느 커밋의 어느 파일, 몇 번째 줄을 읽었는지는 각주에 있습니다.

{{#include generated/repositories.md}}

## F.4 참고한 책

**Philip Kiely, *Inference Engineering*, Baseten Books, 2026.** 데이터센터의
GPU 위에서 기존 엔진(vLLM, SGLang, TensorRT-LLM)을 써서 생성 모델을
운영하는 법을 다루는 책입니다. 이 책을 기획할 때 검토했고, 다음을 아이디어
수준에서 참고했습니다. 문장과 그림은 가져오지 않았고, 모두 직접 구현하고
재서 새로 썼습니다.

| 참고한 것 | 이 책의 어디 |
|---|---|
| 지연, 처리량, 비용, 품질 사이의 균형이라는 틀과 지표의 낱말(첫 토큰 시간, 초당 토큰, 토큰 사이의 간격, 백분위수) | 1, [14장](../part3/ch14_measurement.md) |
| 병목을 계산하는 법(산술 강도, 루프라인) | [15장](../part3/ch15_bottleneck.md) |
| 양자화한 뒤의 품질을 재는 법 | [21장](../part3/ch21_quantization_quality.md) |
| 어텐션의 최적화와 커널 융합, 긴 문맥 | [24장](../part3/ch24_attention.md) |
| 전문가 혼합(MoE) 모델의 구조 | [9장](../part2/ch09_transformer.md) |
| 추측 디코딩의 여러 방식 | [41장](../part5/ch41_control_and_speed.md) |
| KV 캐시를 다시 쓰는 법(접두사 캐시, 페이지, 끊어 읽기) | 39, [40장](../part5/ch40_throughput.md) |
| 음성 인식의 실전 파이프라인 | [33장](../part4/ch33_speech_recognition.md) |
| LLM에 바탕을 둔 음성 합성의 구조와 지표 | [37장](../part4/ch37_speech_synthesis.md) |

두 책은 방향이 다릅니다. 그 책은 있는 엔진을 **운영**하는 법이고, 이
책은 엔진을 **만드는** 법입니다. 5부를 읽고 운영 쪽이 더 궁금하면 그 책이
이어집니다.

Rust를 배우는 자료는 [부록 A](a_rust.md)의 끝에 있습니다.

## F.5 이 책의 근거를 읽는 법

장마다의 각주는 다음 규칙으로 썼습니다. 근거가 의심스러울 때 어디를
봐야 하는지를 알려 주는 규칙입니다.

**무엇에 근거를 다는가.** 이 책이 직접 재지 않은 사실, 다른 엔진이
무엇을 한다는 설명, 모델과 알고리즘의 출처. 이 책이 잰 숫자의 근거는
`bench/`의 파일과 그것을 만든 스크립트입니다.

**소스 코드.** 저장소, **커밋의 해시**, 파일, 줄 번호를 적었습니다.
엔진은 날마다 바뀌므로 커밋이 없으면 근거가 되지 못합니다.

**논문.** 저자, 제목, 연도, arXiv 번호. 그리고 **어디까지 읽었는지.**
"초록까지만 읽었습니다", "서지 정보만 확인했습니다"가 붙은 각주가
많습니다. 그런 논문에서 가져온 것은 초록에 적힌 주장뿐입니다.

**웹 문서.** 주소와, 인용한 문장을 원문 그대로. 문서는 바뀌므로 읽은
날짜를 적은 곳이 있습니다.

**확인하지 못한 것.** 본문에 "확인하지 않았습니다", "읽지 않았습니다",
"짐작입니다"라고 적었습니다. 그 문장들을 찾으면 이 책이 어디에서 근거
없이 말하는지를 알 수 있습니다. 53.4절의 격차 분류와 [부록 D](d_x86_linux.md)의 예상들이
그런 곳입니다.

각주의 수입니다.

{{#include generated/counts.md}}

논문은 95편, 저장소는 59곳입니다.

이 책은 초고입니다. 저자가 근거를 하나씩 다시 열어 확인하기 전까지,
각주는 "여기를 보면 된다"는 안내이지 "확인이 끝났다"는 보증이 아닙니다.
