# 부록 C. 모델 내려받기와 라이선스

이 책의 저장소에는 **모델 파일이 하나도 없습니다.** 가중치도, 시험에
쓰는 사진과 소리도, 데이터도 없습니다. 모두 원래 공개된 곳에서 받습니다.
이 부록은 무엇을 어디에서 받는지, 그리고 그것이 어떤 조건으로 공개되어
있는지를 한곳에 모읍니다.

먼저 알아 둘 것입니다. **아래의 라이선스 표시는 저자가 배포처의 설명을
읽고 옮겨 적은 것이고, 법률 조언이 아닙니다.** 모델을 제품에 쓰거나 다시
배포하려면 배포처의 원문을 직접 읽어야 합니다. 조건은 바뀌기도 합니다.

## C.1 받는 방법

장마다 스크립트가 있습니다.

```bash
sh scripts/download_smollm2.sh     # 2부의 모델
sh scripts/download_ch27.sh        # 27장의 사진, 모델, 정답
```

스크립트는 모두 같은 일을 합니다.

- **버전을 고정해 받습니다.** Hugging Face Hub의 파일은 주소에 버전(revision,
  커밋의 해시)을 넣어 받습니다. 배포처가 파일을 바꿔도 같은 것을 받습니다.
- **SHA-256 해시를 확인합니다.** 받은 파일의 해시가 스크립트에 적힌 것과
  다르면 멈춥니다.
- **이미 있으면 다시 받지 않습니다.**
- `models/` 아래에 둡니다. 이 디렉터리는 `.gitignore`에 있습니다. 다른
  곳에 두려면 환경 변수 `BADAK_MODELS`로 알려 줍니다.

필요한 도구는 `curl`과 `shasum`뿐입니다. [50장](../part6/ch50_web_ecosystem.md)만 npm이 더 필요합니다.

책의 모든 실험을 따라 하면 `models/`가 수십 GB가 됩니다(저자의 것은
37GB입니다. 변환하고 양자화한 파일들이 들어 있어서입니다). 장마다 필요한
것만 받으면 됩니다.

## C.2 남이 만든 모델

원래 배포처에서 그대로 받는 것입니다.

| 모델 | 받는 곳 (Hugging Face Hub의 저장소) | 라이선스 | 크기 | 스크립트 | 장 |
|---|---|---|---:|---|---|
| SmolLM2-135M-Instruct | `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache-2.0 | 0.27GB | `download_smollm2.sh` | 2부부터 |
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 Community License. **동의와 승인이 필요** | 2.5GB | `download_llama32.sh` | 2부, 3부, 5부 |
| SmolLM2-360M, 1.7B Instruct | `HuggingFaceTB/SmolLM2-360M-Instruct`, `-1.7B-Instruct` | Apache-2.0 | 0.7GB, 3.4GB | `download_ch41.sh` | 41, 46, 47 |
| ViT-Tiny, ViT-B/16 | `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`, `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | Apache-2.0 | 23MB, 346MB | `download_ch32.sh [base]` | 32 |
| YOLOX-Tiny | YOLOX 프로젝트의 릴리스 `0.1.1rc0` (GitHub) | Apache-2.0 | 20MB | `download_ch32.sh` | 32 |
| Whisper-tiny | `openai/whisper-tiny` | Apache-2.0 (저장소의 설명에 적힌 것) | 0.15GB | `download_ch33.sh` | 33, 38, 50 |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | 91MB | `download_ch34.sh [onnx]` | 34 |
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M. **쓰임에 제한이 있다** | 4.3GB | `download_ch35.sh [text]` | 35 |
| SmolVLM-256M-Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` | Apache-2.0 | 0.5GB | `download_ch36.sh` | 36 |
| OuteTTS-0.1-350M | `OuteAI/OuteTTS-0.1-350M` | CC-BY-4.0 | 0.7GB | `download_ch37.sh` | 37, 38 |
| WavTokenizer의 디코더 | `novateur/WavTokenizer-large-speech-75token` | MIT | 1.75GB | `download_ch37.sh decoder` | 37, 38 |
| 다른 엔진용 SmolLM2-135M | `litert-community/SmolLM2-135M-Instruct`, `executorch-community/SmolLM2-135M` | Apache-2.0 | 143MB, 543MB | `download_ch47.sh` | 47 |
| 브라우저 엔진용 모델들 | `onnx-community/whisper-tiny`, `mlc-ai/…-MLC`, `mlc-ai/binary-mlc-llm-libs` 등 | 아래 C.5절 | 4.4GB | `download_ch50.sh` | 50, 52 |

버전(커밋의 해시)과 파일마다의 SHA-256은 스크립트에 있습니다.

### 조심할 모델들

**Llama 3.2.** 받기 전에 Hugging Face에서 Meta의 라이선스에 동의하고
승인을 받아야 합니다. 스크립트는 승인받은 계정의 토큰을 씁니다(환경 변수
`HF_TOKEN`, 또는 `hf auth login`이 저장해 둔 것). **토큰을 스크립트나
저장소에 적으면 안 됩니다.** 승인 없이 따라 하려면 SmolLM2-135M으로
2부를 읽을 수 있습니다. 책의 숫자는 대부분 두 모델 모두로 적혀 있습니다.

**Stable Diffusion 1.5.** 라이선스에 만든 그림으로 해서는 안 되는 일이
적혀 있습니다. 받기 전에 그 저장소의 설명을 읽어야 합니다. 원래의
저장소(`runwayml/…`)는 내려갔고, 이 책이 받는 곳은 그
사본입니다([35장](../part4/ch35_image_generation.md)).

**WavTokenizer의 디코더.** 배포된 파일이 PyTorch의 피클입니다. 피클은
읽을 때 코드를 실행할 수 있는 형식입니다. 이 책은 가중치만 읽는 방식
(`torch.load(weights_only=True)`)으로만 열고, 독자의 기계에서
safetensors로 바꿉니다(`reference/ch37_convert.py`, 37.3절). 바꾼 파일은
이 책이 배포하지 않습니다.

**OuteTTS.** CC-BY-4.0은 출처를 밝히면 쓸 수 있는 조건입니다. 만든 소리를
공개할 때 적용됩니다.

## C.3 이 책이 만들어 올린 것

이 책이 직접 만든 파일은 Hugging Face Hub의 한 조직에 있습니다.
<https://huggingface.co/building-an-inference-engine-from-scratch>

| 저장소 | 무엇 | 장 |
|---|---|---|
| `mnist-mlp` | 이 책이 학습한 손글씨 숫자 분류 모델과 PyTorch의 정답 | 7 |
| `mnist-tflite` | 같은 모델을 `.tflite`로 바꾼 것 둘과 LiteRT의 정답 | 26 |
| `imagenet-tflite` | 남이 공개한 ResNet-50과 MobileNetV2의 가중치를 `.tflite`와 ONNX로 **형식만 바꾼 것**, 8비트 정수로 양자화한 MobileNetV2, 그리고 27~38장의 정답 파일들 | 27, 29, 30, 32~38 |
| `smollm2-135m-reference` | SmolLM2-135M의 정답(가운데 값, 토크나이저의 시험 벡터, 생성한 글). **가중치는 없다** | 9~13 |
| `llama32-1b-reference` | Llama 3.2 1B의 정답. **가중치는 없다.** Llama의 라이선스 파일을 함께 두었다 | 9~13 |

세 가지 원칙을 지켰습니다.

- **남의 가중치를 다시 올린 것은 한 곳뿐입니다.** `imagenet-tflite`의
  ResNet-50(원본 BSD-3-Clause)과 MobileNetV2(원본 Apache-2.0)입니다. 두
  라이선스가 다시 배포하는 것을 허락하고, 원본의 출처와 라이선스 전문을
  그 저장소에 함께 두었습니다. 값은 바꾸지
  않았습니다([27장](../part4/ch27_convolution.md)).
- **그 밖에는 정답만 올렸습니다.** "정답"은 기준 구현(PyTorch, LiteRT,
  ONNX Runtime, transformers)이 낸 출력과 가운데 값입니다. 엔진의
  테스트가 이것과 견줍니다. 모델에서 나온 숫자이므로 Llama 3.2의
  정답에는 그 모델의 라이선스 파일을 붙였습니다.
- **직접 만들 수 있습니다.** 올린 것마다 그것을 만든 스크립트가
  `reference/`에 있습니다(`ch07_train_mnist.py`, `ch27_make_tflite.py`,
  `chNN_make_reference.py`). 내려받는 대신 그 스크립트를 돌려도 됩니다.

올리지 않은 것도 있습니다. [41장](../part5/ch41_control_and_speed.md)의 LoRA 어댑터와 [46장](../part6/ch46_shrinking_models.md)에서 학습한 작은
모델들입니다. 독자의 기계에서 학습 스크립트로 만듭니다
(`reference/ch41_train_lora.py`, `ch46_train.py`).

## C.4 사진, 소리, 글

| 자료 | 받는 곳 | 조건 | 스크립트 | 장 |
|---|---|---|---|---|
| MNIST | PyTorch 프로젝트가 운영하는 미러 | 이 책은 허가 조건을 확인하지 않았습니다. 연구와 교육에 널리 쓰이는 데이터입니다 | `download_mnist.sh` | 7, 26 |
| 시험 사진 다섯 장 | 위키미디어 공용 | 모두 올린 사람이 권리를 포기한 것(CC0). 사진마다의 출처는 스크립트에 있다 | `download_ch27.sh` | 27부터 |
| `jfk.wav` | whisper.cpp 저장소의 `samples/` | 케네디 대통령의 1961년 취임 연설. 미국 연방 정부의 저작물 | `download_ch33.sh` | 33, 38, 50 |
| STS 벤치마크 | sentence-transformers 프로젝트가 sbert.net에 올려 둔 한 파일짜리 | 이 책은 허가 조건을 확인하지 않았습니다 | `download_ch34.sh` | 34 |
| WikiText-2 (시험 부분) | llama.cpp의 `scripts/get-wikitext-2.sh`와 같은 곳 | 위키백과의 글. CC BY-SA | `download_wikitext2.sh` | 21, 46 |
| llama.cpp의 토크나이저 시험 자료 | llama.cpp 저장소 (커밋 `b4c0e00a4fcc`) | MIT (llama.cpp의 라이선스) | `download_ch08.sh` | 8 |
| [47장](../part6/ch47_on_device_llm.md)의 80문항 | 이 저장소의 `reference/ch47_questions.tsv` | 이 책이 만든 것 | | 47 |

## C.5 50장의 파일들

[50장](../part6/ch50_web_ecosystem.md)은 브라우저 엔진 셋이 읽는 형식으로 **같은 모델을 다시 받습니다.**

| 파일 | 받는 곳 | 조건 |
|---|---|---|
| SmolLM2-135M의 ONNX 버전 | `HuggingFaceTB/SmolLM2-135M-Instruct`의 `onnx/` | Apache-2.0 |
| Whisper-tiny의 ONNX 버전 | `onnx-community/whisper-tiny` | Apache-2.0 |
| SmolLM2-135M의 MLC 버전 | `mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC` | Apache-2.0 |
| Llama 3.2 1B의 MLC 버전 (16비트, 4비트) | `mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC`, `-q4f16_1-MLC` | Llama 3.2 Community License |
| MLC가 컴파일한 모델의 코드 (`.wasm`) | GitHub의 `mlc-ai/binary-mlc-llm-libs` | **저장소에 라이선스 파일이 없습니다** |

마지막 줄 때문에 [52장](../part7/ch52_ml_compilers.md)은 그 파일에서 꺼낸 셰이더를 저장소에 옮겨 두지
않았습니다. 센 숫자와 본문의 몇 줄만 실었습니다.

## C.6 이 저장소의 라이선스

- **소스 코드**(`crates/`, `apps/`, `scripts/`, `reference/`, `notebooks/`)는 MIT
  라이선스입니다.
- **책의 글과 그림**(`book/`, `docs/`)은 그 라이선스에 들어 있지 않습니다.
  저자가 모든 권리를 가집니다.

다른 엔진의 코드를 옮긴 곳에는 그곳에 출처를 적었습니다. 대개는 개념을
읽고 새로 썼습니다. 숫자를 그대로 가져온 것(예를 들어
[32장](../part4/ch32_vit_detection.md)의 지수 함수 근사의 계수)은 주석과
본문의 각주에 출처가 있습니다.

## C.7 모델이 사라졌을 때

모델의 저장소는 내려가기도 합니다(Stable Diffusion 1.5의 원래 저장소가
그랬습니다). 스크립트가 받지 못하면 이렇게 합니다.

1. 스크립트에 적힌 **SHA-256 해시**로 같은 파일의 다른 사본을 찾습니다.
   해시가 같으면 같은 파일입니다.
2. 사본이 없으면 같은 구조의 다른 모델로 바꿉니다. 엔진은 모델의 이름이
   아니라 구조와 텐서의 이름을 봅니다. 숫자는 달라지지만 코드는 돕니다.
3. 정답 파일은 `reference/`의 스크립트로 다시 만들 수 있습니다. 기준
   구현의 버전을 `requirements_chNN.txt`에 맞추십시오.
