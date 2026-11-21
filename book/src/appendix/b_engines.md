# 부록 B. 비교 엔진 설치와 빌드

이 책은 장마다 우리 엔진을 실전 엔진과 견줬습니다. 이 부록은 그 엔진들을
**이 책이 쓴 판 그대로** 준비하는 방법을 모읍니다. 엔진은 빨리 바뀝니다.
새 판을 받으면 숫자가 달라지고, 명령의 이름이 바뀌어 스크립트가 돌지 않을
수도 있습니다. 책의 숫자를 다시 내 보려면 여기 적힌 판을 쓰십시오.

비교 엔진은 **저장소에 들어 있지 않습니다.** 받는 곳은 `third_party/`
아래이거나 Python의 가상 환경이고, 둘 다 `.gitignore`에 있습니다.

준비하는 방법은 세 가지입니다.

| 방법 | 엔진 | 쓰는 장 |
|---|---|---|
| 소스를 받아 빌드 | llama.cpp, whisper.cpp, stable-diffusion.cpp | 13, 20, 21, 25, 33, 35, 36, 39, 47 |
| Python 꾸러미 | LiteRT, ONNX Runtime, NCNN, MNN, PyTorch, LiteRT-LM, ExecuTorch | 26~38, 45, 47, 51 |
| npm 꾸러미 | ONNX Runtime Web, Transformers.js, WebLLM | 50 |

필요한 도구: C++ 컴파일러와 CMake(소스 빌드), Python과 꾸러미 설치
도구(`pip`이나 `uv`), Node.js와 npm([50장](../part6/ch50_web_ecosystem.md)만). 저자의 맥에서는 CMake 4.4,
Python 3.12, Node.js 25였습니다.

## B.1 llama.cpp

가장 많이 쓴 비교 엔진입니다. 스크립트 하나가 받고 빌드합니다.

```bash
sh scripts/build_llama_cpp.sh           # Metal(GPU)과 Accelerate를 쓰는 기본 빌드
sh scripts/build_llama_cpp.sh --plain   # 직접 짠 CPU 코드만 남긴 빌드
```

- 판: 커밋 `b4c0e00a4fcc`(태그 `b12428`). 스크립트가 이 커밋만 받습니다.
- 놓이는 곳: `third_party/llama.cpp/build/bin/`과 `build_plain/bin/`.
- 빌드하는 것: `llama-tokenize`, `llama-completion`, `llama-bench`,
  `llama-quantize`. [39장](../part5/ch39_inference_server.md)의 서버 비교에는 `llama-server`를 더 줍니다
  (`sh scripts/build_llama_cpp.sh llama-server`).

**빌드가 둘인 까닭**은 [13장](../part2/ch13_first_showdown.md)에 있습니다. 기본 빌드의 llama.cpp는 GPU와
Apple의 행렬 라이브러리를 씁니다. 우리 엔진의 CPU 커널과 같은 조건으로
견주려면 그것들을 뺀 빌드가 필요합니다(`-DGGML_METAL=OFF -DGGML_BLAS=OFF
-DGGML_ACCELERATE=OFF`).

모델은 llama.cpp의 형식(GGUF)으로 바꿔야 합니다.

```bash
sh scripts/convert_to_gguf.sh smollm2_135m    # model_f32.gguf, model_bf16.gguf, model_q4_k_m.gguf
sh scripts/quantize_gguf.sh smollm2_135m      # model_q8_0.gguf, model_q4_0.gguf (20장)
```

변환기는 llama.cpp에 든 Python 스크립트입니다. `reference/requirements.txt`의
꾸러미에 `sentencepiece`가 더 필요합니다.

[13장](../part2/ch13_first_showdown.md)과 [25장](../part3/ch25_rematch.md)은 llama.cpp의 라이브러리를 직접 부르는 작은 C++ 프로그램을
씁니다(점수를 꺼내고, 문맥을 채운 뒤 재려고). 따로 빌드합니다.

```bash
sh scripts/build_ch13_tools.sh
sh scripts/build_ch25_tools.sh
```

[36장](../part4/ch36_multimodal.md)(멀티모달)은 llama.cpp가 GGUF로 올려 둔 SmolVLM 파일
(`ggml-org/SmolVLM-256M-Instruct-GGUF`)을 씁니다. 받는 방법과 도는 명령은
그 장에 있습니다.

## B.2 whisper.cpp

[33장](../part4/ch33_speech_recognition.md)에서 씁니다. 스크립트가 없습니다. 저자가 쓴 명령은 이렇습니다.

```bash
git clone --depth 1 --branch v1.8.7 https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --config Release
curl -fsSL -o ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

- 판: 태그 `v1.8.7`(커밋 `48f628a84833`).
- 모델 파일 `ggml-tiny.bin`(78MB)은 가중치를 16비트로 담고 있습니다. 위의
  주소는 판을 고정하지 않은 것입니다(`main`). [33장](../part4/ch33_speech_recognition.md)의 측정 때 받은 파일의
  해시를 적어 두지 않았습니다.
- 벤치마크에 디렉터리를 알려 줍니다.

```bash
WHISPER_CPP=/받은/곳/whisper.cpp sh scripts/bench_ch33.sh
```

스크립트는 그 안의 `build/bin/whisper-cli`와 `ggml-tiny.bin`을 찾고, GPU를
끄고(`-ng`) 잽니다.

## B.3 stable-diffusion.cpp

[35장](../part4/ch35_image_generation.md)에서 씁니다. 역시 손으로 준비합니다.

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout 532d2c15c430a785a712b2d94281f04e733008a4
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release                          # Metal을 쓰는 기본 빌드
cmake --build build -j 12
cmake -B build_cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF     # CPU만
cmake --build build_cpu -j 12
```

- 판: 커밋 `532d2c15c430`. 측정 때는 그날의 맨 끝을 받았고 그것이 이
  커밋이었습니다. 위의 `git checkout` 줄은 그 커밋으로 맞추는 방법을 적은
  것이고, 이 순서 그대로 다시 돌려 보지는 않았습니다.
- 모델: 한 파일짜리 체크포인트 `v1-5-pruned-emaonly.safetensors`.
  [35장](../part4/ch35_image_generation.md)의 모델과 같은 저장소(`stable-diffusion-v1-5/stable-diffusion-v1-5`,
  판 `451f4fe16113`)에 있습니다. 4.3GB이고 `scripts/download_ch35.sh`는
  이 파일을 받지 않습니다.
- 잰 명령은 이런 꼴입니다(스레드 수 `-t`만 바꿨습니다).

```bash
./build_cpu/bin/sd-cli -m v1-5-pruned-emaonly.safetensors \
    -p "a watercolor painting of a lighthouse on a cliff at sunset" \
    --steps 20 --cfg-scale 7.5 --sampling-method euler -s 35 -t 12
```

## B.4 Python으로 받는 엔진들

LiteRT, ONNX Runtime, NCNN, MNN은 Python 꾸러미 그대로 썼습니다. 장마다
`reference/requirements_chNN.txt`에 판이 고정되어 있습니다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r reference/requirements_ch31.txt
PYTHON=.venv/bin/python sh scripts/bench_ch31.sh
```

벤치마크 스크립트는 `PYTHON`이라는 환경 변수로 어느 Python을 쓸지
받습니다.

| 엔진 | 꾸러미와 판 | 요구 파일 | 쓰는 장 |
|---|---|---|---|
| LiteRT (+ XNNPACK) | `ai-edge-litert` 2.3.0 | `requirements_ch26.txt`부터 | 26~31, 45 |
| LiteRT로 바꾸는 도구 | `litert-torch` 0.10.0, `ai-edge-quantizer` 0.10.1 | `requirements_ch26.txt`, `ch29` | 26, 27, 29 |
| ONNX Runtime | `onnxruntime` 1.32.0 | `requirements_ch30.txt`부터 | 30~34, 45, 51 |
| NCNN | `ncnn` 1.0.20261020, 변환기 `pnnx` 20261020 | `requirements_ch31.txt` | 31, 45 |
| MNN | `MNN` 3.6.2 | `requirements_ch31.txt` | 31, 45 |
| PyTorch | `torch` 2.14.1(4부), 2.15.0(2부의 정답, 47, [51장](../part7/ch51_why_not_pytorch.md)) | 장마다 | 정답 만들기, 32~37, 51 |
| transformers | 5.12.1(4부), 5.23.0(2부의 정답, [51장](../part7/ch51_why_not_pytorch.md)) | 장마다 | 정답 만들기 |
| diffusers | 0.43.0 | `requirements_ch35.txt` | 35 |
| LiteRT-LM | `litert-lm` 0.19.2 | `requirements_ch47.txt` | 47 |
| ExecuTorch | `executorch` 1.6.0 | `requirements_ch47.txt`, `ch51` | 47, 51 |

알아 둘 것들입니다.

- **환경을 장마다 따로 만드는 편이 안전합니다.** ExecuTorch 1.6.0은
  `torch` 2.15.0을 요구하고, 4부의 장들은 2.14.1로 확인했습니다. 한 환경에
  모두 넣으면 판이 충돌합니다.
- **NCNN과 MNN의 모델은 직접 만듭니다.** [30장](../part4/ch30_graph_optimization.md)의 ONNX 파일을
  `reference/ch31_convert.py`가 두 엔진의 형식으로 바꿔
  `models/vision_mobile/`에 둡니다.
- **NCNN의 기본값은 16비트 계산입니다.** 다른 엔진과 같은 계산을 시키려면
  변환할 때 `fp16=0`을 주고 실행할 때 `use_fp16_*`을 꺼야 합니다([31장](../part4/ch31_mobile_engines.md)).
  `ch31_convert.py`와 `ch31_time.py`가 두 가지를 모두 만들고 잽니다.
- **LiteRT의 GPU 델리게이트는 맥용 Python 꾸러미에 없습니다.** [45장](../part6/ch45_gpu_npu.md)에서
  재지 못한 까닭입니다.
- 꾸러미의 바이너리가 어떤 명령(SME, NEON)으로 계산하는지는 판마다 다를
  수 있습니다. [31장](../part4/ch31_mobile_engines.md)은 받은 바이너리를 뜯어 확인했습니다.

## B.5 Core ML과 운영체제의 모델

[45장](../part6/ch45_gpu_npu.md)의 Core ML과 [47장](../part6/ch47_on_device_llm.md)의 Foundation Models는 macOS에 들어 있습니다. 받을
것은 없고, Swift로 쓴 작은 다리를 빌드합니다. Xcode의 명령줄 도구가
필요합니다.

```bash
sh scripts/build_coreml.sh        # target/coreml/libbadak_coreml.dylib
```

저자의 환경은 macOS 27.0.1, Xcode 27.2였습니다. [47장](../part6/ch47_on_device_llm.md)의 운영체제 내장
모델은 그것을 지원하는 기기와 설정에서만 답합니다.

## B.6 브라우저의 엔진들

[50장](../part6/ch50_web_ecosystem.md)의 세 엔진은 npm에서 받습니다. 스크립트가 판을 고정해
`third_party/web/`에 놓고, 그 엔진들이 읽는 모델도 함께 받습니다(4.4GB쯤).

```bash
sh scripts/download_ch50.sh
```

| 엔진 | 꾸러미와 판 | 라이선스 |
|---|---|---|
| ONNX Runtime Web | `onnxruntime-web` 1.31.1 | MIT |
| Transformers.js | `@huggingface/transformers` 4.3.2 | Apache-2.0 |
| WebLLM | `@mlc-ai/web-llm` 0.2.89 | Apache-2.0 |

브라우저는 저자의 맥에 깔린 것을 썼습니다. Chrome 157, Edge 153, Firefox
159, Safari 27.0.1. 브라우저는 스스로 갱신되므로 판을 고정할 수 없습니다.
[50장](../part6/ch50_web_ecosystem.md)의 숫자가 가장 빨리 낡을 숫자입니다.

## B.7 읽기만 한 엔진들

다음 엔진들은 소스나 문서를 읽었을 뿐 돌리지 않았습니다. 준비하는 방법도
이 책에 없습니다. 읽은 판은 그 장의 참고 문헌에 있습니다.

| 엔진 | 장 | 돌리지 않은 까닭 |
|---|---|---|
| vLLM, SGLang, TensorRT LLM | 42 | 데이터센터의 GPU가 없다 |
| MLC LLM (맥에서 직접) | 47 | 실행 꾸러미가 그 프로젝트의 사이트에서만 배포되고, 모델을 기기용으로 컴파일하는 단계가 있다 |
| TVM, IREE | 52 | MLC가 컴파일해 둔 결과를 읽는 것으로 대신했다 |
| Android의 AICore, 제조사의 NPU | 45, 47 | 실기기가 없다 |

## B.8 판을 올릴 때

엔진의 새 판으로 다시 재고 싶다면 순서는 이렇습니다.

1. 판을 한 곳에서 바꿉니다. 스크립트의 `COMMIT`, `requirements_chNN.txt`의
   숫자.
2. **답이 같은지 먼저** 봅니다. 장마다 `chNN_verify` 예제나 대조하는
   스크립트가 있습니다. 엔진의 기본값이 바뀌어 다른 계산을 하게 되는 일이
   드물지 않습니다([13장](../part2/ch13_first_showdown.md)의 플래시 어텐션, [31장](../part4/ch31_mobile_engines.md)의 16비트).
3. 그다음에 벤치마크를 돌립니다.
4. 결과와 함께 판을 적습니다. `bench/chNN/`의 파일들이 그 기록입니다.
