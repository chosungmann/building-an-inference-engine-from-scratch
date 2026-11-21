| 장 | 모델 | 일 | 조건 | badak | 상대 | 상대의 값 | 우리가 몇 배 빠른가 |
|---|---|---|---|---:|---|---:|---:|
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 프롬프트 처리 | CPU, 16비트·f32 가중치 | 437토큰/초 | llama.cpp | 200토큰/초 | **2.19** |
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 토큰 생성 | CPU, 16비트·f32 가중치 | 97.4토큰/초 | llama.cpp | 51.9토큰/초 | **1.88** |
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 프롬프트 처리 | CPU, Q4_0 | 653토큰/초 | llama.cpp | 1,348토큰/초 | 0.48 |
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 토큰 생성 | CPU, Q4_0 | 242토큰/초 | llama.cpp | 239토큰/초 | 1.01 |
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 프롬프트 처리 | f32, 둘 다 GPU를 써서 | 2,751토큰/초 | llama.cpp (Metal) | 5,273토큰/초 | 0.52 |
| [25장](../part3/ch25_rematch.md) | Llama 3.2 1B | 토큰 생성 | f32, 가장 빠른 설정끼리 | 97.4토큰/초 | llama.cpp (Metal) | 90토큰/초 | **1.08** |
| [47장](../part6/ch47_on_device_llm.md) | SmolLM2-135M | 프롬프트 처리 | CPU, Q4_0 | 3,340토큰/초 | llama.cpp | 9,372토큰/초 | 0.36 |
| [47장](../part6/ch47_on_device_llm.md) | SmolLM2-135M | 토큰 생성 | CPU, Q4_0 | 681토큰/초 | llama.cpp | 687토큰/초 | 0.99 |
| [47장](../part6/ch47_on_device_llm.md) | SmolLM2-135M | 토큰 생성 | 가장 빠른 설정끼리 | 681토큰/초 | LiteRT-LM (GPU) | 255토큰/초 | **2.67** |
| [47장](../part6/ch47_on_device_llm.md) | SmolLM2-135M | 토큰 생성 | CPU, 실수 가중치 | 406토큰/초 | ExecuTorch (XNNPACK) | 17토큰/초 | **23.88** |
| [31장](../part4/ch31_mobile_engines.md) | ResNet-50 | 사진 한 장 | 스레드 1 | 15.15ms | LiteRT + XNNPACK | 7.96ms | 0.53 |
| [31장](../part4/ch31_mobile_engines.md) | ResNet-50 | 사진 한 장 | 스레드 12 | 7.73ms | LiteRT + XNNPACK | 5.86ms | 0.76 |
| [31장](../part4/ch31_mobile_engines.md) | MobileNetV2 | 사진 한 장 | 스레드 1 | 4.26ms | LiteRT + XNNPACK | 1.97ms | 0.46 |
| [31장](../part4/ch31_mobile_engines.md) | MobileNetV2 | 사진 한 장 | 가장 빠를 때끼리 | 1.14ms | LiteRT + XNNPACK (스레드 4) | 1.35ms | **1.18** |
| [31장](../part4/ch31_mobile_engines.md) | MobileNetV2 int8 | 사진 한 장 | 스레드 1 | 11.51ms | LiteRT + XNNPACK | 1.7ms | 0.15 |
| [45장](../part6/ch45_gpu_npu.md) | MobileNetV2 | 사진 한 장 | CPU끼리 | 3.06ms | Core ML (CPU) | 3.22ms | **1.05** |
| [45장](../part6/ch45_gpu_npu.md) | MobileNetV2 | 사진 한 장 | 상대는 NPU | 3.06ms | Core ML (뉴럴 엔진) | 0.4ms | 0.13 |
| [45장](../part6/ch45_gpu_npu.md) | ResNet-50 | 사진 한 장 | 상대는 NPU | 12.15ms | Core ML (뉴럴 엔진) | 1.03ms | 0.08 |
| [32장](../part4/ch32_vit_detection.md) | YOLOX-Tiny | 사진 한 장 | 스레드 1 | 27.2ms | ONNX Runtime | 12.4ms | 0.46 |
| [32장](../part4/ch32_vit_detection.md) | YOLOX-Tiny | 사진 한 장 | 스레드 12 | 8.23ms | ONNX Runtime | 7.31ms | 0.89 |
| [32장](../part4/ch32_vit_detection.md) | ViT-Tiny | 사진 한 장 | 스레드 1 | 12.38ms | PyTorch | 6.14ms | 0.50 |
| [32장](../part4/ch32_vit_detection.md) | ViT-Tiny | 사진 한 장 | 가장 빠를 때끼리 | 8.67ms | PyTorch (스레드 4) | 5.31ms | 0.61 |
| [33장](../part4/ch33_speech_recognition.md) | Whisper-tiny | 인코더, 30초 | 스레드 1 | 132ms | whisper.cpp | 242ms | **1.83** |
| [33장](../part4/ch33_speech_recognition.md) | Whisper-tiny | 인코더, 30초 | 스레드 12 | 74.7ms | whisper.cpp | 45ms | 0.60 |
| [33장](../part4/ch33_speech_recognition.md) | Whisper-tiny | 탐욕 디코딩 | 스레드 12 | 30.1ms | whisper.cpp | 16ms | 0.53 |
| [34장](../part4/ch34_embedding_search.md) | all-MiniLM-L6-v2 | 대량 처리 | 스레드 12 | 13,149글/초 | ONNX Runtime | 5,605글/초 | **2.35** |
| [34장](../part4/ch34_embedding_search.md) | all-MiniLM-L6-v2 | 대량 처리 | 스레드 1 | 2,280글/초 | PyTorch | 2,830글/초 | 0.81 |
| [34장](../part4/ch34_embedding_search.md) | all-MiniLM-L6-v2 | 질의 하나, 8토큰 | 스레드 1 | 0.65ms | ONNX Runtime | 0.96ms | **1.48** |
| [34장](../part4/ch34_embedding_search.md) | all-MiniLM-L6-v2 | 질의 하나, 256토큰 | 가장 빠를 때끼리 | 16.24ms | ONNX Runtime (스레드 12) | 4.3ms | 0.26 |
| [35장](../part4/ch35_image_generation.md) | Stable Diffusion 1.5 | U-Net 한 번 | 스레드 12 | 0.66초 | PyTorch | 0.39초 | 0.59 |
| [35장](../part4/ch35_image_generation.md) | Stable Diffusion 1.5 | 잡음 걷어 내기, 20걸음 | CPU, 스레드 12 | 26.6초 | stable-diffusion.cpp | 77.5초 | **2.91** |
| [35장](../part4/ch35_image_generation.md) | Stable Diffusion 1.5 | 잡음 걷어 내기, 20걸음 | 상대는 GPU | 26.6초 | stable-diffusion.cpp (Metal) | 8초 | 0.30 |
| [36장](../part4/ch36_multimodal.md) | SmolVLM-256M | 사진 보기 | 스레드 12 | 186ms | PyTorch | 101ms | 0.54 |
| [36장](../part4/ch36_multimodal.md) | SmolVLM-256M | 사진 보기 | CPU끼리 | 186ms | llama.cpp | 396ms | **2.13** |
| [36장](../part4/ch36_multimodal.md) | SmolVLM-256M | 토큰 하나 쓰기 | 스레드 12 | 3.4ms | PyTorch | 7.8ms | **2.29** |
| [37장](../part4/ch37_speech_synthesis.md) | OuteTTS-0.1-350M | 토큰 하나 쓰기 | 스레드 12 | 4.1ms | PyTorch | 12.5ms | **3.05** |
| [37장](../part4/ch37_speech_synthesis.md) | OuteTTS-0.1-350M | 소리 1초 분량 풀기 | 가장 빠를 때끼리 | 12.7ms | PyTorch (스레드 1) | 27.2ms | **2.14** |
| [37장](../part4/ch37_speech_synthesis.md) | OuteTTS-0.1-350M | 소리 16초 분량 풀기 | 가장 빠를 때끼리 | 189ms | PyTorch (스레드 1) | 150ms | 0.79 |
| [50장](../part6/ch50_web_ecosystem.md) | SmolLM2-135M | 토큰 생성 | 16비트 가중치, Chrome | 338토큰/초 | WebLLM (GPU) | 212토큰/초 | **1.59** |
| [50장](../part6/ch50_web_ecosystem.md) | Llama 3.2 1B | 토큰 생성 | 16비트 가중치, Chrome | 93토큰/초 | WebLLM (GPU) | 125토큰/초 | 0.74 |
| [50장](../part6/ch50_web_ecosystem.md) | Llama 3.2 1B | 토큰 생성 | 가장 빠른 설정끼리, Chrome | 93토큰/초 | WebLLM (GPU, 4비트) | 207토큰/초 | 0.45 |
| [50장](../part6/ch50_web_ecosystem.md) | Whisper-tiny | 11초 받아쓰기 | Chrome | 293ms | Transformers.js (GPU) | 208ms | 0.71 |
| [51장](ch51_why_not_pytorch.md) | SmolLM2-135M | 프롬프트 처리 | CPU | 2,888토큰/초 | PyTorch | 2,798토큰/초 | 1.03 |
| [51장](ch51_why_not_pytorch.md) | SmolLM2-135M | 토큰 생성 | CPU | 358토큰/초 | PyTorch | 97토큰/초 | **3.69** |
| [51장](ch51_why_not_pytorch.md) | SmolLM2-135M | 띄워서 글 한 줄 쓰고 끝내기 | CPU | 0.15초 | PyTorch | 5.33초 | **35.53** |
