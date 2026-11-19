# 진행 상황

세션을 시작할 때 이 문서를 먼저 읽고, 장을 끝낼 때마다 갱신합니다.

## 다음에 할 일

**4부가 끝났습니다. 39장**을 씁니다(추론 서버. 5부의 첫 장). 제안서의
39장 줄을 먼저 읽으십시오: HTTP와 스트리밍 응답 직접 구현, OpenAI 호환
API, 접두사 캐시, KV 캐시 저장과 복원.

- 8장이 "39장에서 서버를 만들 때 이 부분을 고칩니다"라고 약속해 둔 것이
  있습니다(대화 틀의 뼈대와 사용자의 글을 따로 토큰화하기. 8.9절).
- 7장이 배치 처리(지연 시간과 처리량의 줄다리기)를 40장으로 미뤄
  두었습니다(7장의 배치 측정 절).
- 36장과 38장이 "사진까지 읽은 캐시", "대화의 KV 캐시를 남겨 두기"를
  5부로 미뤘습니다. 38장은 "처음 한 번이 느리다(미리 데워 두기)"도
  5부로 미뤘습니다.
- 엔진은 표준 라이브러리만 씁니다. HTTP도 직접 짭니다(제안서).

## 38장 메모 (종합 실습: 음성 대화 파이프라인)

- 새 모델은 없습니다. 33장의 Whisper-tiny, 2부의 SmolLM2-135M-Instruct
  (또는 Llama 3.2 1B), 37장의 OuteTTS와 WavTokenizer 디코더를 잇습니다.
  Hub에 올린 것은 정답뿐입니다(`reference_ch38.safetensors`, 판
  `219c7314a294`. `scripts/download_ch38.sh`).
- `talk::Assistant`(모델 셋을 쥠), `respond(소리, Settings, 콜백)` →
  `Turn`(받아쓴 글, 답, 문장들, 소리, 시각 일곱 개, 소리 조각의 기록,
  토큰들). `Speak::{BySentence, AfterAnswer}`. `talk::Sentences`는
  마침표·물음표·느낌표 **뒤의 빈칸을 본 다음에** 문장을 내줍니다.
- 응답 지연 = 사용자의 말이 끝난 때부터 첫 소리까지. **말이 끝났는지
  알아차리는 시간은 들어 있지 않습니다**(파일로 받음).
- 결과(스레드 12): SmolLM2로 253ms(듣기 121, 첫 문장 42, 첫 낱말의
  소리 90). 다 쓴 뒤에 넘기면 374ms. **Llama 3.2 1B는 한 문장으로
  답해서 두 방법이 같습니다(805 / 768ms).** 문장마다 넘기는 이득은
  답에 문장의 경계가 일찍 나올 때만 있습니다. 스레드 1개는 803ms이고
  소리가 20번 끊김(만드는 속도가 실시간의 0.80배). PyTorch로 같은
  사슬은 400ms쯤(디코더 빼고, 스레드를 늘려도 그대로).
- 39장을 시작하면서 `talk.rs`가 대화 글 앞에 글머리 토큰
  (`tokenizer.prefix()`)을 붙이지 않은 것을 찾아 고쳤습니다. SmolLM2에는
  글머리 토큰이 없어 대조(38.5절)는 그대로였고, Llama 3.2 1B의 답과
  숫자가 바뀌어 다시 쟀습니다(고치기 전: 두 문장, 567 / 880ms).
- 사슬 전체가 PyTorch와 토큰 단위로 같습니다(답 60, 소리 토큰 1,332).
  두 방식의 소리는 표본까지 같습니다.
- 정답 스크립트가 두 번 틀렸습니다(끝 토큰 번호가 토크나이저에 없음,
  `decode`가 목록을 돌려줌).
- 37장의 `tts::prompt`를 기준 구현의 글 다듬기에 맞췄습니다(`it's` →
  `its`. 37장 커밋에 합침).
- CI는 정답만 받습니다. 디코더의 가중치가 없으면 대조와 시험이
  건너뜁니다.
- 남긴 것: 마이크·스피커, 말의 끝 검출, 끼어들기, 받아쓰기 흘려보내기,
  두 모델을 함께 돌리기, 이어지는 대화.

## 37장 메모 (음성 합성)

- **저자가 정한 것(2026-11-20)**: 모델은 `OuteAI/OuteTTS-0.1-350M`
  (판 `2191ede86648`, CC-BY-4.0). 오디오 디코더는 WavTokenizer의 원본
  체크포인트(`novateur/WavTokenizer-large-speech-75token`, 판
  `f0eafde4078a`, MIT)를 `torch.load(weights_only=True)`로만 열어 독자의
  기계에서 safetensors로 바꾼다(`reference/ch37_convert.py`. 바꾼 것을
  Hub에 올리지 않는다). 견줄 엔진은 llama.cpp가 아니라 PyTorch다(고정해
  둔 llama.cpp에는 예전 OuteTTS 예제가 없다. 본문 37.2절에 적었다).
- Hub에 올린 것은 정답뿐입니다(`reference_ch37.safetensors`, 판
  `5e2f60b70a58`). 정답 스크립트는 OuteTTS 저장소(커밋 `f5eac6e70d79`)의
  디코더 모듈을 불러 쓰므로 `OUTETTS=<그 저장소의 경로>`가 필요합니다.
- `tts::OuteTts`(`load`, `generate(토큰, 끝 토큰, 한도, &mut Sampler,
  콜백)`), `tts::prompt`, `tts::decoder::Backbone`(`forward`, `spectrum`,
  `fast`), `tts::Voice`(`load`, `fast`, `decode`, `decode_part(코드, 처음,
  끝, 앞, 뒤)`). `badak_models`가 이제 `badak_audio`에 기댑니다.
- `badak_audio`: `Istft`(창의 **제곱**의 합으로 나누는 것은 기준 구현의
  규칙), `fft::Split`(길이가 작은 홀수 × 2의 거듭제곱일 때. 1,280 =
  5 × 256), `resample`, `Wav::to_bytes`/`save`.
- 틀린 것 둘: 12장에 있는 반복 벌점을 새로 짜다가 나온 횟수만큼 깎게
  만들어 203번째 토큰에서 갈렸다(12장의 `Sampler`를 쓰도록 고침).
  `decode_part`에서 `usize::MAX`를 더해 넘쳤다(`saturating_add`).
- 결과: 토큰 302개 일치, 디코더 몸통 4.5e-4(값의 크기 52), 소리 1.0e-4.
  스레드 12: 첫 소리 77ms, 실시간의 2.3배, 토큰마다 4.1ms(PyTorch
  12.5ms), 디코더 1초 분량 12.7ms(PyTorch 스레드 1개 27ms, 12개 364ms).
  16초 분량은 189ms 대 150ms로 PyTorch가 빠름.
- **조각내어 푼 소리는 한 번에 푼 소리와 다릅니다**(파형의 SNR 0~3dB,
  멜 2.5~5dB). 디코더의 어텐션과 그룹 정규화가 소리 전체를 봅니다.
- **소리의 품질은 확인하지 못했습니다.** Whisper-tiny로 받아쓰면 네 번
  가운데 한 번(번호 1)만 열한 낱말 중 여덟이 맞습니다.
- CI는 언어 모델의 토큰만 대조합니다(디코더의 가중치를 만들려면
  Python이 필요).
- 남긴 것: 목소리 고르기(화자), 숫자 풀기, 긴 글, 흘려보내기에 맞는
  디코더, 디코더의 남은 스레드 하나짜리 구간, 품질 측정, llama.cpp 비교.

## 36장 메모 (멀티모달)

- 모델은 **원래 배포처에서** 받습니다(`scripts/download_ch36.sh`, 0.5GB).
  `HuggingFaceTB/SmolVLM-256M-Instruct`(판 `d039ab0d4b93`, Apache-2.0).
  사진은 27장의 것. Hub에 올린 것은 정답뿐입니다
  (`reference_ch36.safetensors`, 판 `632124d754f7`).
- `smolvlm::SmolVlm`: `load`, `fast`, `vision.encode(픽셀)`, `connect`,
  `see`, `embed(토큰들, 사진의 벡터)`, `generate`. `shuffle`이 이웃
  4 × 4를 잇습니다. 35장의 `diffusion::layers`를 씁니다
  (`Kernel::load_flat`을 더함).
- `LlamaModel::from_parts`, `embed`, `hidden_cached_from`을 더했습니다.
  2부에서 책에 실은 코드는 건드리지 않았습니다.
- `badak_image::lanczos`: `resize`(Pillow의 `LANCZOS`와 식으로 만든
  사진에서 바이트까지 같음. 세로를 먼저 하는 규칙이 여기에도 있음),
  `squash(사진, 2048, 512)`.
- JPEG를 푼 픽셀이 `jpeg-decoder`와 Pillow에서 0.15% 다릅니다(27장이
  이미 적어 둔 차이를 세어 본 것). transformers의 전처리는 Pillow와도
  1~2% 다릅니다.
- 사진을 여러 조각으로 나누는 기능은 껐습니다. llama.cpp는 512 × 512
  사진에도 두 장을 인코딩합니다.
- 결과: 토큰 48개 일치. 스레드 12: 비전 305ms(빠른 길 186, PyTorch 101,
  llama.cpp CPU 396, Metal 23~35), 글 읽기 23ms, 토큰마다 3.4ms(PyTorch
  7.8).
- `tanh` GELU를 스레드에 나누고 근사해 394 → 186ms.
- 남긴 것: 사진 나누기, 여러 장, 샘플링, 양자화, PNG 읽기, GPU.

## 35장 메모 (이미지 생성)

- 모델은 **원래 배포처에서** 받습니다(`scripts/download_ch35.sh`, 4.3GB.
  `text`를 붙이면 토크나이저와 텍스트 인코더만 0.5GB).
  `stable-diffusion-v1-5/stable-diffusion-v1-5`(판 `451f4fe16113`,
  CreativeML OpenRAIL-M. 내려간 원래 저장소의 사본). Hub에 올린 것은
  정답뿐입니다(`reference_ch35.safetensors`, 판 `88881548b761`).
- `tokenizer::clip::ClipTokenizer`: `load`, `words`, `bpe`, `encode(글, 77)`.
  NFC는 없습니다.
- `diffusion::layers`: `Map`(자리 × 채널), `Kernel`(선형 층과 합성곱.
  `linear`, `conv(map, 보폭)`, `pack`), `GroupNorm`, `attend`, `layer_norm`.
  빠른 길에서는 자리들을 토막 내어 토막을 스레드에 나눕니다.
- `diffusion::{TextEncoder, Unet, Decoder}`, `Schedule`, `ddim_step`,
  `guide`, `denoise(unet, 잡음, 글, 빈 글, Settings, 콜백)`, `noise(번호,
  개수)`, `to_bytes`. `badak_image::png::encode`, `Image::save_png`.
- 결과: U-Net 한 번 차이 7.9e-6, 20걸음 뒤 383바이트가 1씩 다름. U-Net
  한 번 11.47 / 3.62 / 1.55초(스레드 1/4/12), 빠른 길 2.17 / 0.95 / 0.66,
  PyTorch 0.95 / 0.50 / 0.39. 디코더 30.2 / 11.3 / 5.1, 빠른 길 4.61 /
  1.87 / 1.27, PyTorch 2.08 / 1.40 / 1.31. 그림 한 장 28초(스레드 12).
  stable-diffusion.cpp CPU 85초, Metal 10초.
- 빠른 길의 병목은 어텐션(40~57%). 가장 큰 단의 소프트맥스.
- 정답 스크립트에서 `strict=False`로 가중치를 읽어 무작위 가중치로 돈
  일이 있었습니다. 다시는 그렇게 읽지 않습니다.
- CI는 토크나이저와 텍스트 인코더만 대조합니다(U-Net이 3.4GB).
- 남긴 것: 다른 스케줄러, 두 실행을 한 묶음으로, VAE 인코더, 양자화,
  GPU, NFC.

## 34장 메모 (임베딩과 검색)

- 모델과 글은 **원래 배포처에서** 받습니다(`scripts/download_ch34.sh`).
  `sentence-transformers/all-MiniLM-L6-v2`(판 `1110a243fdf4`,
  Apache-2.0)의 `model.safetensors`, `config.json`, `vocab.txt`, 그리고
  sbert.net의 `stsbenchmark.tsv.gz`(벤치마크의 원래 배포처는 403).
  Hub에 올린 것은 정답뿐입니다(`reference_ch34.safetensors`, 판
  `d0cc98f26772`).
- `tokenizer::wordpiece::WordPiece`: `from_vocab`, `normalize`, `words`,
  `pieces`, `encode(글, 한도)`. 유니코드 표(`NONSPACING_MARK`,
  `PUNCTUATION`, `OTHER`, `DECOMPOSITION`)는
  `scripts/generate_unicode_tables.py`가 만듭니다. 한글은 식으로 풉니다.
  NFC(합치기)는 없습니다.
- `bert::Bert`: `load(디렉터리)`, `forward(글들, Batching, 채울 토큰)` →
  `Trace`, `embed_many(글들, 채울 토큰, Plan)`, `fast()`.
  `Batching::{Padded, Ragged}`, `layout`이 글마다의 `Span`을 정합니다.
  `Plan { batch, batching, sorted, across }`. `across`는 묶음들을 풀에
  나누고, 풀이 바쁘면 안쪽 계산은 저절로 스레드 하나가 됩니다.
- 빠른 길의 어텐션 안의 곱은 32토큰부터 SME(`SME_ATTENTION`).
- `search::Index`: `extend`, `scores`(선형 층의 커널), `search(질의, k)`.
- 결과: 토큰 16/16 같음, 임베딩 차이 3e-7, STS 스피어먼 0.8203.
  스레드 1: 기본 339글/초, 빠른 길 2,280, ORT 1,966, PyTorch 2,830.
  스레드 12: 빠른 길 13,149(묶음들을 나누기. 묶음 안에서 나누면 2,745),
  ORT 5,605, PyTorch 3,670. 8토큰 질의 0.65 / 0.96 / 1.50ms, 256토큰
  16.7 / 14.2 / 10.2ms(ORT 스레드 12개 4.3ms).
- 남긴 것: 근사 검색, 벡터 줄이기, 긴 문서 나누기, 재순위 모델, SME
  커널을 스레드에 나누기(긴 글 하나에서 ORT에 지는 까닭), NFC.

## 33장 메모 (음성 인식)

- 새 크레이트 `badak_audio`: `Wav`(16비트 PCM만), `fft`(`Complex`, `dft`,
  `fft`, `Bluestein`), `mel`(`hann`, `hertz_to_mel`, `mel_filters`,
  `LogMel::WHISPER.spectrogram`), `vad`(`loudness`, `segments`,
  `windows`). 스레드 풀 때문에 `badak_core`에 기댑니다.
- 멜 스펙트로그램은 transformers의 것과 1.4e-5. 길이 400의 변환은
  Bluestein(길이 1,024의 FFT 둘, `f64`). whisper.cpp는 반으로 나누다
  홀수(25)에서 정의대로 계산합니다.
- `badak_models::whisper`: `Whisper::load(디렉터리)`, `encode(&mel)` →
  `Encoded`(인코더의 출력과, 디코더 층마다 미리 만든 교차 어텐션의 키와
  값), `start()` → `State`(KV 캐시), `step(&encoded, &mut state, 토큰)`,
  `greedy`, `beam`, `detect_language`, `fast()`. `Vocabulary`는
  `vocab.json`, `added_tokens.json`, `generation_config.json`을 읽고
  토큰을 글로만 바꿉니다(글을 토큰으로 바꾸지 않습니다).
- 모델은 `openai/whisper-tiny`(판 `169d4a4341b3`, 카드에 Apache-2.0)를
  그대로 받고, 소리는 whisper.cpp v1.8.7의 `samples/jfk.wav`. Hub에 올린
  것은 정답뿐(`imagenet-tflite`의 `reference_ch33.safetensors`, 판
  `310b423c9eb9`).
- transformers 5.12.1과 견줘 인코더 출력 1e-3(값의 크기 17), 첫 점수
  5e-5, **탐욕 23토큰과 빔 24토큰이 모두 일치**.
- **빔 서치의 끝 토큰.** 처음에는 후보마다의 상위 5개 안에 끝 토큰이
  있으면 받아들였고, 글이 10토큰에서 끊겼습니다. 모든 후보를 통틀어
  상위 `width` 안일 때만 받아들이고, 다 쓴 글이 `width`개여도 쓰고 있는
  후보가 더 좋으면 계속하게 고쳤습니다. 본문 33.11절에 실패 그대로
  적었습니다.
- 긴 소리: 크기 기반 VAD(조용한 토막 + 12dB) → 30초 이하의 창. 시험용
  긴 소리가 없어 jfk.wav를 다섯 번 이어 붙여 씁니다(64초, 1.0초에
  받아씀). 같은 연설이 창에 따라 쉼표가 달라집니다.
- whisper.cpp v1.8.7은 `SME = 1 | ACCELERATE = 1`을 찍습니다. 무엇으로
  계산하는지 가려내지 못했습니다. `-ng`로 GPU를 끄고 쟀습니다.
- 인코더를 빠르게 한 것 셋: 선형 층과 어텐션 안의 두 곱을 SME 커널로,
  줄마다의 소프트맥스를 스레드에, 지수 함수를 `exp_fast`로(소프트맥스에
  5,400만 번).
- 남긴 것: 시각 토큰, 30초 창을 시각으로 잇기, 앞 창의 글을 다음 창에
  넣기, 온도를 올려 다시 하기, 낱말 오류율, 한국어, 표본 속도 바꾸기,
  학습한 VAD.
- 도구: `scripts/download_ch33.sh`, `reference/ch33_make_reference.py`,
  `scripts/bench_ch33.sh`(WHISPER_CPP에 빌드한 디렉터리), `ch33_mel`
  (`badak_audio`), `ch33_verify [fast]`, `ch33_bench [fast]`,
  `ch33_transcribe [파일.wav]`.

## 32장 메모 (비전 트랜스포머와 객체 검출)

- 모델 둘 다 **원래 배포처에서** 받습니다(`scripts/download_ch32.sh`).
  ViT-Tiny는 `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`의
  `model.safetensors`(판 `7d3afdd0cf93`, Apache-2.0), YOLOX-Tiny는 YOLOX
  릴리스 `0.1.1rc0`의 `yolox_tiny.onnx`(Apache-2.0). Hub에 올린 것은
  정답뿐입니다(`imagenet-tflite`의 `reference_ch32.safetensors`, 판
  `aaec0043dade`).
- `badak_models::vit::Vit`: 코드로 적은 ViT. `Vit::load(경로, 헤드 수)`,
  `forward(&[f32])`가 `Trace`를 돌려줍니다. `fast()`는 선형 층을 31장의
  SME 커널로, GELU와 소프트맥스를 `nn::exp_fast`로 계산합니다. 늘어놓은
  가중치는 **값의 주소**로 찾습니다(구조체의 주소로 찾으면 모델을 옮길
  때 깨집니다. 한 번 그렇게 짰다가 고쳤습니다).
- 코어에 더한 것: `ops::layer_norm`, `ops::gelu`, `nn::erf`(A&S 7.1.26),
  `nn::gelu`, `nn::exp_fast`(Cephes `expf`의 계수), `nn::silu_in_place`,
  `nn::gelu_in_place`, `nn::softmax_fast_in_place`. 11장의 소프트맥스는
  그대로입니다(같은 비트 시험들이 기댑니다).
- 그래프에 더한 것: `Op::{Concat, Slice, Resize, Reshape, Sigmoid}`
  (`shaping.rs`), `Activation::Silu`, 패스 `merge_duplicates`,
  `swap_reshapes`, `fuse_silu`, 그리고 `sink_transposes`가 `Concat`과
  `Sigmoid`를 지나가게 한 것. `PASSES`가 11개가 됐습니다.
- **ONNX를 옮기는 코드가 마디마다 형상을 정합니다**(`Importer::emit`).
  `Concat`의 축이 4차원에서와 3차원에서 뜻이 달라서입니다. `Slice`의
  "끝까지"(int64 최댓값) 때문에 ONNX 파서가 32비트를 넘는 정수를 자르게
  고쳤습니다.
- YOLOX-Tiny: 마디 279 → 612 → 129. ONNX Runtime과의 차이 1.4e-4.
  `detect::{decode_yolox, overlap, non_max_suppression, COCO}`,
  `Image::letterbox`. 문턱은 점수 0.3, 겹침 0.45.
- **YOLOX의 전처리는 BGR, 0~255 그대로**입니다. 같은 태그의 코드
  (`preproc`)는 정규화를 하지만 릴리스의 파일은 그것과 맞지 않습니다.
  `reference/ch32_preprocess.py`로 넣어 봐서 정했습니다.
- SiLU를 `f32::exp`로 하나씩 하면 YOLOX에서 20ms가 듭니다(SME 커널일 때
  전체의 절반). `exp_fast`로 바꿔 NEON 97.9 → 81.0ms, SME 51.2 → 27.2ms.
- 남긴 것: ViT를 ONNX 그래프로(층 정규화·GELU가 여러 마디로 풀려 나옴),
  ViT의 텐서 할당 줄이기, SME 커널을 스레드에 나누기, 정확도(mAP), 정수
  해석기의 새 연산.
- 도구: `reference/ch32_make_reference.py`, `ch32_preprocess.py`,
  `ch32_time.py`, `scripts/bench_ch32.sh`, `ch32_vit_verify`,
  `ch32_vit_bench [fast]`, `ch32_yolox_verify`, `ch32_bench [neon|sme]`,
  `ch32_detect`(`badak_image`).

## 31장 메모 (모바일 추론 엔진 해부와 대결)

- 비교 엔진: LiteRT 2.3.0, ONNX Runtime 1.32.0, ncnn 1.0.20261020,
  MNN 3.6.2. 모두 Python 패키지 그대로(`reference/requirements_ch31.txt`).
  NCNN과 MNN의 모델은 30장의 ONNX를 `reference/ch31_convert.py`로 바꿔
  `models/vision_mobile/`에 둡니다(Hub에 올리지 않았습니다. 변환기가
  있으면 누구나 만들 수 있습니다).
- **NCNN의 기본값은 16비트**(가중치 저장도, 계산도). `pnnx … fp16=0`과
  `opt.use_fp16_*`를 꺼야 다른 엔진과 같은 계산입니다. 16비트는 1.5배
  빠르고 점수가 0.1(ResNet-50)~1.1(MobileNetV2) 달라집니다.
- 누가 SME를 쓰는가: LiteRT+XNNPACK(KleidiAI), ONNX Runtime(KleidiAI의
  `kai_kernel_matmul_clamp_f32_f32p2vlx1_…`, `objdump`로 확인), MNN(SME2
  어셈블리). NCNN은 NEON뿐(바이너리에 `fmopa` 없음).
- `badak_core::simd::sme`: `asm!`으로 짠 `FMOPA` 커널. `.arch_extension
  sme`를 조각의 첫 줄에 적으면 안정판에서 어셈블됩니다. 망가뜨리는
  레지스터(v0~v31, p0~p15, ffr)를 `streaming!` 매크로가 모두 적습니다.
  스트리밍 모드에는 `linear` 한 번에 한 번만 들어갑니다(묶음마다
  드나들면 작은 층에서 1.3~1.9배 느림). SME 확인은 macOS의
  `sysctlbyname("hw.optional.arm.FEAT_SME")`뿐이고 다른 운영체제에서는
  "없음"입니다. Miri는 어셈블리를 따라가지 못합니다(시험은 `not(miri)`).
- NEON 커널 500억/초, SME 커널 3,000~7,800억/초(7~16배). 행이 24개이면
  2.9배(타일 넷 가운데 둘이 빔), 벡터 하나이면 1.7배.
- `badak_graph::outer::OuterConv2d`: 필터를 한 번 늘어놓아 쥐고
  (`OnceLock`, 주소와 길이로 확인), 창을 사진에서 곧바로 늘어놓습니다
  (`pack_windows`. 보폭 1이고 한 줄 안이면 NEON 전치, 아니면 값 하나씩).
  ResNet-50 스레드 1개: 93.8 → 18.5(펼친 뒤 늘어놓기) → 15.2ms.
- SME는 스레드를 늘려도 덜 늡니다. 커널만 재면 스레드 12개에서 NEON
  10.6배, SME 2.8배. 까닭(SME 장치를 코어들이 나눠 쓰는지)은 확인하지
  못했습니다. MNN은 스레드 12개에서 코어 3개만 쓰고 스레드 4개보다
  느립니다.
- 층 하나짜리 측정(`ch31_layers`)의 스레드 12개 값은 우리 쪽이 유리하게
  나옵니다(스레드가 깨어 있는 채로 되풀이). 본문에 그렇게 적었습니다.
- 한 번, 스레드 12개의 `1 × 1` 합성곱(SME)이 60µs가 아니라 10ms로
  나왔고 재현되지 않았습니다.
- 정수 커널: `dots`(필터 넷씩, `SDOT`)와 `Requantize::apply_all`(한 번
  반올림을 NEON으로). MobileNetV2 int8 스레드 1개 22.4 → 11.5ms, 12개 3.4 →
  2.1ms. 70개 층 비트 일치 그대로. 깊이별 합성곱(두 번 반올림)은
  `vqrdmulh`와 음수의 절반에서 달라 값마다 합니다.
- github.com/microsoft/onnxruntime은 이 환경에서 404여서 소스를 읽지
  못했습니다.
- 도구: `reference/ch31_convert.py`, `ch31_time.py`, `ch31_layers.py`,
  `scripts/bench_ch31.sh`, `ch31_bench <모델> <neon|sme>`
  (`BADAK_PROFILE=1`이면 연산별 시간), `ch31_layers`, `ch31_sme`(코어).

## 30장 메모 (그래프 최적화)

- `badak_formats::protobuf`(꼬리표, varint, 묶어 적기)와
  `badak_formats::onnx`(`ModelProto` → `Model`. 필드 번호는 onnx 1.23.2의
  `onnx.proto`). `raw_data`는 정렬이 맞을 때만 파일을 빌려 씁니다
  (MobileNetV2에서 262개 가운데 47개). 값을 다른 파일에 둔 텐서, 성긴
  텐서, 마디 안의 그래프는 거절합니다. `int64`는 `I32`로 줄여 담습니다.
- `Graph::from_onnx`(`badak_graph::onnx`)는 **기계적으로** 옮깁니다. 축을
  따지는 마디(Conv, BatchNormalization, 풀링)마다 앞뒤에 `TRANSPOSE`를
  두르고, 필터도 `TRANSPOSE` 마디를 거칩니다. 형상은 비워 두고
  `Graph::infer_shapes(&registry)`로 채웁니다. 옮기는 연산은 열네 가지.
- 새 `Op`: `BatchNorm`, `Clip`, `Flatten`, `Identity`. `Padding`이
  `badak_graph`의 것이 되었고 `Explicit([[위, 아래], [왼쪽, 오른쪽]])`이
  생겼습니다. `Axis::new`의 마지막 인자가 축의 번호(0 세로, 1 가로).
- `badak_graph::optimize`: 패스 여덟(`PASSES`). 상수 접기(레지스트리의
  커널로 실행), IDENTITY 지우기, TRANSPOSE 뒤로 옮기기·상쇄, 형상 바꾸기
  줄이기, 배치 정규화 접기, 활성화 함수 붙이기, PAD 합치기.
  `optimize::run(graph, registry, 패스들)`이 아무것도 바뀌지 않을 때까지
  되풀이하고 패스마다 `prune`합니다. 정수 그래프는 건드리지 않습니다.
- **`Interpreter::new`는 최적화도 제자리 계산도 하지 않습니다.** 앞 장의
  숫자가 그대로 나오게 둔 것입니다. `optimize(&mut graph, &registry)`를
  부르고 `Interpreter::in_place(…)`로 만들어야 이 장의 결과가 나옵니다.
- 제자리 계산: `Kernel::overwrites`/`run_in_place`(ADD, MUL, RELU, CLIP,
  BATCH_NORM, FLATTEN, IDENTITY), `interpreter::share`가 수명을 합칩니다.
  덮어쓸 입력은 읽는 창을 만들지 않습니다. Miri:
  `cargo +nightly miri test -p badak_graph --lib optimize`(80초).
- 결과: ResNet-50 448 → 73마디, MobileNetV2 414 → 65. 고치기 전과의 차이
  5e-6, ONNX Runtime과 3e-5. 스레드 12개에서 46.8 → 10.67ms, 15.97 →
  1.66ms. 대부분은 우리가 넣은 `TRANSPOSE`를 없앤 몫. 배치 정규화 접기
  13~24%, 활성화 함수 2~16%. `.tflite`의 `PAD` 합치기: MobileNetV2 2.11 →
  1.64ms, 자리 9.81 → 6.02MB. 제자리 계산: ResNet-50 자리 9.63 → 7.23MB,
  시간은 그대로.
- **2밀리초짜리 측정은 차례마다 20~60% 흔들립니다.** MobileNetV2의 스레드
  12개 줄은 일곱 차례를 쟀습니다(`MODELS=… THREADS=… FIRST=4`).
- ONNX Runtime 1.32.0: 최적화를 꺼도 17.8ms(스레드 1개, ResNet-50),
  "모두"에서 12.4ms. 그 단계에서 `NhwcFusedConv`와 `Transpose`가 생깁니다.
  github.com/microsoft/onnxruntime이 이 환경에서 404여서 소스는 읽지
  못했고 문서(onnxruntime.ai)만 확인했습니다.
- 모델: `resnet50.onnx`, `mobilenetv2.onnx`, `reference_onnx.safetensors`를
  Hub의 `imagenet-tflite`에 추가(판 `97689a166031`). PyTorch의 예전
  내보내기(`dynamo=False`, `training=PRESERVE`,
  `do_constant_folding=False`)로 만들었습니다.
- 남긴 것: 최적화한 그래프의 저장(쓰는 코드가 없음), 같은 계산 합치기,
  합성곱 + 덧셈 합치기, `SAME` + `PAD`, NCHW 커널, 정수 그래프의 패스.
- 도구: `scripts/download_ch30.sh`, `reference/ch30_make_onnx.py`,
  `reference/ch30_time.py`, `scripts/bench_ch30.sh`, `ch30_inspect`,
  `ch30_optimize`, `ch30_verify`, `ch30_bench <onnx|tflite> <모델> <단계>`.

## 29장 메모 (정수 양자화 추론)

- `badak_graph::quantized`: `QuantizedInterpreter`(자리가 `Vec<i8>`. 순서·
  형상·수명·계획은 26장의 함수를 그대로 부름), 고정소수점 곱셈 셋
  (`multiply` 한 번 반올림, `multiply_twice_rounded` gemmlowp식,
  `multiply_half_up` NEON식), 정수 합성곱·깊이별 합성곱·선형 층·덧셈·
  평균·덧대기·축 바꾸기.
- **LiteRT의 자체 커널(builtin)과 비트가 같습니다**: 출력 5,000개와 한
  사진의 중간 텐서 70개 전부(`tests/quantized_tflite.rs`, `ch29_layers`).
  맞추는 데 필요했던 것: 합성곱은 한 번 반올림, 깊이별 합성곱과 평균은
  두 번 반올림, 선형 층은 "두 번, 절반은 큰 쪽으로". 평균은
  `reference/reduce.h`의 `QuantizedMeanOrSum`(배율을 왼쪽으로 옮겨 원소
  수로 정수 나눗셈). 소스에 적힌 규칙이 아니라 층마다 대 봐서 찾은
  것이고, 선형 층의 규칙은 값 하나가 근거입니다.
- XNNPACK 경로의 출력은 builtin과 다릅니다(1,000개 중 460~740개, 최대 9).
- 파서: 텐서의 `Quantization`(눈금·영점·축), `int8` 상수는 `U8` 텐서로
  읽고 쓰는 쪽이 부호를 붙임, **FlatBuffers 밖에 둔 버퍼**(`offset`/`size`)
  지원. `ai-edge-quantizer`가 4MB 파일에도 그 방식을 씁니다.
- 책의 safetensors 읽기 코드는 `I8`을 모릅니다. 정답 파일은 `int8`을
  `u8` 바이트로 저장합니다.
- 모델: `mobilenetv2_int8.tflite`(4.1MB), `reference_int8.safetensors`
  (14MB, 중간 텐서 포함)를 Hub의 `imagenet-tflite`에 추가(판
  `e5c6da2e6729`). 보정은 시험 사진 다섯 장의 조각 45개. 도구가 범위를
  최솟값·최댓값의 이동 평균으로 잡아서 출력의 최댓값이 8.4에 잘립니다
  (잔 95% → 41%, 기타는 동점).
- 속도(M4 Max): 스레드 1개 int8 22.4ms(f32 11.0ms), 12개 3.4ms. LiteRT
  builtin int8 4.0ms, XNNPACK 1.7ms. 우리 정수 커널은 출력마다 하는
  배율 곱하기(스칼라)가 병목(합성곱이 출력 값 하나에 3.5ns). `dot_neon`을
  넣었지만 시간은 그대로였습니다.
- Miri: `cargo +nightly miri test -p badak_graph --lib quantized`(3초).
  NEON 내적은 Miri가 흉내 내지 못해 `cfg(miri)`에서 평범한 반복문을 씁니다.
- 도구: `scripts/download_ch29.sh`, `reference/ch29_quantize.py`,
  `reference/ch29_time.py`, `ch29_verify`, `ch29_layers`, `ch29_bench`.

## 28장 메모 (합성곱 커널 최적화)

- `conv::Method`에 `Tiled`(출력을 조각내어 조각마다 펼치고 곱하고 활성화
  함수까지), `Indirect`(펼치지 않음), `Whole`(27장의 방법)이 생겼고
  `Auto`가 `worth_tiling`으로 `Tiled`와 `Whole` 가운데 고릅니다: 조각
  하나에 벡터가 8개 이상 돌아가면 조각냄. 스레드 12개에서 출력
  `28 × 28` 이상은 조각내기, `14 × 14` 이하는 통째로.
- 조각의 크기는 `BADAK_CONV_TILE`(바이트, 기본 64KB). 16KB~1MB에서 차이
  5% 안.
- 깊이별 합성곱은 출력의 줄 단위로, `ADD`/`MUL`은 4,096원소 조각으로
  스레드에 나눕니다(활성화 함수를 조각마다 이어서).
- `winograd`: `F(2 × 2, 3 × 3)`. 바꾼 필터를 `OnceLock`에 쥡니다. 스레드
  1개에서 채널 128 이상인 층이 1.3~1.9배, 64채널은 0.93배. 스레드
  12개에서는 기본값보다 느려서(13.3 대 11.7ms) 기본값이 아닙니다.
  `winograd::conv_2d_winograd`를 등록하면 씁니다.
- `Indirect`는 자동 벡터화에 맡긴 커널이라 1.5배 느립니다. 채널 3개인
  첫 층은 80억/초.
- `badak_core::simd::half`: FEAT_FP16 명령으로 16비트 계산(Rust 1.94부터
  안정). `f16` 타입은 불안정이라 `vreinterpretq_f16_u16`로 읽습니다.
  1.2~1.9배, 층 하나의 상대 오차 1e-3. 엔진에는 연결하지 않았습니다.
  Miri로는 보지 않았습니다.
- `ch28_layout`: 평범한 반복문으로 NHWC 29~59억, NCHW 99~161억, 채널 묶음
  194~215억/초.
- **측정 스크립트의 함정**: macOS의 `head`에는 `-q`가 없습니다.
  `set -e`인 스크립트가 조용히 죽습니다. 긴 측정은 `caffeinate -i`로
  감싸고(기계가 잠들면 값이 틀어집니다) 알림이 오는 방식으로 돌리십시오.
- 도구: `ch28_layers [모델] [스레드] [whole|tiled|indirect|winograd|auto]`,
  `ch28_half`, `ch28_layout`, `scripts/bench_ch28.sh`.

## 27장 메모 (합성곱과 비전 분류)

- `badak_graph`에 `conv`(`Axis`: 한 축의 기하와 `SAME`/`VALID`. `Conv2d`의
  세 방법 `Direct`/`Im2col`/`Auto`, `DepthwiseConv2d`)와 `spatial`(`Pool2d`,
  `Pad`, `Mean`, `Transpose`)이 생겼습니다. 텐서는 NHWC, 필터는 OHWI
  (깊이별은 `(1, 높이, 너비, 출력 채널)`).
- 설정을 담은 정수 텐서(덧댈 길이, 축, 순서)는 `Graph::from_tflite`가 읽어
  `Op`의 필드로 옮기고 마디의 입력에서 뺍니다. 그래서 그래프의 값 목록에는
  쓰이지 않는 `i32` 상수가 남아 있습니다. 해석기는 마디가 읽는 상수만
  `f32`인지 확인합니다.
- 펼친 행렬은 `arena::with_scratch`에서 빌립니다. 계획 밖입니다.
- 새 크레이트 `badak_image`: JPEG 풀기(`jpeg-decoder` 0.3, 이 책의 둘째
  외부 크레이트), 줄이기 둘(`Resampling::Area`는 Pillow의 `BILINEAR`와
  바이트까지 같음, `TwoTaps`는 교과서의 쌍선형), 가운데 자르기,
  `Preprocess::IMAGENET`.
- **Pillow의 줄이기에는 문서에 없는 규칙이 있습니다**: 세로로 줄어드는
  픽셀 수가 가로의 두 배보다 많으면 세로 패스를 먼저 합니다(`Resample.c`
  863~867번째 줄). 시험 사진으로는 드러나지 않았고, 식으로 만든 사진
  (`tests/pillow.rs`)이 잡았습니다.
- 모델: `timm/resnet50.tv_in1k`(판 `78f3ecfdb38e`, BSD-3)와
  `timm/mobilenetv2_100.ra_in1k`(판 `5afa12513b04`, Apache-2.0)를
  `litert-torch`로 변환. 입력 `(1, 3, 224, 224)`, 맨 앞에 `TRANSPOSE`.
  배치 정규화와 ReLU는 변환기가 접어 넣어 파일에 없습니다. MobileNetV2의
  본래 설정은 bicubic인데 이 장은 두 모델에 같은 쌍선형을 썼습니다.
- 시험 사진 다섯 장은 위키미디어 공용의 CC0 원본, 모델과 정답은 Hub의
  `imagenet-tflite`(`scripts/download_ch27.sh`, 판과 해시 고정). CI가
  받아서 `ch27_verify`와 통합 시험을 돌립니다. 고양이 사진은 두 모델 모두 틀립니다
  (샴고양이, 왈라비). PyTorch와 LiteRT도 같은 답입니다.
- 숫자: 정의 그대로 1,669ms → 펼치기 19.2ms → `1 × 1`은 그대로 18.3ms.
  중간 결과 75.68MB → 9.63MB(하한과 같음. 순서대로는 12.04MB).
  `4032 × 3024` 사진 한 장: 풀기 56.7ms, 줄이기 20.1ms, 모델 18.3ms
  (Pillow는 풀기 27.8ms, 줄이기 13.0ms). 둘만 보고 줄이면 점수가 2.7~4.2
  달라지고 골든 리트리버의 확률이 80% → 28%.
- 도구: `scripts/bench_ch27.sh`(PYTHON에 `requirements_ch27.txt`를 설치한
  파이썬. 이 맥에서는 `python3.12`를 직접 가리켜야 했습니다),
  `reference/ch27_make_tflite.py`, `reference/ch27_time.py`,
  `ch26_inspect vision_tflite/resnet50`.

## 26장 메모 (그래프 해석기)

- 새 크레이트 `badak_graph`: `graph`(값·마디·`Op`, `from_tflite`, 위상
  정렬 `order`), `kernels`(`Kernel` 트레이트: `shapes`와 `run`. 커널은
  받은 자리를 **모두 덮어써야** 합니다. 자리는 0으로 채워져 있지 않습니다),
  `registry`(이름 → 커널을 만드는 함수), `memory`(`lifetimes`, `plan`,
  `lower_bound`, `check`), `interpreter`(`Interpreter::with`가 순서·형상·
  수명·자리를 정하고 `run`은 메모리를 얻지 않습니다).
- `badak_formats`에 `flatbuffers`(표·vtable·벡터를 범위 검사하며 따라감.
  시험용 쓰기 함수 `build`가 `#[cfg(test)]`로 들어 있음)와 `tflite`(스키마의
  필드 번호를 손으로 옮김. 스키마는 LiteRT 커밋 `5d9e2a42e02b`의
  `tflite/converter/schema/schema.fbs`).
- `Interpreter::run`의 `unsafe`는 `check_steps`가 확인한 사실(단계마다
  쓰는 구간이 다른 구간과 겹치지 않음)에 기댑니다. 계획기나 `run`을 고치면
  `cargo +nightly miri test -p badak_graph --lib`를 다시 돌리십시오(2분쯤).
  Miri는 `exp`의 마지막 자리를 흔들므로 소프트맥스를 거친 값은 허용 오차로
  견줍니다.
- 메모리 계획의 기본값은 LiteRT와 같은 "큰 것부터"(`Strategy::BySize`).
  잔차 MNIST 모델에서는 "순서대로"가 이기고(1,040 대 1,296원소), Llama
  블록에서는 "큰 것부터"가 이깁니다(27.26 대 29.36MB). 어느 쪽도 최적을
  보장하지 않습니다.
- 22.13절의 숙제는 **계산까지만** 했습니다(`ch26_llama_block`: 45.61MB →
  27.26MB, 22장 때의 블록은 54.00MB로 22.11절의 측정과 일치). Llama 엔진은
  코드로 적은 엔진으로 남습니다. 22.13절, 23.12절, 24.11절, 25장 끝의
  문장을 그에 맞게 고쳤습니다.
- 숫자(M4 Max): 준비 18µs(`mlp`)·36µs(`residual`), 한 장 4.33µs·44.0µs,
  만 장을 한 번에 장당 0.33µs·2.83µs, 마디 하나의 수고 6~9ns, 실행 중 할당
  0번. LiteRT(Python `invoke`, XNNPACK) 1.96µs·14.2µs. LiteRT의
  `residual`은 14~20µs를 오갑니다.
- **LiteRT가 빠른 까닭을 `sample`로 확인했습니다**: 그래프를 XNNPACK에
  통째로 넘기고(델리게이트), XNNPACK은 KleidiAI의 SME2 커널
  (`…1x32__neonsme2`)을 부릅니다. 이 맥은 `hw.optional.arm.FEAT_SME2: 1`.
  우리 NEON 커널은 벡터 하나일 때 220~260억 곱셈/초, 여덟일 때
  430~510억입니다(`ch26_single_vector`). 벡터 하나에 여덟 행을 묶는 실험은
  효과가 없어서 버렸습니다. 25.9절의 Accelerate 격차도 같은 까닭일 수
  있으나 확인하지 않았습니다.
- 도구: `scripts/download_ch26.sh`, `scripts/bench_ch26.sh`(PYTHON에
  `reference/requirements_ch26.txt`를 설치한 파이썬), `reference/
  ch26_make_tflite.py`, `reference/ch26_time_litert.py`. CI가 `ch26_verify`를
  돌립니다.
- 링크 도구의 함정: "사진 16장"이 16장(章)으로 연결됩니다. 사진의 수는
  "열여섯 장"처럼 적으십시오.

## 24장 메모 (어텐션)

- KV 캐시는 키·값 헤드마다 따로 담습니다(`LayerCache.keys: Vec<Store>`).
  한 헤드의 키들이 나란히 놓여 `simd::linear_with`에 "가중치"로 들어갑니다.
  `Store::scores`(질의 여럿 × 앞의 키들), `Store::mix`(값을 비중대로 섞기).
- **점수의 내적이 17.3절의 정의로 바뀌었습니다**(`simd::dot`). 9장의
  `attention`도 같은 정의를 씁니다. 정답과의 상대 오차 5.9e-7 → 6.9e-7.
  이전 장들의 출력 파일에 든 점수의 마지막 자리가 지금과 다를 수 있습니다.
- `simd::Weight`를 `F16`에도 구현했고(ARM은 한 명령으로 넷씩, x86은 하나씩),
  `simd::widen`이 값들을 f32로 넓힙니다. 16비트 KV 캐시가 f32와 같은 속도.
- 일의 단위는 (새 토큰, 키·값 헤드). 새 토큰이 하나여도 스레드에 나뉩니다.
- `ops::zip`(함수에 `Sync` 필요), `ops::rms_norm`, `rope::apply`,
  `Activations::fill`이 스레드를 씁니다. MLP의 `silu`와 `mul`은 `zip` 하나로
  합쳤습니다(같은 비트).
- 문맥 4,096토큰에서 토큰 생성 17.5 → 76.4토큰/초(f32 캐시), 8.4 → 74.4
  (16비트). 어텐션 46.2 → 1.9ms.
- 토큰 하나의 시간이 실행마다 10.3ms와 11.8ms 사이를 오갑니다(빈 문맥,
  23장의 엔진도 같음). 원인을 찾지 못했습니다.

## 23장 메모 (GPU)

- `crates/badak_gpu`: 엔진에서 유일하게 외부 크레이트(`wgpu` 30)를 씁니다.
  제안서의 결정 2번 그대로입니다. 실행기(`block_on`)와 바이트 변환은 직접
  짰습니다. GPU가 없으면 `Gpu::new()`가 `None`이고 테스트는 건너뜁니다.
- `Gpu::new()`는 표준 밖의 기능 셋을 켭니다(있을 때만): 계산에 쓰는 버퍼를
  CPU가 곧바로 열기(`MAPPABLE_PRIMARY_BUFFERS`), GPU의 시계(`TIMESTAMP_QUERY`,
  이것은 표준의 선택 기능), 행렬 연산(`EXPERIMENTAL_COOPERATIVE_MATRIX`).
  `Gpu::standard()`는 브라우저와 같은 조건입니다.
- 선형 층 커널(`linear::Variant`): Simple, Wide, Exact, Block, Cooperative.
  256×8192×2048에서 3,228 / 5,218 / 5,778 / 20,600 / 44,852억 곱셈/초(CPU
  5,654). **셰이더에서 배열과 안쪽 반복문을 쓰면 wgpu의 범위 검사 때문에
  3분의 1로 느려집니다. 풀어 적으십시오.**
- **버퍼를 읽기와 쓰기를 모두 열 수 있게 만들지 마십시오.** wgpu가 쓰기
  전용으로 빠른 메모리에 두어 읽기가 23배 느려집니다. 입력은 `Gpu::input`,
  결과는 `Gpu::zeros`. 값을 담아 버퍼를 만든 뒤에는 빈 명령을 보내야 임시
  버퍼가 쌓이지 않습니다(`Gpu::upload`가 합니다).
- GPU의 시계는 명령 묶음을 보낸 뒤 **따로** 꺼내야 맞는 값이 나옵니다
  (`run_timed`). 질의 묶음은 하나를 돌려 씁니다.
- GPU의 결과는 CPU와 비트가 같지 않습니다(wgpu가 Metal의 빠른 수학을 끄지
  않음). 같은 계산을 두 번 하면 같습니다. GPU 테스트는 허용 오차로 견줍니다.
- 모델 연결: `badak_core::packed::Matrix::Delegated`(곱셈을 맡긴 행렬),
  `ops::Operator`(텐서 하나를 받아 하나를 내는 계산), `BlockWeights::
  mlp_override`, `LlamaModel::override_mlps`. `badak_gpu::offload::Offload`가
  행렬(`matrix`)과 MLP 전체(`mlp`)를 GPU에 올립니다. 벡터가 `min_count`(32)
  보다 적으면 CPU에서 계산합니다.
- 결과(256토큰 프롬프트 뒤 32토큰): CPU 412/71, 표준 WebGPU만 926/15, 표준
  밖의 기능도 1,716/19, MLP를 통째로 2,116/22, 32개부터만 GPU 1,907/77
  토큰/초. llama.cpp Metal은 5,532/90.
- 가중치는 f32로 펼쳐 올립니다(3.89GB, 2.6초). 지켜야 하는 메모리 4.2GB.
  BF16/양자화 가중치를 GPU에서 그대로 쓰는 일, 어텐션을 GPU에 올리는 일,
  계산 전체를 한 번에 보내는 일은 하지 않았습니다(26장의 그래프에서).
- WGSL 명세의 부동소수점 조항(15.7절)은 읽지 못했습니다. 본문에서 그렇게
  밝혔습니다.

## 2부의 모델

| | Llama 3.2 1B Instruct (주) | SmolLM2 135M Instruct (보조) |
|---|---|---|
| 내려받기 | `scripts/download_llama32.sh` (Meta 승인과 토큰 필요) | `scripts/download_smollm2.sh` |
| 위치 | `models/llama32_1b/` | `models/smollm2_135m/` |
| 자동 테스트 | 로컬에서만 (파일이 없으면 건너뜀) | CI에서 실행 |
| 정답 파일 | Hub의 `llama32-1b-reference` | Hub의 `smollm2-135m-reference` |

- 저자가 2026-11-17에 Llama 3.2 승인을 받았고, 같은 날 주 모델을 Llama로
  정했습니다. 본문의 숫자와 출력은 Llama의 것이고, SmolLM2는 승인 없이 따라
  하는 독자와 CI를 위한 것입니다.
- **Llama 3.2의 정답 파일도 공개했습니다(2026-11-17, 저자 결정).** Llama 3.2
  Community License 원문을 확인한 결과 재배포는 조건부로 허용됩니다(1.a,
  1.b항). 조건은 라이선스 사본 동봉, `NOTICE` 파일의 고지문, "Built with
  Llama" 표시이고, Hub 저장소 `llama32-1b-reference`에 셋 다 넣었습니다.
  정답 파일의 `embed`는 가중치의 일부(행 11개)이고 나머지는 계산 결과입니다.
  유료 출판 전에는 이 부분을 전문가에게 확인받기로 했습니다.
- 책에 싣는 출력에는 Llama의 가중치 값이 드러나지 않게 합니다(텐서 목록은
  이름과 형상만).

## 2부 진행 메모

- 8장은 모델 가중치 없이 llama.cpp 저장소의 단어 사전 전용 GGUF로
  진행했습니다(`scripts/download_ch08.sh`). Llama 3의 토크나이저가 거기
  들어 있어 라이선스 동의 없이 쓸 수 있습니다.
- llama.cpp는 `scripts/build_llama_cpp.sh`로 `third_party/llama.cpp`에
  빌드합니다(Git이 무시, 커밋 `b4c0e00a4fcc` = 태그 `b12428`). 지금은
  `llama-tokenize`만 빌드해 두었습니다.
- 대화 형식은 Jinja 해석기 없이 Llama 3와 ChatML 두 가지를 코드로
  적었습니다. 뼈대와 사용자 글을 따로 토큰화하는 일은 37장으로 미뤘습니다.
- BF16은 읽을 때 f32로 펼칩니다(9.3절). 복사가 일어나고 메모리가 두 배
  듭니다.
- 어텐션은 토큰마다, 헤드마다 내적을 직접 하는 단순한 구현입니다. 행렬 곱을
  쓰지 않습니다. 배치(여러 글을 한 번에)는 아직 없습니다.
- 10장부터 토크나이저는 모델 디렉터리의 `tokenizer.json`에서 읽습니다
  (`Tokenizer::from_tokenizer_json`). 사전 분할 규칙은 실행하지 않고 어느
  규칙인지 알아보기만 합니다(GPT-2, 숫자를 낱개로 떼는 GPT-2, Llama 3).
  글 앞에 붙이는 토큰은 `post_processor`의 틀에서 읽습니다
  (`Tokenizer::prefix`). `config.json`의 `bos_token_id`를 무조건 붙이면
  SmolLM2의 답이 바뀝니다(10.6절).
- 9장의 정답 파일은 시작 토큰 없이 만든 글(토큰 11개)입니다. 검증에는
  지장이 없습니다. `hidden_states`의 마지막 원소는 마지막 정규화를 거친
  값입니다(10.7절).
- 블록을 이어 대조할 때의 허용 오차는 상대 `1e-4`이고, 첫 토큰을 뺀 나머지를
  따로 한 번 더 비교합니다. 첫 토큰의 한 자리에 아주 큰 값이 있어서입니다.
- `[profile.test]`에 `opt-level = 2`를 넣었습니다. 모델 전체를 도는 테스트
  때문입니다.
- 11장: KV 캐시는 `llama/cache.rs`에 따로 있습니다. 9장의 `attention`과
  10장의 `block`은 그 장의 설명과 맞도록 고치지 않았고, 캐시를 쓰는
  `attention_cached`, `block_cached`가 같은 계산을 한 벌 더 갖고 있습니다.
  두 벌이 비트 단위로 같은지는 `tests/llama_generate.rs`가 지킵니다. 16장에서
  행렬 곱을 바꾸면 이 비교를 허용 오차가 있는 것으로 바꿔야 할 수 있습니다.
- 캐시는 `Vec`에 덧붙이는 방식이고 최대 길이를 확인하지 않습니다
  (`max_position_embeddings`를 아직 읽지 않음). 22장에서 고칩니다.
- 대화 예제는 형식을 단어 사전의 특수 토큰으로 알아봅니다(`<|im_start|>`가
  있으면 ChatML). 시스템 프롬프트는 넣지 않습니다.
- 12장: 난수 생성기는 `badak_core::random::Pcg64`(NumPy의 PCG64와 같은 수),
  샘플러는 `badak_models::sampling`입니다. 손질의 순서는 transformers와 같고
  (벌점, 온도, top-k, top-p), `temperature_last`로 llama.cpp처럼 온도를 뒤로
  보낼 수 있습니다. 단계와 순서가 코드에 박혀 있어, 단계가 늘면 사슬
  구조로 바꿔야 합니다. `generation_config.json`은 아직 읽지 않습니다.
- 12장에서 2장의 `prob::softmax`와 `log_softmax`가 쌍별 합산을 쓰도록
  고쳤습니다. 값 12만 개를 차례로 더하면 상대 오차가 `4e-4`까지 났습니다.
  이 때문에 10장의 확률 출력을 다시 뽑았습니다(69.99% → 69.95%).
  PyTorch의 f32 소프트맥스에도 `1.4e-4`의 오차가 있어, 샘플링의 정답 파일은
  f64로 계산합니다.
- 11장의 `generate`는 그대로 두고 `generate_sampled`를 따로 두었습니다.
- 22장: 메모리. `badak_bench::alloc::Counting`(전역 할당자 자리에 끼우는 세는
  장치)과 `Snapshot`. 예제가 `#[global_allocator]`로 직접 등록합니다.
- 양자화한 가중치는 `llama::pack`으로 `model_q8_0.safetensors`,
  `model_q4_0.safetensors`에 적어 두고(행렬은 `(행, 행의 바이트)` 모양의 `U8`
  텐서, 이름은 원본과 같음) `LlamaFiles::with_format`이 있으면 비춰 씁니다.
  없으면 20장처럼 읽을 때 양자화합니다. **모델 디렉터리에 이 파일이 있어야
  빠릅니다**(`cargo run --release -p badak_models --example ch22_pack`).
  `Packed`의 바이트는 이제 `u8` 텐서(`storage`)에 담깁니다.
- `ThreadPool::run_chunks`는 잠금 없이 나눕니다(세 번째 `unsafe`). 18장의
  잠금 판은 `run_chunks_with_locks`로 남겼습니다(18장 본문이 인용).
  macOS에서 `Mutex`는 처음 잠글 때 힙에 할당합니다.
- `prob::softmax_in_place`(`softmax`와 같은 비트), `KvCache::reserve`
  (`generate`가 부름). 어텐션은 토큰마다 벡터 하나를 헤드들이 돌려 씁니다.
- 할당: 프롬프트 256토큰 426,100 → 22,611 → 552 → 40번, 새 토큰 하나 7,493 →
  2,067 → 520 → 8번(22.8절, 22.10절, 아레나). 토큰 생성 +2.5~7%, 프롬프트 처리
  ±1%, 아레나는 지켜야 하는 메모리 +20MB.
- `badak_core::dims::Dims`: 형상과 스트라이드를 텐서 안에 담는 짧은 목록
  (넷까지 안에, 그보다 많으면 힙). `&[usize]`로 읽힙니다.
- `badak_core::arena`: 스레드마다 하나인 아레나. `arena::scope(일)` 안에서
  `Tensor::build`로 만드는 f32 텐서가 덩어리에서 자리를 얻습니다. 덩어리를
  되돌려도 되는지는 `Arc::get_mut`이 확인하므로 구간 밖으로 텐서를 들고
  나가도 틀리지 않고 느려질 뿐입니다. 밖으로 꺼낼 때는 `Tensor::assign`(가진
  자리에 덮어쓰기)이나 `detached`. `hidden_cached`가 블록마다 구간을 엽니다.
  `BADAK_ARENA=off`로 끕니다. **새 연산자는 `Tensor::from_vec` 대신
  `Tensor::build`로 결과를 만드십시오.**
- `arena::with_scratch(len, 일)`: 스레드마다 돌려 쓰는 f32 자리. **지난 값이
  남아 있습니다.** `Activations::fill`도 같은 방식.
- 아레나, 텐서, `Dims`, 스레드 풀의 테스트는 Miri로도 통과합니다
  (`cargo +nightly miri test -p badak_core -- arena tensor:: dims:: threads::`).
  `unsafe`를 고치면 다시 돌리십시오. Miri에서는 `arena::INITIAL_CAPACITY`가
  작습니다(`cfg!(miri)`). `mmap`과 SIMD 코드는 Miri로 보지 않았습니다.
- **속도를 견줄 때는 고치기 전과 뒤를 번갈아 돌립니다.** 따로 쟀더니 없는
  10% 차이가 보였습니다(22.8절). 고치기 전의 엔진은 `git worktree`로 받습니다.
- 메모리 숫자는 둘을 함께 적습니다: 올라온 메모리(resident)와 지켜야 하는
  메모리(footprint). `scripts/measure_ch22_memory.sh`.
- 21장: `badak_models::llama::perplexity`. `token_loss`(f64로 모음),
  `Losses`(토큰마다의 손실을 버리지 않음. `standard_error`, 짝지은 비교
  `difference_from`), `evaluate`(llama.cpp와 같은 방법: 512토큰씩, 조각마다
  글머리 토큰, 뒤 절반의 255자리만 채점). **llama.cpp와 일치**: 64조각 15.302
  ± 0.341 대 15.3021 ± 0.34088, 글 전체 14.001 ± 0.104 대 14.0009 ± 0.10411.
- 평가 글은 WikiText-2 시험 부분(`scripts/download_wikitext2.sh`, llama.cpp와
  같은 파일, SHA-256 고정, 저장소에 넣지 않음). 289,077토큰. 조각 하나 1.5초,
  전체 15분. 방법을 견줄 때는 앞의 64조각과 짝지은 비교를 씁니다.
- 손실 증가(64조각, 원래 모델 대비): Q8_0 가중치만 +0.0003(오차 안), 입력도
  8비트 +0.0013, Q4_K_M +0.032, Q4_0 +0.185, 19장의 4비트 +0.207, 텐서마다
  8비트 +0.119, 텐서마다 4비트는 퍼플렉서티 1,600만.
- 민감도(하나만 Q4_0으로): `down_proj` +0.068이 가장 크고 값당으로는 `v_proj`.
  블록 1이 +0.045로 유난함(원인 미확인). 따로 잰 것의 합(0.146) < 모두(0.185).
- `LlamaModel::hidden_cached`(새 토큰 전부의 마지막 은닉 벡터), `KvPrecision`,
  `packed::dequantize_q6_k`. `ch21_perplexity`의 `from_gguf`가 Q4_K_M 파일의
  가중치를 f32로 되돌려 우리 모델에 넣습니다(q/k 행 순서 복원 포함).
- 토크나이저의 `bos()`는 `with_bos_eos`로 넣어 줘야 채워집니다. 글머리 토큰은
  `prefix().first()`로 얻으십시오(21.3절의 버그).
- 20장: 가중치를 줄인 채로 계산합니다. `simd::Weight` 트레이트(`f32`, `Bf16`)로
  17장의 커널이 BF16을 읽는 순간에만 f32로 바꿉니다(결과는 비트 단위로 같음).
  `badak_core::packed`: `Format::{Q8_0, Q4_0}`(llama.cpp와 같은 바이트. 20.3절에서
  13억 바이트를 대조), `Packed`, `Matrix`(Dense | Packed), `Activations`(입력을
  곱하기 직전에 32개마다 8비트로), 기준 구현 `dot_q8_0`/`dot_q4_0`, ARM의
  `packed/neon.rs`(`four`: 내적 넷을 레지스터 네 칸에, `piece`), x86의
  `packed/avx2.rs`. `ops::LinearWeight`로 `ops::linear`가 가중치의 저장 방식에
  맞는 커널을 고릅니다. `simd::split_rows`가 행 나누기를 형식과 무관하게 합니다.
- **정수 내적의 정의**: 묶음마다 정수 32쌍의 곱을 정수로 더하고,
  `products as f32 * (w_scale * x_scale)`를 f32 합에 차례로 더합니다(FMA 아님).
  입력의 눈금은 f32(llama.cpp는 f16), 반올림은 짝수 쪽.
- `LlamaModel::load`의 기본값이 **파일에 저장된 그대로(BF16)**가 되었습니다.
  `BADAK_WEIGHTS=f32|stored|q8_0|q4_0`으로 바꿉니다. `load_as`로 직접 정할 수도
  있습니다. `LlamaFiles`의 `embedding()`/`block()`은 여전히 f32(9장의 코드와
  테스트용)이고 `with_format`으로 바꿉니다. 모델의 행렬 가중치 형은 `Matrix`.
- 토큰 생성 스레드의 기본값을 6에서 8로 올렸습니다(20.7절).
- Q8_0/Q4_0은 **읽을 때마다 양자화**합니다(1.2초, 메모리 최대치 2.5GB/2.0GB).
  양자화한 파일을 두고 비춰 쓰는 일은 22장으로 넘겼습니다. 우리 묶음이 GGUF와
  바이트가 같으므로 GGUF를 그대로 읽는 길도 있습니다(q/k의 행 순서만 다름).
- 20.5절에 적은 커널의 앞선 두 판(0.124, 0.082사이클)은 저장소에 없습니다.
  측정값만 본문에 있습니다.
- Rust 안정판에서 `f16` 형과 `vmmlaq_s32`(i8mm)는 쓸 수 없습니다(1.100 기준,
  이슈 116909, 117223). `vdotq_s32`(dotprod), `vcvt_f32_f16`(fp16)은 쓸 수 있습니다.
- llama.cpp 비교: `scripts/quantize_gguf.sh`, `scripts/bench_llama_cpp_quantized.sh`
  (`bench/ch20/`). 순수 CPU 빌드의 BF16은 너무 느려 재지 못했습니다.
- 19장: `badak_core::quant`. 16비트 실수 변환(`f16_from_f32` 등, 가까운
  값으로, 한가운데는 짝수 쪽. NumPy·PyTorch와 대조하고 65,536개 전수 확인),
  담아 두는 형 `F16`/`Bf16`, 정수 눈금 `Quantizer`(대칭은 `-(2^(b-1)-1)..`로
  정수 하나를 버림, 비대칭은 0을 정확히 지킴), `fake_quantize`, `sqnr_db`.
- 텐서에 `DType::F16`, `DType::Bf16`을 넣었습니다(4.2절, 6.4절의 약속).
  `SafeTensors::tensor_as_stored`가 펼치지 않고 돌려주고, `tensor`는 여전히
  BF16을 f32로 펼칩니다. 엔진(`LlamaModel::load`)은 아직 펼쳐 읽습니다.
- `LlamaModel::matrices`와 `replace_matrices`로 행렬 가중치(임베딩 + 블록당
  7개)를 훑거나 바꿔 끼웁니다. 21장의 구성 요소별 민감도에 쓸 수 있습니다.
- 19장의 측정(Llama 3.2 1B): BF16으로 바꾸면 달라지는 값 0개, f16은 29만 개
  (2^-17 미만의 값). 가중치의 가장 큰 값은 텐서 전체로 표준편차의 10~32배,
  행 하나로 4배. SQNR은 4비트가 텐서/행/32개마다 2.2/15.1/20.0dB, 8비트가
  24.5/40.2/45.1dB. 흉내 양자화한 모델은 8비트(32개마다)에서 점수 오차 1%,
  4비트(32개마다)에서 " Paris" 57.7%, 4비트(텐서마다)에서 망가짐.
- 책에는 Llama 가중치의 값을 싣지 않았습니다(종류별 통계와 모델의 출력만).
- 18장: 스레드 풀은 `badak_core::threads`(표준 라이브러리만). `ThreadPool::run`이
  번호 `0..count`를 스레드들이 하나씩 뽑아 가게 하고, `run_chunks`가 슬라이스를
  조각내어 나눕니다. 일꾼은 `call`(일의 번호 + 참여할 일꾼 수를 한 원자적 값에
  담은 것)을 보고 움직이고, 참여하지 않는 일꾼은 잠든 채로 둡니다. 풀이 이미
  일하고 있으면(`busy`) 맡긴 쪽이 혼자 합니다. `unsafe`는 수명을 지운 함수
  포인터 하나이고, `Wait`의 `Drop`이 "일꾼이 끝나기 전에 `run`이 돌아가지
  않는다"를 지킵니다. 풀의 크기는 64가 한계(`asleep` 비트마스크).
- 엔진 전체가 `threads::global()` 하나를 씁니다. 크기는
  `available_parallelism`(이 기계 16) 또는 `BADAK_THREADS`. 입력 벡터가
  하나면 스레드 6개(`DEFAULT_GENERATION_THREADS`), 여럿이면 전부.
  기다리는 방법은 "0.2ms 양보하며 기다리다 잠듦"(`BADAK_WAITING=sleep|spin`으로
  실험). 성능 코어 수를 표준 라이브러리로 알 수 없어 프롬프트 처리의 기본값이
  "전부"입니다. SmolLM2에서는 12개가 더 빠릅니다(18.10절).
- `simd::linear`가 스레드에 나눕니다. **가중치의 행을 나눕니다**(스레드당 4조각,
  `PIECES_PER_THREAD`). 벡터가 여럿이면 조각별 결과를 따로 모았다가 옮겨
  적습니다. 곱셈 2^18번 미만은 나누지 않습니다. `linear_with`가 17장의
  스레드 하나짜리 함수입니다. `attention_cached`는 새 토큰별로, `ops::map`은
  원소 4,096개부터 나눕니다. `zip`, `rms_norm`, `rope`는 나누지 않았습니다.
- **스레드 수와 무관하게 선형 층의 결과가 비트 단위로 같습니다**(테스트
  `any_number_of_threads_gives_the_same_bits`). 합이 아니라 내적을 나눴습니다.
- 프로파일러는 고치지 않았습니다. 구간은 맡기는 스레드에서만 엽니다. 일꾼이
  실행하는 함수 안에는 `profile::scope`를 두지 마십시오(기록이 빠집니다).
  `ch14_profile`은 토큰 수를 인자로 받습니다(`-- 모델 256 0`).
- **코어를 모두 쓰는 측정은 처음 몇 초가 빠릅니다**(pp256이 처음 415~445,
  그 뒤 360~390토큰/초). 원인 미확인(열 또는 전력 관리로 짐작). 기준 측정은
  되풀이 15번의 중앙값을 씁니다. 토큰 생성은 흔들리지 않습니다.
- 임시 브랜치 CI로 리눅스 x86에서 테스트 통과를 확인했습니다. macOS 러너는
  대기열이 길었습니다.
- 17장: 엔진이 쓰는 커널은 `badak_core::simd`. **계산의 정의는
  `simd::dots_portable`**입니다(합 8개, `mul_add`, `reduce`의 순서). NEON
  (`simd/neon.rs`)과 AVX2(`simd/avx2.rs`) 구현은 그와 같은 비트를 내야 하고
  테스트가 확인합니다. 구현은 실행할 때 고릅니다(`Backend::detect`).
  반복문은 `drive!` 매크로 하나를 구현마다 찍어 냅니다. 16장의 `kernel`
  모듈은 그대로 남아 있지만 엔진은 더 이상 쓰지 않습니다.
- 가중치를 걷는 방법은 벡터 수에 따라 셋입니다(`Walk`): 1개면 먼 네 줄기,
  2~7개면 이웃 두 행, 8개부터 이웃 네 행. 커널 모양은 입력 둘 × 가중치 넷.
- **x86(AVX2) 코드는 이 맥에서 컴파일도 실행도 못 합니다.** CI의 리눅스
  기계에서 테스트로 정확성을 확인하고, CI가 `ch17_backend` 예제로 선택된 구현과
  결과의 지문을 로그에 남깁니다(NEON과 AVX2가 `7ca8a60337244d8d`로 같음).
  x86 코드를 고칠 때는 임시 브랜치에 올려 `gh workflow run CI --ref`로 확인한
  뒤 본 커밋에 넣습니다. x86에서의 속도는 재지 않았습니다(다음 판).
- "같은 비트" 약속의 범위는 선형 층의 커널입니다. 지수 함수(`silu`,
  소프트맥스)는 운영체제의 수학 라이브러리에 달려 있어 모델 전체의 출력이
  기계마다 같다고는 약속하지 않습니다.
- 16장: 빠른 커널은 `badak_core::kernel`. `dots::<M, N, L>`이 내적 `M × N`개를
  합 `L`개로 나누어 계산합니다. 엔진은 `LANES = 16`, `ROWS = 4`를 쓰고, 입력
  벡터가 5개(`MIN_VECTORS_FOR_ROWS`) 미만이면 가중치를 한 행씩 읽습니다.
  `ops::linear`와 `ops::matmul`이 `kernel::linear`를 부릅니다. 2장의
  `linalg`(`dot`, `matvec`, `matmul`)와 3장의 `nn`은 그대로입니다.
- **내적 하나는 어떤 묶음으로 계산해도 같은 순서로 계산됩니다.** 그래서 캐시를
  쓴 결과가 여전히 비트 단위로 같고 테스트를 고치지 않았습니다. 이 약속 때문에
  `3 × 4, 합 8개` 모양을 쓰지 않았습니다. 17장(FMA), 18장(스레드), 23장(GPU)에서
  이 약속을 어디까지 지킬지 다시 정해야 합니다.
- 16장 뒤의 기준: Llama 3.2 1B pp256 27.8, tg32 14.0토큰/초
  (`bench/baseline/*_ch16.txt`). 이 측정에서 클록이 4.0GHz로 찍혔습니다
  (14장 기준은 4.35). 원인 미확인.
- 블로킹(`linear_blocked`)은 스레드 1개에서 효과가 없어 기본값은 끊지
  않음입니다. 18장에서 다시 잽니다.
- 15장: 기준 기계의 한계. 메모리 읽기는 스레드 1개 83GB/초, 12개
  275GB/초(4개에서 239). 계산은 스레드 1개가 1초에 197억 곱셈(합을 16개로
  나눈 내적, L1 안에서). 이 숫자가 `ch15_budget`의 기본값입니다. 토큰
  생성의 상한은 스레드 1개 15.9, 12개 55.6토큰/초이고 llama.cpp는 그
  81~94%에 있습니다. 지금의 `badak`은 메모리가 아니라 덧셈의 의존 사슬에
  막혀 있습니다(행렬 크기와 무관하게 곱셈당 2.0사이클).
- 14장: 측정 도구는 `crates/badak_bench`(`measure`, `Summary`, `Counters`,
  `environment`)이고, 엔진 안의 계측 프로파일러는 `badak_core::profile`입니다.
  `profile::scope`가 5장의 연산자 다섯(`linear`, `rms_norm`, `embedding`,
  `map`, `zip`)과 `mlp`, `attention_cached`, `rope::apply`, `logits`의 첫 줄에
  들어갔습니다. 스레드마다 기록을 따로 두므로 18장에서 손봐야 합니다.
  표본 프로파일은 `scripts/profile_sample.sh`(macOS의 `sample`).
- 하드웨어 카운터는 macOS에서만 읽습니다(명령과 사이클). 리눅스는 `None`.
- 기준 측정은 `bench/baseline/`에 있고 `ch14_baseline`으로 다시 잽니다.
  Llama 3.2 1B: pp256 2.16토큰/초, tg32 1.76토큰/초. SmolLM2 135M: pp256
  24.5, tg32 19.7. **전원을 연결하고 잰 값입니다.** 배터리로 잰 기록은
  `_battery` 파일로 남겼고 생성 속도가 2~4% 낮습니다(14.11절). 엔진을 고친
  뒤 견줄 때는 전원을 연결한 상태에서 잽니다.
- 14장에서 `prob::argmax`를 두 번에 나누는 구현으로 바꿨습니다(14배 빠름).
- 13장: llama.cpp 비교 도구는 `scripts/build_llama_cpp.sh`(`--plain`은
  Metal·Accelerate를 뺀 판), `scripts/convert_to_gguf.sh`,
  `scripts/build_ch13_tools.sh`, `scripts/bench_llama_cpp.sh`,
  `bench/ch13/llama_logits.cpp`입니다. 변환기에는 `sentencepiece`가 필요합니다.
  측정 기록은 `bench/ch13/`에 있습니다. 이 비교는 CI에서 돌지 않습니다.
- 회전의 짝짓기: 우리는 PyTorch 방식(앞 절반과 뒤 절반), GGUF는 이웃한 둘.
  변환기가 `q_proj`, `k_proj`의 행을 재배열합니다(13.2절). GGUF의 가중치로
  `badak`을 돌리려면 회전을 이웃 짝 방식으로 하거나 행을 되돌려야 합니다.
  아직 GGUF에서 모델을 읽어 실행하지는 않습니다(20장에서 필요).
- 엔진을 견줄 때의 기준은 f64로 계산한 점수입니다(`logits.safetensors`).
  llama.cpp는 플래시 어텐션을 끄고 캐시를 f32로 두어야 `1e-6` 수준으로
  일치합니다. 기본 설정은 `1.6e-4`.

## Hugging Face Hub에 올린 것

조직: <https://huggingface.co/building-an-inference-engine-from-scratch>

| 저장소 | 장 | 판 | 내용 |
|---|---|---|---|
| `mnist-mlp` | 7 | `ff9e117d7f1f` | `model.safetensors`, `reference.safetensors` |
| `imagenet-tflite` | 27 | `fd225d824adb` | `resnet50.tflite`(원본 BSD-3-Clause), `mobilenetv2.tflite`(원본 Apache-2.0), `labels.txt`, `reference.safetensors`, 원본의 라이선스 파일 둘. 남의 가중치를 형식만 바꾼 것 (저자 승인 2026-11-19) |
| `imagenet-tflite` (추가) | 29 | `e5c6da2e6729` | `mobilenetv2_int8.tflite`, `reference_int8.safetensors` |
| `imagenet-tflite` (추가) | 30 | `97689a166031` | `resnet50.onnx`, `mobilenetv2.onnx`(27장과 같은 가중치, 최적화를 끄고 내보낸 것), `reference_onnx.safetensors` |
| `imagenet-tflite` (추가) | 32 | `aaec0043dade` | `reference_ch32.safetensors`(정답만. 모델 파일은 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 33 | `310b423c9eb9` | `reference_ch33.safetensors`(정답만. 모델과 소리는 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 34 | `d0cc98f26772` | `reference_ch34.safetensors`(정답만. 모델과 글은 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 35 | `88881548b761` | `reference_ch35.safetensors`(정답만. 모델은 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 36 | `632124d754f7` | `reference_ch36.safetensors`(정답만. 모델은 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 37 | `5e2f60b70a58` | `reference_ch37.safetensors`(정답만. 모델과 디코더는 원래 배포처에서 받음) |
| `imagenet-tflite` (추가) | 38 | `219c7314a294` | `reference_ch38.safetensors`(정답만. 사슬을 따라 쓴 토큰들) |
| `mnist-tflite` | 26 | `9bb2098554f5` | `mlp.tflite`, `residual.tflite`, `reference.safetensors`(LiteRT의 확률과 예측) |
| `smollm2-135m-reference` | 9~13 | `1a490305d063` | `reference.safetensors`, `tokenizer_vectors.json`, `generation.json`, `sampling.safetensors`, `logits.safetensors` (가중치 없음. 원본은 Apache 2.0) |
| `llama32-1b-reference` | 9~13 | `01b2f0fcedd3` | `reference.safetensors`, `tokenizer_vectors.json`, `generation.json`, `sampling.safetensors`, `logits.safetensors`, `LICENSE.txt`, `NOTICE`, `USE_POLICY.md` (Llama 3.2 Community License) |

올리는 방법: 저자가 자기 터미널에서 `uvx --from huggingface_hub hf auth
login`으로 로그인해 둔 상태입니다. 토큰은 세부 권한 토큰이고 조직의 저장소
쓰기 권한이 있습니다. 올릴 때는 `huggingface_hub`의 `upload_folder`를 쓰고,
돌려받은 판(커밋 해시)과 파일 해시를 `scripts/download_chNN.sh`에 고정합니다.

## 3부에서 갚을 빚

장을 쓰면서 "뒤에서 빠르게 만든다"고 약속한 것들입니다. 해당 장을 쓸 때
빠뜨리지 않도록 모아 둡니다.

| 약속 | 나온 곳 | 갚을 곳 |
|---|---|---|
| ~~단순 행렬 곱 `1024 × 1024`가 0.84초~~ | 2.5절 | 16.6절에서 갚음 (0.038초). 스레드는 18장 |
| ~~브로드캐스팅 덧셈이 빠른 길보다 12배 느림~~ | 5.5절 | 17.8절에서 갚음 (26.2ms → 1.5ms) |
| ~~`linear`가 벡터마다 `matvec`을 따로 부름~~ | 5.3절 | 16.4절에서 갚음 |
| ~~`Mlp::forward`에 활성화 함수 융합이 없음~~ | 3.7절 | 24.7절에서 LLM의 MLP에 손으로 함. 그래프에서는 30.9절에서 갚음(`fuse_activations`) |
| ~~연산자마다 새 텐서를 할당함. 텐서 하나에 할당 4번, 토큰 하나에 2,067번~~ | 5장 전반, 22.9절 | 22.9~22.11절에서 갚음 (토큰 하나에 8번. 형상을 텐서 안에, 자리 돌려 쓰기, 아레나) |
| 아레나가 블록이 끝날 때까지 자리를 돌려주지 않음 (프롬프트 처리에서 +20MB). 텐서의 수명을 읽어 자리를 계획 | 22.13절 | 26장 |
| MNIST 1만 장 분류가 PyTorch보다 약 420배 느림 (412ms 대 1ms) | 7.6절 | 16.8절에서 35배, 18.12절에서 3.9배로 줄임 (3.9ms). 남은 것은 Accelerate의 몫 |
| ~~한 장씩 넣을 때 장당 41µs. 계산이 아닌 비용이 큼~~ | 7.6절 | 16~17장의 커널로 4.7µs (PyTorch 6.5µs). 22.9절에서 정리 |
| ~~토큰화 시간이 전체에서 차지하는 비중을 모델 시간과 비교~~ | 8.11절 | 13.6절에서 갚음 (6백만 분의 1) |
| 대화의 뼈대와 사용자 글을 따로 토큰화 (특수 토큰 주입 방지) | 8.9절 | 37장 |
| ~~BF16을 f32로 펼쳐 읽음. 메모리 두 배, 복사 발생~~ | 9.3절 | 20.1절, 20.9절에서 갚음 (BF16 그대로 계산. 읽기 0.01초) |
| ~~토큰 11개에 블록 하나가 0.32초 (MLP 0.27초, 어텐션 0.05초)~~ | 9.11절 | 16~18장에서 갚음 (모델 전체가 토큰 256개에 0.66초) |
| ~~어텐션의 계산이 글 길이의 제곱으로 늘어남~~ | 9.6절 | 24장에서 24배 빠르게 함. 기울기는 모델의 구조가 정함(24.10절) |
| ~~다음 토큰 하나에 2.88초 (토큰 6개). PyTorch의 38배~~ | 10.8절 | 16~18장에서 갚음 |
| ~~메모리 5.0GB. 가중치 파일의 두 배~~ | 10.3절 | 20.9절에서 갚음 (프로세스의 메모리 0.01GB, 가중치는 파일을 비춘 2.47GB) |
| 프롬프트 256토큰 처리가 119초. PyTorch의 490배. 토큰을 묶어 계산하지 못함 | 11.8절 | 16장 |
| ~~새 토큰 하나에 0.58초. PyTorch의 21배~~ | 11.8절 | 18.12절에서 갚음 (0.020초. PyTorch 0.028초) |
| ~~KV 캐시가 f32. 토큰 하나에 64KB, 최대 길이에서 8.6GB~~ | 11.8절 | 21.7절에서 갚음 (f16 선택 가능, 품질 차이 없음, 메모리 절반) |
| ~~f16 KV 캐시를 읽을 때마다 전체를 f32로 펼침~~ | 21.7절 | 24.5절에서 갚음 (커널이 읽으면서 넓힘) |
| 가중치마다 비트를 다르게 주는 섞는 방법을 만들지 않음 (측정만 함) | 21.6절 | 25장에서 필요하면 |
| KV 캐시의 최대 길이 확인 없음 (자리를 미리 잡는 `reserve`는 22.7절에서 넣음) | 11.8절 | 5부 (서빙) |
| 앞부분이 같은 요청의 캐시 재사용 | 11.9절 | 5부 (서빙) |
| ~~llama.cpp 대비 프롬프트 처리 2,600배, 생성 53배 (같은 정밀도, 기본 설정)~~ | 13.5절 | 25장에서 다시 잼: 1.9배 느림, 1.08배 빠름. 남은 것은 25.9절 |
| 양자화한 가중치의 프롬프트 처리가 llama.cpp의 절반 (내적을 넷씩 계산. llama.cpp는 아홉씩) | 25.9절 | 정해지지 않음 (20장의 연습 5) |
| 제안서의 26장은 `.tflite` 해석기. 22~25장에서 "26장의 그래프에서"라고 미룬 것들(GPU에 계산 전체, 자리 계획)이 LLM 쪽에도 닿는지 26장을 쓸 때 확인 | 22.13, 23.12, 24.7, 25.9절 | 26장, 30장 |
| ~~`argmax`가 top-k 40보다 느림 (0.50ms 대 0.15ms)~~ | 12.8절 | 14.10절에서 갚음 (0.04ms) |
| 선형 층이 순전파 시간의 99.8%. 곱셈 하나에 명령 2.4개, 사이클 2.0개, IPC 1.17 | 14.7절, 14.9절 | 16장, 17장 |
| ~~큰 모델에서 곱셈당 사이클이 20% 더 듦~~ | 14.9절 | 15.4절에서 갚음 (메모리가 아니라 행의 길이) |
| ~~토큰 생성 1.76토큰/초. 스레드 1개의 상한은 15.9~~ | 15.6절 | 16.8절에서 14.0. 나머지는 17장 |
| ~~프롬프트 처리가 토큰을 묶지 않아 산술 강도가 0.25에 머묾~~ | 15.5절 | 16.4절에서 갚음 (27.8토큰/초) |
| ~~스레드 12개의 토큰 생성 상한 55.6. 메모리에 막힘~~ | 15.6절 | 20.7절에서 갚음 (BF16 98, Q8_0 162, Q4_0 230) |
| ~~대칭 눈금이 정수 하나를 버림. 4비트에서 16분의 1~~ | 19.4절 | 20.2절에서 갚음 (`Q4_0`, +1.1dB) |
| 눈금의 끝을 가장 큰 값에 맞춤. 눈금을 찾는 양자화는 만들지 않음 (`Q4_K`는 읽기만). Q4_K_M이 Q4_0보다 손실이 6분의 1 | 19.6절, 20.8절, 21.5절 | 다음 판 (만들 값어치가 있음을 21장이 보임) |
| 양자화한 가중치의 프롬프트 처리가 llama.cpp의 절반 (592 대 1,284). 8비트 행렬 곱 명령(i8mm)을 Rust 안정판에서 못 씀 | 20.7절 | 25장 |
| ~~Q8_0·Q4_0을 읽을 때마다 양자화함 (1.2초, 메모리 최대치 2GB 이상)~~ | 20.9절 | 22.2절에서 갚음 (0.01초, 0.01GB) |
| Q4_0의 토큰 생성이 곱셈이 아닌 일들에 막힘 (230토큰/초, 메모리 161GB/초) | 20.7절 | 24장 |
| 입력을 정수로 바꾸는 일(`Activations::quantize`)이 한 스레드, 스칼라 | 20.4절 | 24장 |
| ~~양자화한 모델의 품질을 글 하나로만 봄~~ | 19.8절 | 21.5절에서 갚음 (퍼플렉서티) |
| 프롬프트 처리에서 `zip`, `rms_norm`, `rope`를 스레드에 나누지 않음 (합쳐 3%) | 18.9절 | 24장 |
| 프롬프트 처리의 기본 스레드 수가 효율 코어를 포함함. 작은 모델에서 손해 | 18.10절 | 6부 (기기별 설정) |
| top-p만 쓸 때 12만 개 전체를 정렬 (1.5ms) | 12.8절 | 모델이 빨라진 뒤 (18장 이후) |
| 순전파가 중간 결과를 모두 쥐고 있음 | 10.1절 | 26장에서 수명 분석과 계획기를 만듦. Llama 엔진에는 적용하지 않음(26.13절) |
| 아레나가 블록의 텐서를 모두 더한 만큼 쥠 (54MB) | 22.13절 | 26.13절에서 계산 (27.26MB가 하한). 엔진에는 적용하지 않음 |
| 계산 전체를 GPU에 올려 한 번에 보내기 | 23.12절 | 그래프는 26장에서 생김. 올리는 일은 남음(26.14절) |
| 벡터 하나짜리 선형 층이 LiteRT(SME2)의 절반 이하 | 26.12절 | 31.6절에서 SME 커널을 짰지만 벡터 하나에서는 1.7배뿐. 긴 벡터의 `fmla` 커널이 필요(31장 연습 문제 6) |
| ~~합성곱이 선형 층의 커널을 빌려 씀. 스레드 12개에서 5.7배, LiteRT의 절반~~ | 27.12절 | 28.9절에서 갚음 (8.7배, LiteRT의 1.4배). 커널 자체는 31장 |
| ~~`ADD`가 ReLU를 따로 훑음, `PAD`가 마디로 남아 텐서를 한 번 더 만듦~~ | 27.12절 | 28.4절(ADD), 30.10절(PAD)에서 갚음 |
| ~~펼친 행렬(7.4MB)이 메모리 계획 밖~~ | 27.11절 | 28.2절에서 줄임 (조각마다 64KB 이하. 뒤쪽 층은 여전히 통째로 펼치지만 작음) |
| 16비트 연산을 그래프 전체에 쓰기 | 28.8절 | 남음. 29장은 정수 전용 해석기를 따로 만들었고 `f32` 해석기는 그대로 |
| 정수 커널이 LiteRT 자체 커널의 5분의 1 (배율 곱하기가 스칼라) | 29.10절 | 31.11절에서 절반 갚음(22.4 → 11.5ms). 아직 3배. 깊이별 합성곱이 남음 |
| 실수와 정수가 섞인 그래프, 해석기 둘을 하나로 | 29.8절 | 필요한 모델이 나올 때 |
| 스레드의 기본값이 효율 코어를 포함 (16개가 12개보다 느림) | 18.10절, 27.12절 | 6부 |

## 미결 사항

| 항목 | 결정 시점 |
|---|---|
| 주 LLM 최종 확정 (Llama 3.2 1B 예정) | 2부 착수 시 |
| 음성 합성 모델 확정 | 4부 착수 시 |
| mdBook 0.5 계열로 올릴지 | 급하지 않음 |

## 결정된 사항

| 항목 | 결정 |
|---|---|
| 코드 라이선스 | MIT. 원고와 그림에는 적용하지 않음 |
| 다이어그램 | SVG 파일로 생성 (`scripts/figures/`) |
| 텐서의 자료형 | 실행 시점에 결정 (`DType` + 열거형 저장소). 지금은 `f32`, `i32`, `u8` |
| 오류 처리 | 텐서부터는 패닉 대신 `Result<_, Error>` |
| 메모리 매핑 | `libc` 크레이트 없이 `mmap`을 직접 선언. 유닉스 외에는 통째로 읽기로 대체 |
| 정답 파일 | 공식 Python 패키지로 만들어 저장소에 넣음. 스크립트는 `reference/`, 버전은 `requirements.txt`에 고정 |
| 모델 파일의 위치 | 어떤 모델도 저장소에 넣지 않음. 남의 모델은 원래 위치에서, 우리 모델과 정답은 저자의 Hugging Face Hub 계정에서 내려받음 |
| 대조의 세 가지 원인 | 결과가 다르면 우리가 틀렸거나, 상대가 틀렸거나, 비교 방법이 틀린 것. 8장에서 llama-tokenize의 기본 옵션(역슬래시 표기 변환) 때문에 없는 차이가 보였음. 비교 도구의 옵션을 먼저 확인 |
| 비교 조건 | PyTorch는 이 맥에서 Accelerate를 쓰고 스레드 수와 무관하게 약 2 TFLOP/s를 냄(7.6절). 엔진을 비교할 때는 상대가 쓰는 하드웨어를 먼저 확인하고, 조건을 맞춘 비교와 맞추지 않은 비교를 나누어 잼 |
| 아직 읽지 못하는 것 | 16비트 실수(19장), GGUF의 양자화 형식(20장) |

## 장별 상태

상태는 `완료`, `작업 중`, `대기` 중 하나입니다. `완료`는 초고를 커밋했다는
뜻이며, 저자 검토는 별도로 진행됩니다.

| 장 | 제목 | 상태 | 비고 |
|---|---|---|---|
| - | 표지 | 완료 | |
| - | 서문 | 완료 | 저장소 사용법은 코드가 생기면 보강 |
| 1 | 추론 엔진의 지형도 | 완료 | 개관 장, 코드 없음 |
| 2 | 추론에 필요한 수학 | 완료 | `badak_core`의 `linalg`, `prob`, `approx`. 단순 행렬 곱 첫 측정 |
| 3 | 신경망은 함수다 | 완료 | `nn` 모듈: 활성화 함수, `Linear`, `Mlp`, 손으로 만든 XOR 모델 |
| 4 | 텐서 만들기 | 완료 | `tensor`와 `error` 모듈. 자료형은 실행 시점에 결정. 뷰와 복사의 비용 측정 |
| 5 | 연산자 만들기 | 완료 | `ops` 모듈. 빠른 길과 일반적인 길. 쌍별 합산, 비교 보고서 `compare` |
| 6 | 모델 파일 읽기 | 완료 | `badak_formats` 크레이트: JSON, 메모리 매핑, safetensors, GGUF. 핵심 크레이트에 빌린 저장소 추가 |
| 7 | 첫 완주: 손글씨 숫자 분류 | 완료 | `badak_models` 크레이트, IDX 읽기, 첫 노트북, 첫 Hub 업로드. PyTorch와 1만 장 전부 일치 |

| 8 | 토크나이저 | 완료 | `badak_models::tokenizer`. llama.cpp와 완전 일치(시험 자료 47건 × 2, 원고·소스 전체) |
| 9 | 트랜스포머 해부 | 완료 | `badak_models::llama`. 첫 블록의 부품 10개가 PyTorch와 일치 (두 모델) |
| 10 | 한 토큰 생성하기 | 완료 | `LlamaModel`, `tokenizer.json` 읽기. 모든 블록이 PyTorch와 일치 (두 모델). " Paris" 70%. 토큰 하나에 2.88초, PyTorch의 38배 |
| 11 | KV 캐시와 생성 루프 | 완료 | `llama::cache`, `llama::generate`. 캐시 결과가 비트 단위로 같고 PyTorch와 같은 토큰을 뽑음 (두 모델). 24토큰 생성 200초 → 16.5초 |
| 12 | 샘플링 | 완료 | `random::Pcg64`, `sampling`. 걸러 낸 분포가 transformers와 일치(7가지 설정, 두 모델). 2장 `softmax`의 합산 오차를 찾아 고침 |
| 13 | 첫 대결: llama.cpp | 완료 | 세 엔진의 점수가 f64 기준과 `2e-6` 안에서 일치. 격차를 스레드·Accelerate·GPU·양자화로 분해. 2부 끝 |

| 14 | 측정과 프로파일링 | 완료 | `badak_bench` 크레이트, `core::profile`. 선형 층이 99.8%임을 계측과 표본으로 확인. 명령·사이클 카운터. `argmax` 14배. 기준 측정 고정 |
| 15 | 병목 계산하기 | 완료 | 메모리 대역폭과 계산 한계를 재서 상한을 계산. llama.cpp의 측정이 상한의 81~94%. 14.9절의 짐작을 바로잡음 |
| 16 | 행렬 곱 최적화 | 완료 | `core::kernel`. 토큰 생성 1.76 → 14.0, 프롬프트 처리 2.16 → 27.8토큰/초. 스레드 1개의 llama.cpp와 같거나 앞섬. 결과는 여전히 비트 단위로 재현 |
| 17 | SIMD | 완료 | `core::simd`: 기준 구현, NEON, AVX2, 실행 시 선택. 토큰 생성 14.0 → 18.1, 프롬프트 처리 27.8 → 42.4토큰/초. NEON과 AVX2가 같은 비트 |
| 18 | 멀티스레딩 | 완료 | `core::threads`: 직접 만든 스레드 풀. 토큰 생성 18.1 → 50.5(메모리 상한의 93%), 프롬프트 처리 42.4 → 389토큰/초. 스레드 수와 무관하게 같은 비트 |
| 19 | 양자화 이론 | 완료 | `core::quant`: f16/BF16 변환, 정수 눈금, 흉내 양자화. 텐서에 16비트 자료형. 속도는 그대로. BF16은 무손실, 4비트는 32개마다 눈금을 정해야 답이 맞음 |
| 20 | LLM 가중치 양자화 | 완료 | BF16 그대로 계산(같은 비트), `core::packed`의 Q8_0·Q4_0과 정수 커널. 토큰 생성 52.5 → 96.2(BF16) → 160(Q8_0) → 222(Q4_0). llama.cpp의 파일과 바이트 동일. 프롬프트 처리는 llama.cpp의 절반 |
| 21 | 양자화 품질 측정 | 완료 | `llama::perplexity`. llama.cpp와 소수점 넷째 자리까지 일치. Q8_0은 원래 모델과 구별 안 됨, Q4_0은 손실 +0.185, Q4_K_M은 +0.032. `down_proj`와 블록 1이 민감. f16 KV 캐시는 품질 차이 없음 |
| 22 | 메모리 | 완료 | 양자화한 가중치를 파일에 두고 비춰 씀(읽기 1.2초 → 0.01초). 할당: 프롬프트 42만 → 40번, 토큰 7,493 → 8번(형상을 텐서 안에, 자리 돌려 쓰기, 아레나). 토큰 생성 +2.5~9%. 속도 비교는 번갈아, 쉬면서 |
| 23 | GPU 연산 | 완료 | `badak_gpu`(wgpu). 선형 층 커널 다섯: 단순한 것은 CPU보다 느리고 행렬 연산은 7.9배. 프롬프트 처리 412 → 2,116토큰/초(MLP를 통째로 GPU에). 토큰 생성은 CPU에 둠. 같은 비트는 포기. llama.cpp Metal과 2.6배 차이 |
| 24 | 어텐션 최적화와 커널 융합 | 완료 | KV 캐시를 헤드마다 나란히 놓아 17장의 커널로 점수를 냄. 문맥 4,096토큰에서 토큰 생성 17.5 → 76.4토큰/초. 16비트 캐시가 f32와 같은 속도. MLP의 활성화와 곱셈을 융합. 입력의 양자화를 스레드에 나눠 Q8_0 프롬프트 처리 +18% |
| 25 | 재대결: llama.cpp | 완료 | llama.cpp를 쉬면서 다시 잼. 기본 설정(GPU) 대비 프롬프트 처리 1.9배 느림, 토큰 생성 1.08배 빠름(13장: 2,600배, 53배). CPU의 토큰 생성은 같거나 빠름. 남은 격차: 양자화한 커널, GPU에 계산 전체, Accelerate. 3부 끝 |
| 26 | 그래프 해석기 | 완료 | FlatBuffers와 `.tflite` 파서, `badak_graph`(레지스트리, 실행 계획, 수명 분석, 메모리 계획). LiteRT와 만 장 모두 같은 답. 잔차 모델의 중간 결과 16.5KB → 4.2~5.2KB, Llama 블록 45.6MB → 27.3MB(계산만). 한 장 추론은 LiteRT가 2.2~3.1배 빠름(SME2) |
| 27 | 합성곱과 비전 분류 | 완료 | 합성곱 세 방법(정의, im2col, `1 × 1`), 깊이별 합성곱, 풀링·덧대기·평균·축 바꾸기. ResNet-50과 MobileNetV2가 LiteRT·PyTorch와 같은 점수. `badak_image`: Pillow와 바이트까지 같은 줄이기. 전처리가 모델보다 네 배 오래 걸림. LiteRT가 스레드 하나로 2배 빠름(28장으로). |
| 28 | 합성곱 커널 최적화 | 완료 | 출력을 조각내어 펼치기, 모양을 보고 방법 고르기. ResNet-50 16.9 → 11.1ms, MobileNetV2 5.6 → 2.4ms(스레드 12개). 위노그라드(스레드 1개에서만 이득), 간접 합성곱(1.5배 느림), 채널 배치 실험, 16비트 연산(1.9배, 오차 1e-3)은 재 보고 기본값에 넣지 않음. LiteRT(스레드 1개)의 1.4배 |
| 29 | 정수 양자화 추론 | 완료 | 눈금과 영점, 채널별 양자화, 고정소수점 곱셈. int8 MobileNetV2가 LiteRT의 자체 커널과 비트까지 같음(70개 층 전부). LiteRT 안의 반올림 세 가지를 층마다 대조해 찾음. 파일 14.1 → 4.1MB, 자리 9.8 → 2.5MB. 속도는 LiteRT 자체 커널의 5분의 1 |
| 30 | 그래프 최적화 | 완료 | Protocol Buffers와 ONNX 읽기, 기계적으로 옮기기, 패스 여덟, 제자리 계산. ResNet-50 448 → 73마디, 스레드 12개 46.8 → 10.7ms. `.tflite`의 `PAD` 합치기로 MobileNetV2 2.11 → 1.64ms |
| 31 | 모바일 추론 엔진 해부와 대결 | 완료 | LiteRT·ONNX Runtime·NCNN·MNN 설계 비교와 실측. 순위는 SME를 쓰는가, 스레드 수, 정밀도, 층의 모양으로 갈림. `asm!`으로 SME 커널(NEON의 7~16배), ResNet-50 스레드 1개 93.8 → 15.2ms(LiteRT 8.0). 정수 커널 22.4 → 11.5ms |
| 32 | 비전 트랜스포머와 객체 검출 | 완료 | ViT-Tiny를 코드로(PyTorch와 1.5e-5), 같은 코드로 ViT-B/16(403 → 64ms, PyTorch 36ms), YOLOX-Tiny를 그래프로(연산 다섯, 패스 넷. 612 → 129마디). 해독과 비최대 억제, 사진 파일에서 상자까지. SiLU의 지수 함수가 병목 |
| 33 | 음성 인식 | 완료 | WAV, FFT(Bluestein), 멜 스펙트로그램, Whisper-tiny(인코더·디코더·교차 어텐션·KV 캐시), 탐욕과 빔 서치, 크기 기반 VAD. transformers와 토큰 일치. 스레드 1개 인코더 573ms(SME 132ms), whisper.cpp 242ms |
| 34 | 임베딩과 검색 | 완료 | WordPiece(유니코드 분해 표, 한글은 식), all-MiniLM-L6-v2(PyTorch와 3e-7, STS 0.8203), 채우기와 이어 붙이기, 묶음들을 스레드에 나누기(12스레드 13,149글/초, ORT 5,605), 정확한 검색 |
| 35 | 이미지 생성 | 완료 | Stable Diffusion 1.5를 코드로(CLIP 토크나이저·텍스트 인코더, U-Net, VAE 디코더, DDIM, 가이던스). diffusers와 383바이트 차이. 20걸음 28초(SME), sd.cpp CPU 85초·Metal 10초. 걸음·가이던스와 시간, 병목은 어텐션 |
| 36 | 멀티모달 | 완료 | SmolVLM-256M(SigLIP ViT + 이웃을 잇는 커넥터 + Llama). 사진의 벡터를 토큰 자리에 끼우기, 란초스 전처리(Pillow와 바이트 일치), transformers와 토큰 48개 일치. 0.37초에 사진 보고 48토큰. JPEG 풀기의 차이, tanh GELU 병목 |
| 37 | 음성 합성 | 완료 | OuteTTS-0.1-350M(2부의 Llama) + WavTokenizer 디코더(피클을 가중치만 읽기로 열어 변환) + 거꾸로 가는 푸리에 변환. PyTorch와 토큰 302개·소리 일치. 첫 소리 77ms, 실시간의 2.3배. 조각내어 풀면 다른 소리, Whisper로 되받아쓰기 |
| 38 | 종합 실습: 음성 대화 파이프라인 | 완료 | Whisper → 언어 모델 → 음성 합성. 문장마다 넘겨 응답 지연 253ms(다 쓴 뒤 374ms). Llama 3.2 1B는 한 문장으로 답해 두 방법이 같음(0.8초). PyTorch와 사슬 전체의 토큰 일치. 스레드 1개에서는 소리가 끊김. 말의 끝 검출은 빠져 있음 |

5부 이후의 장은 해당 부를 시작할 때 이 표에 추가합니다. 전체 목록은
[제안서](proposal.md) 7절에 있습니다.
