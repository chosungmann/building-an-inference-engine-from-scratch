# 진행 상황

세션을 시작할 때 이 문서를 먼저 읽고, 장을 끝낼 때마다 갱신합니다.

## 다음에 할 일

**14장. 측정과 프로파일링**을 씁니다. 3부의 첫 장입니다.

- 지표(첫 토큰 시간, 초당 토큰, 토큰 간 지연), 백분위수, 벤치마크 설계,
  프로파일러, 하드웨어 카운터(제안서).
- 2부의 측정은 모두 "초시계 수준"이라고 적어 왔습니다. 13.4절의 일곱 규칙을
  지키는 측정 도구를 `badak` 안에 만듭니다(되풀이, 준비 운동, 평균과
  표준편차 또는 백분위수, 환경 기록). 13장의 `ch13_bench`가 출발점입니다.
- 프로파일러로 시간이 어디에 쓰이는지 직접 봅니다. 9.11절과 10.8절에서
  계산으로 추정한 것(시간의 대부분이 선형 층)을 프로파일로 확인합니다.
  macOS에서는 `sample`이나 Instruments, 하드웨어 카운터는 권한 문제가
  있을 수 있으니 무엇이 가능한지부터 확인합니다.
- 갚을 빚: `argmax`가 top-k 40보다 느린 이유(12.8절).
- 3부 내내 쓸 기준 측정(13.7절의 표)을 14장의 도구로 다시 재서 고정합니다.

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
| 단순 행렬 곱 `1024 × 1024`가 0.84초 | 2.5절 | 16~18장 |
| 브로드캐스팅 덧셈이 빠른 길보다 12배 느림 | 5.5절 | 17장 |
| `linear`가 벡터마다 `matvec`을 따로 부름 | 5.3절 | 16장 |
| `Mlp::forward`에 활성화 함수 융합이 없음 | 3.7절 | 24장, 30장 |
| 연산자마다 새 텐서를 할당함 | 5장 전반 | 22장, 26장 |
| MNIST 1만 장 분류가 PyTorch보다 약 420배 느림 (412ms 대 1ms) | 7.6절 | 16~18장에서 다시 측정 |
| 한 장씩 넣을 때 장당 41µs. 계산이 아닌 비용이 큼 | 7.6절 | 22장 |
| ~~토큰화 시간이 전체에서 차지하는 비중을 모델 시간과 비교~~ | 8.11절 | 13.6절에서 갚음 (6백만 분의 1) |
| 대화의 뼈대와 사용자 글을 따로 토큰화 (특수 토큰 주입 방지) | 8.9절 | 37장 |
| BF16을 f32로 펼쳐 읽음. 메모리 두 배, 복사 발생 | 9.3절 | 19장, 22장 |
| 토큰 11개에 블록 하나가 0.32초 (MLP 0.27초, 어텐션 0.05초) | 9.11절 | 16~18장 |
| 어텐션의 계산이 글 길이의 제곱으로 늘어남 | 9.6절 | 24장 |
| 다음 토큰 하나에 2.88초 (토큰 6개). PyTorch의 38배 | 10.8절 | 13장에서 llama.cpp와 비교, 16~18장 |
| 메모리 5.0GB. 가중치 파일의 두 배 | 10.3절 | 19장, 22장 |
| 프롬프트 256토큰 처리가 119초. PyTorch의 490배. 토큰을 묶어 계산하지 못함 | 11.8절 | 16장 |
| 새 토큰 하나에 0.58초. PyTorch의 21배 | 11.8절 | 16~18장 |
| KV 캐시가 f32. 토큰 하나에 64KB, 최대 길이에서 8.6GB | 11.8절 | 19장 |
| KV 캐시를 `Vec`에 덧붙임. 최대 길이 확인 없음 | 11.8절 | 22장 |
| 앞부분이 같은 요청의 캐시 재사용 | 11.9절 | 5부 (서빙) |
| llama.cpp 대비 프롬프트 처리 2,600배, 생성 53배 (같은 정밀도, 기본 설정). 조각별 크기는 13.7절의 표 | 13.5절 | 14~24장, 25장에서 다시 측정 |
| `argmax`가 top-k 40보다 느림 (0.50ms 대 0.15ms). 원인 미확인 | 12.8절 | 14장 |
| top-p만 쓸 때 12만 개 전체를 정렬 (1.5ms) | 12.8절 | 모델이 빨라진 뒤 (18장 이후) |
| 순전파가 중간 결과를 모두 쥐고 있음 | 10.1절 | 22장, 26장 |

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

3부 이후의 장은 해당 부를 시작할 때 이 표에 추가합니다. 전체 목록은
[제안서](proposal.md) 7절에 있습니다.
