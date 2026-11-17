#!/bin/sh
# llama.cpp의 속도를 llama-bench로 잰다 (13장).
#
# 사용법: sh scripts/bench_llama_cpp.sh [모델 디렉터리의 이름]
#
# 프롬프트 256토큰을 처리하는 속도(pp)와 토큰 32개를 만드는 속도(tg)를 다섯
# 번씩 재서 평균과 표준편차를 낸다. 설정을 한 가지씩 바꿔 가며 여섯 번 잰다.
# 먼저 준비할 것:
#   sh scripts/build_llama_cpp.sh
#   sh scripts/build_llama_cpp.sh --plain
#   sh scripts/convert_to_gguf.sh [모델 디렉터리의 이름]
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
MODEL="${BADAK_MODELS:-$ROOT/models}/$NAME"
FULL="$ROOT/third_party/llama.cpp/build/bin/llama-bench"
PLAIN="$ROOT/third_party/llama.cpp/build_plain/bin/llama-bench"
THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)"

# 설명, 실행 파일, 모델 파일, 나머지 인자.
run() {
    label="$1"; bench="$2"; file="$3"; shift 3
    echo "## $label"
    # 표만 남긴다.
    "$bench" -m "$MODEL/$file" -p 256 -n 32 -r 5 -o md "$@" 2>/dev/null | grep '^|'
    echo
}

echo "# llama.cpp, 모델 $NAME, 성능 코어 ${THREADS}개"
echo
run "1. 직접 짠 CPU 코드만, F32, 스레드 1개"        "$PLAIN" model_f32.gguf    -t 1 -fa off
run "2. 스레드를 ${THREADS}개로"                      "$PLAIN" model_f32.gguf    -t "$THREADS" -fa off
run "3. 기본 빌드(Accelerate 포함), CPU만, 기본 설정" "$FULL"  model_f32.gguf    -t "$THREADS" -ngl 0
run "4. GPU(Metal)까지"                               "$FULL"  model_f32.gguf    -t "$THREADS" -ngl 99
run "5. 4비트 양자화(Q4_K_M), CPU만"                  "$FULL"  model_q4_k_m.gguf -t "$THREADS" -ngl 0
run "6. 4비트 양자화(Q4_K_M), GPU까지"                "$FULL"  model_q4_k_m.gguf -t "$THREADS" -ngl 99
