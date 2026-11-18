#!/bin/sh
# 25장: llama.cpp의 속도를 다시 잰다. 실행 사이에 쉰다.
#
# 사용법: sh scripts/bench_llama_cpp_rematch.sh [모델 디렉터리의 이름]
#
# 13장과 20장에서 잰 설정들을 한 번에, 22.12절의 방법으로 다시 잰다. 설정을
# 바꿀 때마다 REST초(기본 5초) 쉬고, llama-bench에도 측정 사이에 3초씩 쉬게
# 한다(--delay). 프롬프트 256토큰(pp)과 토큰 32개(tg)를 다섯 번씩 잰다.
# 먼저 준비할 것:
#   sh scripts/build_llama_cpp.sh
#   sh scripts/build_llama_cpp.sh --plain
#   sh scripts/convert_to_gguf.sh [모델 디렉터리의 이름]
#   sh scripts/quantize_gguf.sh [모델 디렉터리의 이름]
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
MODEL="${BADAK_MODELS:-$ROOT/models}/$NAME"
FULL="$ROOT/third_party/llama.cpp/build/bin/llama-bench"
PLAIN="$ROOT/third_party/llama.cpp/build_plain/bin/llama-bench"
REST="${REST:-5}"

# 설명, 실행 파일, 모델 파일, 나머지 인자.
run() {
    label="$1"; bench="$2"; file="$3"; shift 3
    sleep "$REST"
    echo "## $label"
    "$bench" -m "$MODEL/$file" -p 256 -n 32 -r 5 --delay 3 -o md "$@" 2>/dev/null | grep '^|'
    echo
}

echo "# llama.cpp, 모델 $NAME, 실행 사이에 ${REST}초, 측정 사이에 3초 쉼"
echo
echo "# 직접 짠 CPU 코드만 (Accelerate와 GPU를 뺀 빌드)"
echo
run "F32"  "$PLAIN" model_f32.gguf  -t 8,12,16 -fa off
run "Q8_0" "$PLAIN" model_q8_0.gguf -t 8,12,16 -fa off
run "Q4_0" "$PLAIN" model_q4_0.gguf -t 8,12,16 -fa off
echo "# 기본 빌드 (Accelerate 포함)"
echo
run "F32, CPU만"            "$FULL" model_f32.gguf    -t 12 -ngl 0
run "F32, GPU까지"          "$FULL" model_f32.gguf    -t 12 -ngl 99
run "Q4_0, GPU까지"         "$FULL" model_q4_0.gguf   -t 12 -ngl 99
run "Q4_K_M, CPU만"         "$FULL" model_q4_k_m.gguf -t 12 -ngl 0
run "Q4_K_M, GPU까지"       "$FULL" model_q4_k_m.gguf -t 12 -ngl 99
