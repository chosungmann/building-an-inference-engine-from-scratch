#!/bin/sh
# 25장: 짧은 실행 하나의 시간과 메모리를 두 엔진에서 잰다 (13.6절의 다시 재기).
#
# 사용법: sh scripts/measure_ch25_memory.sh [모델 디렉터리의 이름]
#
# 같은 글 뒤에 토큰 16개를 만드는 실행을 /usr/bin/time -l 로 감싼다. 시간은
# 프로그램 시작부터 끝까지(가중치 읽기 포함)이고, 메모리는 22.1절의 두
# 숫자다. llama.cpp에는 문맥의 길이를 4,096으로 준다(13.6절).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
MODEL="${BADAK_MODELS:-$ROOT/models}/$NAME"
BADAK="$ROOT/target/release/examples/ch11_generate"
LLAMA="$ROOT/third_party/llama.cpp/build/bin/llama-completion"
PROMPT="Once upon a time, in a small village by the sea, there lived an old fisherman who"

measure() {
    label="$1"; shift
    sleep 3
    report="$( { /usr/bin/time -l "$@" >/dev/null; } 2>&1 )"
    seconds="$(echo "$report" | awk '/ real / {printf "%.2f", $1}')"
    resident="$(echo "$report" | awk '/maximum resident set size/ {printf "%.2f", $1 / 1e9}')"
    peak="$(echo "$report" | awk '/peak memory footprint/ {printf "%.2f", $1 / 1e9}')"
    printf '%-34s %6s초 %10sGB %12sGB\n' "$label" "$seconds" "$resident" "$peak"
}

printf '%-34s %8s %12s %14s\n' "엔진과 설정" "시간" "올라온 메모리" "지켜야 하는 메모리"
for weights in stored q8_0 q4_0; do
    measure "badak, $weights" env BADAK_WEIGHTS="$weights" "$BADAK" "$NAME" "$PROMPT" 16
done
for file in model_f32 model_q8_0 model_q4_0; do
    measure "llama.cpp, $file, CPU" "$LLAMA" -m "$MODEL/$file.gguf" -p "$PROMPT" -n 16 -c 4096 \
        -ngl 0 -no-cnv --no-display-prompt --simple-io
    measure "llama.cpp, $file, GPU" "$LLAMA" -m "$MODEL/$file.gguf" -p "$PROMPT" -n 16 -c 4096 \
        -ngl 99 -no-cnv --no-display-prompt --simple-io
done
