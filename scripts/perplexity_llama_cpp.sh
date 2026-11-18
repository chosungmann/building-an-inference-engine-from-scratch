#!/bin/sh
# llama.cpp로 같은 글의 퍼플렉서티를 잰다 (21장).
#
# 사용법: sh scripts/perplexity_llama_cpp.sh [모델 디렉터리의 이름] [조각 수]
#
# F32, Q8_0, Q4_0, Q4_K_M 파일을 잰다. 조각 수를 주지 않으면 글 전체를 쓴다.
# 먼저 준비할 것:
#   sh scripts/build_llama_cpp.sh llama-perplexity
#   sh scripts/convert_to_gguf.sh, sh scripts/quantize_gguf.sh
#   sh scripts/download_wikitext2.sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
CHUNKS="${2:--1}"
MODELS="${BADAK_MODELS:-$ROOT/models}"
TOOL="$ROOT/third_party/llama.cpp/build/bin/llama-perplexity"
TEXT="$MODELS/wikitext2/wikitext-2-raw/wiki.test.raw"

echo "# llama.cpp, 모델 $NAME, WikiText-2 시험 글, 한 번에 512토큰, 조각 수 $CHUNKS"
for file in model_f32.gguf model_q8_0.gguf model_q4_0.gguf model_q4_k_m.gguf; do
    # GPU를 쓰지 않는다. 마지막 줄의 추정값만 남긴다.
    result="$("$TOOL" -m "$MODELS/$NAME/$file" -f "$TEXT" -c 512 --chunks "$CHUNKS" -ngl 0 2>&1 \
        | grep "Final estimate" | sed 's/.*Final estimate: //')"
    printf "%-20s %s\n" "$file" "$result"
done
