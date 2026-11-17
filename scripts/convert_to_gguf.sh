#!/bin/sh
# 모델 디렉터리의 가중치를 llama.cpp가 읽는 GGUF 파일로 바꾼다.
#
# 사용법: sh scripts/convert_to_gguf.sh [모델 디렉터리의 이름...]
#   예: sh scripts/convert_to_gguf.sh smollm2_135m
#   이름을 주지 않으면 llama32_1b와 smollm2_135m 가운데 있는 것을 모두 바꾼다.
#
# models/<이름>/ 아래에 세 파일을 만든다.
#   model_f32.gguf     가중치를 f32로 펼친 것. badak과 조건을 맞춘 비교에 쓴다.
#   model_bf16.gguf    원본과 같은 BF16.
#   model_q4_k_m.gguf  4비트로 양자화한 것. llama.cpp를 쓰는 사람들이 흔히 쓰는 형식.
#
# 먼저 sh scripts/build_llama_cpp.sh 로 llama.cpp를 받아 빌드해 두어야 한다.
# 변환기는 파이썬으로 되어 있고 reference/requirements.txt 의 패키지에 더해
# sentencepiece가 필요하다. 파이썬 실행 파일은 PYTHON으로 바꿀 수 있다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/third_party/llama.cpp"
MODELS="${BADAK_MODELS:-$ROOT/models}"
PYTHON="${PYTHON:-python3}"
[ -x "$LLAMA/build/bin/llama-quantize" ] || { echo "먼저 sh scripts/build_llama_cpp.sh 를 실행하십시오." >&2; exit 1; }

[ $# -gt 0 ] || set -- llama32_1b smollm2_135m
for name in "$@"; do
    directory="$MODELS/$name"
    [ -f "$directory/model.safetensors" ] || { echo "건너뜀: $directory 에 모델이 없습니다"; continue; }
    for type in f32 bf16; do
        out="$directory/model_$type.gguf"
        [ -f "$out" ] && continue
        echo "변환하는 중: $name ($type)"
        # 변환기가 저장소 안의 gguf 패키지를 쓰게 한다.
        PYTHONPATH="$LLAMA/gguf-py" "$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$directory" \
            --outtype "$type" --outfile "$out" >/dev/null 2>&1
    done
    out="$directory/model_q4_k_m.gguf"
    if [ ! -f "$out" ]; then
        echo "양자화하는 중: $name (Q4_K_M)"
        "$LLAMA/build/bin/llama-quantize" "$directory/model_f32.gguf" "$out" Q4_K_M >/dev/null 2>&1
    fi
    ls -l "$directory"/model_*.gguf | awk '{ printf "  %6.0f MB  %s\n", $5 / 1e6, $9 }'
done
