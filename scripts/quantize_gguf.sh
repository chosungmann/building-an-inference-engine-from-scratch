#!/bin/sh
# 20장: llama.cpp의 도구로 F32 GGUF 파일을 양자화한 파일들을 만든다.
#
# 사용법: sh scripts/quantize_gguf.sh [모델 디렉터리 이름]
#
# 먼저 scripts/build_llama_cpp.sh와 scripts/convert_to_gguf.sh를 실행해 두어야
# 한다. models/<이름>/model_q8_0.gguf와 model_q4_0.gguf가 생긴다. 이 파일들은
# 우리가 만든 양자화가 llama.cpp의 것과 같은 바이트를 내는지 견주는 데 쓴다.
set -eu

name="${1:-llama32_1b}"
root="$(cd "$(dirname "$0")/.." && pwd)"
tool="$root/third_party/llama.cpp/build/bin/llama-quantize"
directory="$root/models/$name"

if [ ! -x "$tool" ]; then
    echo "llama-quantize가 없습니다. 먼저 sh scripts/build_llama_cpp.sh를 실행하십시오." >&2
    exit 1
fi
if [ ! -f "$directory/model_f32.gguf" ]; then
    echo "$directory/model_f32.gguf가 없습니다. 먼저 sh scripts/convert_to_gguf.sh $name을 실행하십시오." >&2
    exit 1
fi

for kind in Q8_0 Q4_0; do
    lower="$(echo "$kind" | tr '[:upper:]' '[:lower:]')"
    output="$directory/model_$lower.gguf"
    if [ -f "$output" ]; then
        echo "이미 있습니다: $output"
        continue
    fi
    "$tool" "$directory/model_f32.gguf" "$output" "$kind" > /dev/null 2>&1
    echo "만들었습니다: $output"
done
