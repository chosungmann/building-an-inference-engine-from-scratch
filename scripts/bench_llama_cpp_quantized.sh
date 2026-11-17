#!/bin/sh
# llama.cpp가 저장 방식마다 내는 속도를 llama-bench로 잰다 (20장).
#
# 사용법: sh scripts/bench_llama_cpp_quantized.sh [모델 디렉터리의 이름]
#
# 직접 짠 CPU 코드만 쓰는 빌드로(13.5절), 같은 모델의 F32, Q8_0, Q4_0, Q4_K_M
# 파일을 스레드 수를 바꿔 가며 잰다. BF16 파일은 재지 않는다. 이 빌드에서
# 스레드 1개의 첫 측정이 20분이 지나도 끝나지 않았다. 먼저 준비할 것:
#   sh scripts/build_llama_cpp.sh --plain
#   sh scripts/convert_to_gguf.sh [모델 디렉터리의 이름]
#   sh scripts/quantize_gguf.sh [모델 디렉터리의 이름]
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
MODEL="${BADAK_MODELS:-$ROOT/models}/$NAME"
PLAIN="$ROOT/third_party/llama.cpp/build_plain/bin/llama-bench"

echo "# llama.cpp (직접 짠 CPU 코드만), 모델 $NAME"
echo
for file in model_f32.gguf model_q8_0.gguf model_q4_0.gguf model_q4_k_m.gguf; do
    echo "## $file"
    # 스레드 수를 쉼표로 이어 주면 차례로 잰다. 표만 남긴다.
    "$PLAIN" -m "$MODEL/$file" -p 256 -n 32 -r 5 -o md -t 1,6,12 -fa off 2>/dev/null | grep '^|'
    echo
done
