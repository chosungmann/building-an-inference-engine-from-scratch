#!/bin/sh
# 38장의 정답을 models/talk/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch38.sh
#
# 38장은 새 모델을 쓰지 않는다. 앞 장들의 것을 먼저 받는다.
#   sh scripts/download_ch33.sh        Whisper-tiny와 소리 파일
#   sh scripts/download_smollm2.sh     SmolLM2-135M-Instruct
#   sh scripts/download_ch37.sh decoder && python reference/ch37_convert.py
# 정답(reference.safetensors)은 이 책이 reference/ch38_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 버전을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/talk"
mkdir -p "$OUT"

URL="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/219c7314a29433fa47164e4df2a5148f694a6a63/reference_ch38.safetensors"
EXPECTED="b6db4409f5c72384c28a7d60ee54925135cae97ce38a2dcd48a5833310456100"

if [ ! -f "$OUT/reference.safetensors" ]; then
    echo "내려받는 중: reference.safetensors"
    curl -fsSL "$URL" -o "$OUT/reference.safetensors.partial"
    mv "$OUT/reference.safetensors.partial" "$OUT/reference.safetensors"
fi
actual=$(shasum -a 256 "$OUT/reference.safetensors" | cut -d ' ' -f 1)
if [ "$actual" != "$EXPECTED" ]; then
    echo "해시가 다릅니다: reference.safetensors" >&2
    echo "  기대: $EXPECTED" >&2
    echo "  실제: $actual" >&2
    exit 1
fi
echo "확인: reference.safetensors"
echo "38장의 정답이 $OUT 에 준비되었습니다."
