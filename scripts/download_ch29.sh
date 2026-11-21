#!/bin/sh
# 29장의 정수 모델과 LiteRT의 정답을 models/vision_tflite/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch29.sh
#
# 27장의 사진과 모델도 필요하다(sh scripts/download_ch27.sh). 정수 모델은
# 27장의 mobilenetv2.tflite(원본 Apache-2.0)를 이 책이 8비트 정수로 바꿔
# Hugging Face Hub에 올려 둔 것이다. 버전을 고정해 받고 해시를 확인한다. 직접
# 만들려면 reference/ch29_quantize.py 를 실행한다.
set -eu

REPO="building-an-inference-engine-from-scratch/imagenet-tflite"
REVISION="e5c6da2e6729c8fe5484c0ef0c10852a596a0a69"
HUB="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/vision_tflite"
mkdir -p "$OUT"

FILES="
mobilenetv2_int8.tflite e2433fb944abb1dd6bb257e38036afdaa78e00fbe975bbe54254c1691a601760
reference_int8.safetensors 4e454cbee93d00dd4dbc69d777663fea1af6fefa0e5d67086975fc807ce10bd5
"

echo "$FILES" | while read -r name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$HUB/$name" -o "$OUT/$name.partial"
        mv "$OUT/$name.partial" "$OUT/$name"
    fi
    actual=$(shasum -a 256 "$OUT/$name" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "해시가 다릅니다: $name" >&2
        echo "  기대: $expected" >&2
        echo "  실제: $actual" >&2
        exit 1
    fi
    echo "확인: $name"
done
echo "정수 모델이 $OUT 에 준비되었습니다."
