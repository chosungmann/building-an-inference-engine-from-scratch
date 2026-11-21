#!/bin/sh
# 30장의 ONNX 모델과 ONNX Runtime의 정답을 models/vision_onnx/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch30.sh
#
# 27장의 사진과 정답도 필요하다(sh scripts/download_ch27.sh). ONNX 모델은
# 27장과 같은 가중치(ResNet-50은 BSD-3-Clause, MobileNetV2는 Apache-2.0)를
# 이 책이 PyTorch에서 ONNX로 내보내 Hugging Face Hub에 올려 둔 것이다. 배치
# 정규화 접기와 상수 접기를 끄고 내보냈다. 버전을 고정해 받고 해시를 확인한다.
# 직접 만들려면 reference/ch30_make_onnx.py 를 실행한다.
set -eu

REPO="building-an-inference-engine-from-scratch/imagenet-tflite"
REVISION="97689a1660317ffb9162527bc3062d26222de6c8"
HUB="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/vision_onnx"
mkdir -p "$OUT"

# Hub의 파일 이름, 받아서 둘 이름, 해시.
FILES="
resnet50.onnx resnet50.onnx 8e1c10958759eb7dd647a9d9e3d9ab68adf7ea186270b1c6cb1bf2730d4f4b16
mobilenetv2.onnx mobilenetv2.onnx 10ce9cecacf829b638aedba67fefffdebfded54476fc787b6ef9250a13d37bf3
reference_onnx.safetensors reference.safetensors f033cd1d71cee8892d9d523cacf944446aa138206d71fb3706247f91c0322a26
"

echo "$FILES" | while read -r remote name expected; do
    [ -n "$remote" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$HUB/$remote" -o "$OUT/$name.partial"
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
echo "ONNX 모델이 $OUT 에 준비되었습니다."
