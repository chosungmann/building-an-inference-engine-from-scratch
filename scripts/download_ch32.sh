#!/bin/sh
# 32장의 모델과 정답을 models/vision_transformer/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch32.sh [base]
#
# `base`를 붙이면 ViT-B/16의 가중치(346MB)도 받는다.
#   vit_base.safetensors  timm/vit_base_patch16_224.augreg2_in21k_ft_in1k (Apache-2.0)
#
# 모델 둘은 원래 배포처에서 받는다.
#   vit_tiny.safetensors  timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k (Apache-2.0)
#   yolox_tiny.onnx       YOLOX 프로젝트의 0.1.1rc0 버전 (Apache-2.0)
# 정답(reference.safetensors)은 이 책이 reference/ch32_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 버전을 고정해 받고 해시를 확인한다.
# 27장의 사진과 정답도 필요하다(sh scripts/download_ch27.sh).
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/vision_transformer"
mkdir -p "$OUT"

VIT="https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106/model.safetensors"
YOLOX="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
VIT_BASE="https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/aaec0043dadec6b3da0971bf50bff8926a1342ba/reference_ch32.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
vit_tiny.safetensors $VIT fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed
yolox_tiny.onnx $YOLOX 427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7
reference.safetensors $REFERENCE 5b55404a41929a2eba013faa52a440af2718b555002c987684dc4aa494c3ff00
"

if [ "${1:-}" = "base" ]; then
    FILES="$FILES
vit_base.safetensors $VIT_BASE 32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b
"
fi

echo "$FILES" | while read -r name url expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$url" -o "$OUT/$name.partial"
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
echo "32장의 모델이 $OUT 에 준비되었습니다."
