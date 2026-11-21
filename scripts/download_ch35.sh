#!/bin/sh
# 35장의 모델과 정답을 models/stable_diffusion/ 에 내려받는다. 4.3GB다.
#
# 사용법: sh scripts/download_ch35.sh [text]
#
# `text`를 붙이면 토크나이저와 텍스트 인코더, 정답만 받는다(0.5GB). U-Net과
# VAE 없이 할 수 있는 시험에 쓴다.
#
# 모델은 원래 배포처에서 받는다.
#   stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M)
#   허가 조건은 그 저장소의 설명에 있다. 만든 그림으로 해서는 안 되는 일이
#   적혀 있으니 받기 전에 읽는다.
# `*_config.json`은 정답을 만드는 Python 스크립트만 읽는다.
# 정답(reference.safetensors)은 이 책이 reference/ch35_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 버전을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/stable_diffusion"
mkdir -p "$OUT"

MODEL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/88881548b7612880961129bac8ce64da469fbf60/reference_ch35.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
vocab.json $MODEL/tokenizer/vocab.json e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349
merges.txt $MODEL/tokenizer/merges.txt 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a
text_encoder.safetensors $MODEL/text_encoder/model.safetensors d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334
text_encoder_config.json $MODEL/text_encoder/config.json 845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc
reference.safetensors $REFERENCE afdb9a45fef1dc554f30e920a831d33766fbbd8222bb651a645385919ae439f4
"
if [ "${1:-}" != "text" ]; then
    FILES="$FILES
unet.safetensors $MODEL/unet/diffusion_pytorch_model.safetensors 19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1
unet_config.json $MODEL/unet/config.json 78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96
vae.safetensors $MODEL/vae/diffusion_pytorch_model.safetensors a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c
vae_config.json $MODEL/vae/config.json 786a7d21647ddea6a04b9675c03d3cb45e90a2f3c6da5fbda2c54ade040036de
scheduler_config.json $MODEL/scheduler/scheduler_config.json 699cce92eb7c122e2eb7dfdea78e6187fda76a5ed4a8e42319b85610e620e091
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
echo "35장의 모델이 $OUT 에 준비되었습니다."
