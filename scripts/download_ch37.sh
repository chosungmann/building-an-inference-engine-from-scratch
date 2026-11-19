#!/bin/sh
# 37장의 모델과 정답을 models/outetts/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch37.sh [decoder]
#
# 모델은 원래 배포처에서 받는다.
#   OuteAI/OuteTTS-0.1-350M (CC-BY-4.0). 0.7GB.
# `decoder`를 붙이면 오디오 디코더의 원본 체크포인트(1.75GB)도 받는다.
#   novateur/WavTokenizer-large-speech-75token (MIT)
# 그 파일은 PyTorch의 피클이어서 우리 엔진이 읽지 못한다. 받은 뒤에
#   python reference/ch37_convert.py
# 로 safetensors로 바꾼다(가중치만 읽는 방식으로 연다. 37.5절). 바꾼 파일은
# 이 책이 따로 배포하지 않는다.
# 정답(reference.safetensors)은 이 책이 reference/ch37_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 판을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/outetts"
mkdir -p "$OUT/original"

MODEL="https://huggingface.co/OuteAI/OuteTTS-0.1-350M/resolve/2191ede86648eb8e4c458d98385ba56c09b0155e"
DECODER="https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/f0eafde4078a49b3875d79d33632a252dcd02f60/wavtokenizer_large_speech_320_v2.ckpt"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/5e2f60b70a58e4ce616ac909ab4e6dd6b0e2c279/reference_ch37.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
model.safetensors $MODEL/model.safetensors 13f6618e16e24e4aaf9ab3181c8f1e74d46982ea3655019e4f0f8de8414c1a5d
config.json $MODEL/config.json 9f5c84ef54aa2fb2c24fee08239a7ec8ffdd22639ee18b8119d0874528144265
tokenizer.json $MODEL/tokenizer.json 90a1dfdf7e0236f091a6e6bce9e2b5198cd2530d981c010d85f397ee50b72bc2
reference.safetensors $REFERENCE 6a94081a83093af26cc454e6d21a9569d5c57277f64a3abbbb36555b1b09bd44
"
if [ "${1:-}" = "decoder" ]; then
    FILES="$FILES
original/wavtokenizer.ckpt $DECODER 5dd430d5f0d96e2313babb1b896d0f990b65bfd143b2894fc3851cfc3cca9846
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
echo "37장의 모델이 $OUT 에 준비되었습니다."
