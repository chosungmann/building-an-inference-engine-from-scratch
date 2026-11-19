#!/bin/sh
# 36장의 모델과 정답을 models/smolvlm/ 에 내려받는다. 0.5GB다.
#
# 사용법: sh scripts/download_ch36.sh
#
# 모델은 원래 배포처에서 받는다.
#   HuggingFaceTB/SmolVLM-256M-Instruct (Apache-2.0)
# 사진은 27장의 것을 쓴다(scripts/download_ch27.sh).
# 정답(reference.safetensors)은 이 책이 reference/ch36_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 판을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/smolvlm"
mkdir -p "$OUT"

MODEL="https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/632124d754f71ec6acf11305abc3986b98d3083d/reference_ch36.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
model.safetensors $MODEL/model.safetensors 74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e
config.json $MODEL/config.json b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52
tokenizer.json $MODEL/tokenizer.json 5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a
reference.safetensors $REFERENCE c8d134b26041cf88c6ed9c5b678fea54b9b271c85fe646778d835dbe0c4b0341
"

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
echo "36장의 모델이 $OUT 에 준비되었습니다."
