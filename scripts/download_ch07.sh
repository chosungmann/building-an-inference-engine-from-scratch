#!/bin/sh
# 7장의 손글씨 숫자 분류 모델과 PyTorch 정답을 내려받아 models/mnist_mlp/ 에 둔다.
#
# 사용법: sh scripts/download_ch07.sh
#
# 이 책이 학습해 Hugging Face Hub에 올려 둔 파일이다. 판(revision)을 고정해
# 받고 SHA-256 해시를 확인한다. 직접 학습해서 만들려면 이 스크립트 대신
# reference/ch07_train_mnist.py 를 실행한다.
set -eu

REPO="building-an-inference-engine-from-scratch/mnist-mlp"
REVISION="ff9e117d7f1f3902e55c68b29eda74700c9282c5"
BASE="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/mnist_mlp"
mkdir -p "$OUT"

FILES="
model.safetensors 6049a34ee22aa69a72c7e6b7d5c69a044df6192ae1d154bca6a1b0bc2f8a5b44
reference.safetensors 872f346f792f34bbf4fc8a274adbe9823c102a8af45adc390dce3a7fe3c9d0b3
"

echo "$FILES" | while read -r name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$BASE/$name" -o "$OUT/$name.partial"
        mv "$OUT/$name.partial" "$OUT/$name"
    fi
    actual=$(shasum -a 256 "$OUT/$name" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "해시가 다릅니다: $name" >&2
        echo "  기대: $expected" >&2
        echo "  실제: $actual" >&2
        echo "  직접 학습한 파일이라면 정상입니다. 책의 파일을 받으려면 지우고 다시 실행하십시오." >&2
        exit 1
    fi
    echo "확인: $name"
done
echo "모델이 $OUT 에 준비되었습니다."
