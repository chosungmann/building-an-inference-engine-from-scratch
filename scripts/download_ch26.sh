#!/bin/sh
# 26장의 .tflite 모델 둘과 LiteRT의 정답을 내려받아 models/mnist_tflite/ 에 둔다.
#
# 사용법: sh scripts/download_ch26.sh
#
# 이 책이 만들어 Hugging Face Hub에 올려 둔 파일이다. 버전(revision)을 고정해
# 받고 SHA-256 해시를 확인한다. 직접 만들려면 이 스크립트 대신
# reference/ch26_make_tflite.py 를 실행한다.
set -eu

REPO="building-an-inference-engine-from-scratch/mnist-tflite"
REVISION="9bb2098554f5c5e3cdb3b4c060eebf7abdf7f166"
BASE="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/mnist_tflite"
mkdir -p "$OUT"

FILES="
mlp.tflite 1c38b87144e8e5ab75db517a6482cfb91ac1a80c4429df9e1dc8b5ee2ca0a13d
residual.tflite 1e53b92fd0e01d4d7ff828acba048f23a052ee13b4d9cb2c75b88b2da8ec1cc2
reference.safetensors 33cc629d6715acc15043acda785906947f9c80cb2c3bf88a88dbe435a099571b
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
        echo "  직접 만든 파일이라면 정상입니다. 책의 파일을 받으려면 지우고 다시 실행하십시오." >&2
        exit 1
    fi
    echo "확인: $name"
done
echo "모델이 $OUT 에 준비되었습니다."
