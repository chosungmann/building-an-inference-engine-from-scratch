#!/bin/sh
# MNIST 손글씨 숫자 데이터를 내려받아 models/mnist/ 에 푼다.
#
# 사용법: sh scripts/download_mnist.sh
#
# PyTorch 프로젝트가 운영하는 미러에서 받는다. 받은 파일은 압축을 푼 뒤
# SHA-256 해시를 확인한다. 이미 있는 파일은 다시 받지 않는다.
set -eu

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/mnist"
mkdir -p "$OUT"

# 파일 이름과, 압축을 푼 파일의 SHA-256.
FILES="
train-images-idx3-ubyte ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
train-labels-idx1-ubyte 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
t10k-images-idx3-ubyte 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
t10k-labels-idx1-ubyte ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2
"

echo "$FILES" | while read -r name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$BASE/$name.gz" | gunzip > "$OUT/$name.partial"
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
echo "MNIST가 $OUT 에 준비되었습니다."
