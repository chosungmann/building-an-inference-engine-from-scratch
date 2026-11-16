#!/bin/sh
# 8장의 토크나이저가 쓰는 단어 사전과 시험 자료를 내려받아
# models/llama_cpp_vocab/ 에 둔다.
#
# 사용법: sh scripts/download_ch08.sh
#
# llama.cpp 저장소에 들어 있는 파일이다. 가중치 없이 단어 사전만 담은 GGUF와,
# llama.cpp가 자기 토크나이저를 시험할 때 쓰는 입력(.inp)과 정답(.out)이다.
# 커밋을 고정해 받고 SHA-256 해시를 확인한다.
set -eu

COMMIT="b4c0e00a4fcc0287d0bb32fd7cb0d8b286af73ed"
BASE="https://raw.githubusercontent.com/ggml-org/llama.cpp/$COMMIT/models"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/llama_cpp_vocab"
mkdir -p "$OUT"

FILES="
ggml-vocab-llama-bpe.gguf 97272e430d53bc7688f52d5e0ad8ea8f163ede9f1bbd1694feaa504797d5d96e
ggml-vocab-llama-bpe.gguf.inp be0a11f7071f0c67d3053a2d377d8f35f0dbcb77ff11a60300fb57d36b477cf0
ggml-vocab-llama-bpe.gguf.out 118abf2034a197fc5c9dec5204dfdeee7004c7c70952ae577f66b99e212fab9b
ggml-vocab-gpt-2.gguf cedc56ca6e2e89f63e781696d1fd76b4b1d49e6720dee86463e915f6e90016ac
ggml-vocab-gpt-2.gguf.inp be0a11f7071f0c67d3053a2d377d8f35f0dbcb77ff11a60300fb57d36b477cf0
ggml-vocab-gpt-2.gguf.out 48e8c4cb3ad5c37915b225f872222567c1c4dd4d66951aa53460c8d152ff6d60
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
        exit 1
    fi
    echo "확인: $name"
done
echo "단어 사전이 $OUT 에 준비되었습니다."
