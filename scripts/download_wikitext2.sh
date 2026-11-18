#!/bin/sh
# 21장: 퍼플렉서티를 잴 글(WikiText-2의 시험 부분)을 내려받는다.
#
# 사용법: sh scripts/download_wikitext2.sh
#
# models/wikitext2/wikitext-2-raw/wiki.test.raw가 생긴다. llama.cpp가
# 퍼플렉서티를 잴 때 쓰는 것과 같은 파일이고, 받는 곳도 llama.cpp의
# scripts/get-wikitext-2.sh와 같다. 글은 위키백과에서 온 것으로 CC BY-SA
# 라이선스를 따른다. 저장소에는 넣지 않는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIRECTORY="${BADAK_MODELS:-$ROOT/models}/wikitext2"
ZIP="$DIRECTORY/wikitext-2-raw-v1.zip"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
SHA256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"

mkdir -p "$DIRECTORY"
if [ ! -f "$ZIP" ]; then
    echo "내려받는 중: $URL"
    curl -sSL -o "$ZIP" "$URL"
fi
actual="$(shasum -a 256 "$ZIP" | cut -d ' ' -f 1)"
if [ "$actual" != "$SHA256" ]; then
    echo "파일이 기대한 것과 다릅니다: $ZIP" >&2
    echo "  기대한 SHA-256: $SHA256" >&2
    echo "  실제 SHA-256:   $actual" >&2
    exit 1
fi
unzip -o -q "$ZIP" -d "$DIRECTORY"
echo "준비되었습니다: $DIRECTORY/wikitext-2-raw/wiki.test.raw"
