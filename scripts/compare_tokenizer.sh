#!/bin/sh
# badak의 토크나이저와 llama.cpp의 토크나이저를 여러 파일에서 대조한다.
#
# 사용법: sh scripts/compare_tokenizer.sh
#
# 이 책의 원고와 소스 파일 전체를 두 도구로 토큰화해, 토큰 번호가 하나라도
# 다르면 실패한다. 먼저 scripts/download_ch08.sh 와
# scripts/build_llama_cpp.sh llama-tokenize 를 실행해 두어야 한다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOCAB="${BADAK_MODELS:-$ROOT/models}/llama_cpp_vocab"
# --no-escape: 파일 속의 역슬래시 표기(\n 등)를 llama.cpp가 줄바꿈으로 바꾸지 않게 한다.
LLAMA="$ROOT/third_party/llama.cpp/build/bin/llama-tokenize"
cd "$ROOT"
cargo build -q --release -p badak_models --example ch08_dump
DUMP="$ROOT/target/release/examples/ch08_dump"

status=0
for vocab in ggml-vocab-llama-bpe.gguf ggml-vocab-gpt-2.gguf; do
    files=0 tokens=0 different=0
    for file in book/src/*.md book/src/part*/*.md docs/*.md crates/*/src/*.rs crates/*/src/*/*.rs README.md AGENTS.md; do
        [ -f "$file" ] || continue
        ours=$("$DUMP" "$VOCAB/$vocab" "$file")
        theirs=$("$LLAMA" -m "$VOCAB/$vocab" -f "$file" --ids --no-bos --no-parse-special --no-escape --log-disable 2>/dev/null)
        files=$((files + 1))
        tokens=$((tokens + $(echo "$theirs" | tr -cd ',' | wc -c) + 1))
        if [ "$ours" != "$theirs" ]; then
            different=$((different + 1))
            echo "다름: $vocab $file"
            status=1
        fi
    done
    echo "$vocab: 파일 ${files}개, 토큰 ${tokens}개, 결과가 다른 파일 ${different}개"
done
exit $status
