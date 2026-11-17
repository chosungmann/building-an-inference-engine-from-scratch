#!/bin/sh
# 13장에서 쓰는 작은 C++ 도구를 llama.cpp에 연결해 빌드한다.
#
# 사용법: sh scripts/build_ch13_tools.sh
#
# 먼저 sh scripts/build_llama_cpp.sh 로 llama.cpp를 빌드해 두어야 한다.
# 실행 파일은 third_party/llama.cpp/build/bin/ch13_llama_logits 에 놓인다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/third_party/llama.cpp"
BIN="$LLAMA/build/bin"
[ -d "$BIN" ] || { echo "먼저 sh scripts/build_llama_cpp.sh 를 실행하십시오." >&2; exit 1; }

c++ -std=c++17 -O2 \
    -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
    "$ROOT/bench/ch13/llama_logits.cpp" \
    -L "$BIN" -lllama -Wl,-rpath,"$BIN" \
    -o "$BIN/ch13_llama_logits"
echo "준비되었습니다: $BIN/ch13_llama_logits"
