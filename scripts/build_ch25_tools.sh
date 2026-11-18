#!/bin/sh
# 25장에서 쓰는 작은 C++ 도구를 llama.cpp에 연결해 빌드한다.
#
# 사용법: sh scripts/build_ch25_tools.sh
#
# llama.cpp의 두 빌드에 하나씩 연결한다(13.1절).
#   third_party/llama.cpp/build/bin/ch25_llama_context        Accelerate와 GPU 포함
#   third_party/llama.cpp/build_plain/bin/ch25_llama_context  직접 짠 CPU 코드만
# 먼저 sh scripts/build_llama_cpp.sh 와 sh scripts/build_llama_cpp.sh --plain.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/third_party/llama.cpp"
for build in build build_plain; do
    BIN="$LLAMA/$build/bin"
    [ -d "$BIN" ] || { echo "먼저 llama.cpp를 빌드하십시오: $BIN 이 없습니다." >&2; exit 1; }
    c++ -std=c++17 -O2 \
        -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
        "$ROOT/bench/ch25/llama_context.cpp" \
        -L "$BIN" -lllama -Wl,-rpath,"$BIN" \
        -o "$BIN/ch25_llama_context"
    echo "준비되었습니다: $BIN/ch25_llama_context"
done
