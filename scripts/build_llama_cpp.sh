#!/bin/sh
# 비교 대상인 llama.cpp를 고정된 커밋으로 받아 빌드한다.
#
# 사용법: sh scripts/build_llama_cpp.sh [빌드할 대상...]
#   예: sh scripts/build_llama_cpp.sh llama-tokenize
#   대상을 주지 않으면 이 책이 쓰는 도구를 모두 빌드한다.
#
# 소스는 third_party/llama.cpp 에, 실행 파일은 그 아래 build/bin 에 놓인다.
# CMake와 C++ 컴파일러가 필요하다.
set -eu

# 태그 b12428에 해당하는 커밋.
COMMIT="b4c0e00a4fcc0287d0bb32fd7cb0d8b286af73ed"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/third_party/llama.cpp"

if [ ! -d "$ROOT/.git" ]; then
    mkdir -p "$ROOT"
    git -C "$ROOT" init -q
    git -C "$ROOT" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if ! git -C "$ROOT" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    echo "내려받는 중: llama.cpp $COMMIT"
    git -C "$ROOT" fetch -q --depth 1 origin "$COMMIT"
fi
git -C "$ROOT" -c advice.detachedHead=false checkout -q "$COMMIT"

[ $# -gt 0 ] || set -- llama-tokenize llama-cli llama-bench
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
for target in "$@"; do
    echo "빌드하는 중: $target"
    cmake --build "$ROOT/build" --target "$target" -j >/dev/null
done
echo "준비되었습니다: $ROOT/build/bin"
