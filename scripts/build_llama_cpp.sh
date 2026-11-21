#!/bin/sh
# 비교 대상인 llama.cpp를 고정된 커밋으로 받아 빌드한다.
#
# 사용법: sh scripts/build_llama_cpp.sh [--plain] [빌드할 대상...]
#   예: sh scripts/build_llama_cpp.sh llama-tokenize
#   대상을 주지 않으면 이 책이 쓰는 도구를 모두 빌드한다.
#   39장의 비교에는 llama-server 가 필요하다(--plain 으로도 빌드한다).
#
# 소스는 third_party/llama.cpp 에, 실행 파일은 그 아래 build/bin 에 놓인다.
# --plain 을 주면 build_plain/bin 에 놓인다(13장).
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

# --plain: GPU(Metal)와 외부 행렬 라이브러리(Accelerate, BLAS)를 뺀 버전을
# build_plain 에 따로 빌드한다. llama.cpp가 직접 짠 CPU 코드만 남는다.
BUILD="$ROOT/build"
OPTIONS=""
if [ "${1:-}" = "--plain" ]; then
    shift
    BUILD="$ROOT/build_plain"
    OPTIONS="-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF"
    [ $# -gt 0 ] || set -- llama-bench
fi

[ $# -gt 0 ] || set -- llama-tokenize llama-completion llama-bench llama-quantize
# 서버(39장)의 웹 화면은 빌드할 때 따로 내려받는 파일이고, 고정해 둔
# 커밋과 맞지 않으면 빌드가 실패한다. 화면 없이 빌드한다.
# shellcheck disable=SC2086
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF $OPTIONS >/dev/null
for target in "$@"; do
    echo "빌드하는 중: $target"
    cmake --build "$BUILD" --target "$target" -j >/dev/null
done
echo "준비되었습니다: $BUILD/bin"
