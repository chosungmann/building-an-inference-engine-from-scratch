#!/bin/sh
# 48장: 엔진을 브라우저용 WebAssembly 모듈로 빌드한다.
#
# 쓰는 법: sh scripts/build_web.sh [둘 곳]
# 결과: target/web/badak_<커널>[_threads].wasm 여섯 개.
#       둘 곳을 주면 거기에 .wasm과 JavaScript 파일들, 데모 페이지를 복사한다.
#
# 커널은 셋이다.
#   scalar   벡터 명령 없이. 기준 구현과 같은 비트를 낸다
#   simd     128비트 벡터 명령(simd128)
#   relaxed  거기에 relaxed-simd
# 스레드를 쓰는 것과 쓰지 않는 것을 따로 빌드한다. 브라우저에 맞는 것을
# JavaScript가 골라 올린다.
#
# 필요한 것: rustup target add wasm32-wasip1 wasm32-wasip1-threads
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/target/web"
mkdir -p "$OUT"
cd "$ROOT"

build() {
    name="$1" target="$2" features="$3"
    # 빌드마다 폴더를 따로 쓴다. 설정이 다른 빌드가 서로의 결과를 덮지 않게.
    env RUSTFLAGS="$features" cargo build --quiet --release -p badak_web --target "$target" \
        --target-dir "$ROOT/target/web_build/$name" \
        --config 'profile.release.debug=false' --config 'profile.release.strip=true'
    cp "$ROOT/target/web_build/$name/$target/release/badak_web.wasm" "$OUT/badak_$name.wasm"
    printf '%-22s %8d 바이트, gzip %8d\n' "badak_$name.wasm" \
        "$(wc -c < "$OUT/badak_$name.wasm")" "$(gzip -9 -c "$OUT/badak_$name.wasm" | wc -c)"
}

build scalar wasm32-wasip1 ""
build simd wasm32-wasip1 "-C target-feature=+simd128"
build relaxed wasm32-wasip1 "-C target-feature=+simd128,+relaxed-simd"
build scalar_threads wasm32-wasip1-threads ""
build simd_threads wasm32-wasip1-threads "-C target-feature=+simd128"
build relaxed_threads wasm32-wasip1-threads "-C target-feature=+simd128,+relaxed-simd"

if [ -n "${1:-}" ]; then
    mkdir -p "$1"
    cp "$OUT"/badak_*.wasm "$1/"
    for name in badak wasi thread engine client features setup mnist_demo coi; do
        cp "crates/badak_web/js/$name.js" "$1/"
    done
    cp crates/badak_web/web/index.html "$1/"
    echo "복사했습니다: $1"
fi
