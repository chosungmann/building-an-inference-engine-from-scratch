#!/bin/sh
# 48장: WebAssembly로 빌드한 엔진이 이 기계의 엔진과 같은 답을 내는지 본다.
#
# 쓰는 법: sh scripts/check_web.sh
#
# 브라우저 없이 Node로 돌린다. 올리는 코드(badak.js, wasi.js, thread.js)는
# 브라우저에서 쓰는 것과 같다. CI가 이 스크립트를 돌린다.
#
# 벡터 명령을 쓰지 않는 빌드(scalar)는 비트까지 같아야 한다. 스레드를 넷
# 써도 그래야 한다. 벡터 명령을 쓰는 빌드는 고른 답이 같은지만 본다. 끝자리
# 비트는 기계에 따라 다를 수 있다(본문의 "같은 비트를 낼 수 있는가").
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
sh scripts/build_web.sh
cargo build --quiet --release -p badak_web --example ch48_native

# 견줄 것만 남긴다: 선형 층의 지문, 숫자의 답과 비트, 언어 모델이 쓴 글.
digests() { grep '^linear' "$1" | awk '{print $2, $4}'; grep '^mnist 답' "$1"; }
answers() { grep '^mnist 답' "$1" | awk '{print $3}'; }
native_text() { grep ' short ' "$1" | sed 's/.* text //'; }
node_text() { grep '^llm ' "$1" | awk '{print $3}'; }

WORK="$ROOT/target/web_check"
mkdir -p "$WORK"
target/release/examples/ch48_native llm > "$WORK/native.txt"
failed=0
for run in "scalar 1" "scalar_threads 4" "simd 1" "relaxed 1" "relaxed_threads 4"; do
    set -- $run
    # Node가 오래되어 이 빌드의 명령을 모르면 건너뛴다.
    if ! node crates/badak_web/js/can_run.mjs "$1"; then
        echo "badak_$1.wasm: 이 Node가 모르는 명령이 있어 건너뜁니다"
        continue
    fi
    node crates/badak_web/js/run_node.mjs "target/web/badak_$1.wasm" "$2" llm > "$WORK/$1.txt"
    if [ "$(node_text "$WORK/$1.txt")" != "$(native_text "$WORK/native.txt")" ]; then
        echo "badak_$1.wasm: 언어 모델이 쓴 글이 다릅니다"
        failed=1
    fi
    if [ "$(answers "$WORK/$1.txt")" != "$(answers "$WORK/native.txt")" ]; then
        echo "badak_$1.wasm: 숫자의 답이 다릅니다"
        failed=1
    fi
    if [ "$(digests "$WORK/$1.txt")" = "$(digests "$WORK/native.txt")" ]; then
        echo "badak_$1.wasm (스레드 $2): 비트까지 같습니다"
    else
        echo "badak_$1.wasm (스레드 $2): 답은 같고 끝자리 비트가 다릅니다"
        case "$1" in scalar*) failed=1 ;; esac
    fi
done
exit "$failed"
