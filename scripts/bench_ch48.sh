#!/bin/sh
# 48장의 측정: 같은 엔진을 맥에서 그대로, Node에서, 브라우저 넷에서 돌린다.
#
# 쓰는 법: sh scripts/bench_ch48.sh [chrome edge firefox safari 가운데 몇]
#         PAGES=1 sh scripts/bench_ch48.sh   (긴 측정은 건너뛰고 데모 페이지만)
# 결과: bench/ch48/*.txt
#
# 브라우저는 새 프로필로 띄우고 끝나면 닫는다. Safari만은 쓰던 프로필로
# 열린다. 측정 페이지는 끝날 때 자기가 넣어 둔 것을 지운다.
#
# 필요한 것: sh scripts/download_mnist.sh, download_ch26.sh, download_smollm2.sh,
# rustup target add wasm32-wasip1 wasm32-wasip1-threads, node, python3.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch48"
WORK="$ROOT/target/web_bench"
REPORTS="$ROOT/target/web_reports"
BROWSERS="${*:-chrome edge firefox safari}"
cd "$ROOT"
mkdir -p "$WORK" "$REPORTS" "$OUT"

if [ -z "${PAGES:-}" ]; then
    sh scripts/build_web.sh target/web_site > "$OUT/sizes.txt"
    cargo build --quiet --release -p badak_web --example ch48_native

    # 1. 맥에서 그대로. 스레드 하나와, 엔진이 고른 만큼.
    sleep 3
    BADAK_THREADS=1 target/release/examples/ch48_native llm > "$OUT/native_1.txt"
    sleep 3
    target/release/examples/ch48_native llm > "$OUT/native.txt"

    # 2. Node. 브라우저 없이 같은 .wasm을 돌린다.
    : > "$OUT/node.txt"
    for build in scalar simd relaxed; do
        sleep 3
        { echo "# badak_$build.wasm"; node crates/badak_web/js/run_node.mjs "target/web/badak_$build.wasm" 1 llm; } >> "$OUT/node.txt"
    done
    for threads in 1 2 4 8; do
        sleep 3
        { echo "# badak_relaxed_threads.wasm, 스레드 $threads"; node crates/badak_web/js/run_node.mjs target/web/badak_relaxed_threads.wasm "$threads" llm; } >> "$OUT/node.txt"
    done
else
    sh scripts/build_web.sh target/web_site > /dev/null
fi
# 측정 페이지는 책에 싣지 않는다. 여기에서만 복사한다.
cp crates/badak_web/web/bench.html target/web_site/

# 3. 브라우저. 서버 둘을 띄운다. 8048은 격리 헤더를 붙이고, 8049는 GitHub
#    Pages처럼 붙이지 않는다.
python3 scripts/serve_web.py --isolate --port 8048 --reports "$REPORTS" > /dev/null 2>&1 &
ISOLATED=$!
python3 scripts/serve_web.py --port 8049 --reports "$REPORTS" > /dev/null 2>&1 &
PLAIN=$!
trap 'kill $ISOLATED $PLAIN 2> /dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:8048/bench.html && curl -sf -o /dev/null http://127.0.0.1:8049/bench.html; do sleep 1; done

launch() {
    browser="$1" url="$2" profile="$WORK/profile_$1"
    # KEEP를 주면 앞의 측정이 쓰던 프로필을 그대로 쓴다.
    [ -n "${KEEP:-}" ] || rm -rf "$profile"
    mkdir -p "$profile"
    case "$browser" in
        chrome) "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --user-data-dir="$profile" \
            --no-first-run --no-default-browser-check "$url" > /dev/null 2>&1 & ;;
        edge) "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" --user-data-dir="$profile" \
            --no-first-run --no-default-browser-check "$url" > /dev/null 2>&1 & ;;
        firefox) /Applications/Firefox.app/Contents/MacOS/firefox -profile "$profile" -no-remote "$url" > /dev/null 2>&1 & ;;
        safari) open -F -a Safari "$url" ;;
    esac
}

close() {
    case "$1" in
        safari) pkill -x Safari || true ;;
        *) pkill -f "$WORK/profile_$1" || true ;;
    esac
    sleep 2
}

# 페이지를 열고, 그것이 결과를 보낼 때까지 기다린다.
measure() {
    browser="$1" name="$2" url="$3" limit="$4"
    rm -f "$REPORTS/ch48_$name.txt"
    sleep 3
    launch "$browser" "$url"
    waited=0
    until [ -f "$REPORTS/ch48_$name.txt" ] || [ "$waited" -ge "$limit" ]; do
        sleep 2
        waited=$((waited + 2))
    done
    close "$browser"
    if [ -f "$REPORTS/ch48_$name.txt" ]; then
        cp "$REPORTS/ch48_$name.txt" "$OUT/$name.txt"
    else
        echo "$name: ${limit}초 안에 끝나지 않았습니다" | tee "$OUT/$name.txt"
    fi
}

for browser in $BROWSERS; do
    if [ -z "${PAGES:-}" ]; then
        # 격리된 페이지에서: 커널 셋, 스레드, 언어 모델, 메모리의 끝.
        measure "$browser" "$browser" "http://127.0.0.1:8048/bench.html?name=$browser&exhaust" 900
        # 스레드가 일을 기다릴 때 돌지 않고 바로 잠들게 하면 어떤가.
        if [ "$browser" = chrome ]; then
            measure "$browser" "${browser}_sleep" "http://127.0.0.1:8048/bench.html?name=${browser}_sleep&waiting=sleep" 900
        fi
    fi
    # 헤더가 없는 서버에서: 서비스 워커가 페이지를 격리해 내는가.
    measure "$browser" "${browser}_worker" "http://127.0.0.1:8049/index.html?models=local&report=${browser}_worker" 60
    # 서비스 워커도 없이: 스레드 없는 빌드로 물러나는가.
    measure "$browser" "${browser}_plain" "http://127.0.0.1:8049/index.html?models=local&plain&report=${browser}_plain" 60
    # Hugging Face에서 진짜로 받는다(269MB). 그리고 브라우저를 껐다 켜서
    # 넣어 둔 것을 꺼낸다.
    if [ "$browser" = chrome ]; then
        measure "$browser" "${browser}_hub_first" "http://127.0.0.1:8049/index.html?download&report=${browser}_hub_first" 600
        KEEP=1 measure "$browser" "${browser}_hub_again" "http://127.0.0.1:8049/index.html?download&report=${browser}_hub_again" 120
    fi
done
echo "끝났습니다: $OUT"
