#!/bin/sh
# 49장의 측정: 같은 GPU 커널을 맥에서 그대로(wgpu), 그리고 브라우저 넷의
# WebGPU로 돌린다.
#
# 쓰는 법: sh scripts/bench_ch49.sh [chrome edge firefox safari 가운데 몇]
# 결과: bench/ch49/*.txt
#
# 브라우저는 새 프로필로 띄우고 끝나면 닫는다. Safari만은 쓰던 프로필로
# 열린다.
#
# 필요한 것: sh scripts/download_smollm2.sh, download_ch41.sh(SmolLM2-360M),
# download_llama32.sh(Llama 3.2 1B, Hugging Face의 동의가 필요하다),
# rustup target add wasm32-wasip1 wasm32-wasip1-threads, python3.
# 모델을 바꾸려면 MODELS="smollm2_135m" 처럼 준다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch49"
WORK="$ROOT/target/web_bench"
REPORTS="$ROOT/target/web_reports/ch49"
BROWSERS="${*:-chrome edge firefox safari}"
MODELS="${MODELS:-smollm2_135m smollm2_360m llama32_1b}"
cd "$ROOT"
mkdir -p "$WORK" "$REPORTS" "$OUT"

sh scripts/build_web.sh target/web_site > "$OUT/sizes.txt"
cp crates/badak_web/web/bench_gpu.html crates/badak_web/web/bench.html target/web_site/
cargo build --quiet --release -p badak_web --example ch49_native --example ch49_ramp

# 1. 맥에서 그대로. 표준 WebGPU의 기능만 쓴 것과, 표준 밖의 기능도 쓴 것.
if [ -z "${PAGES:-}" ]; then
    sleep 3
    # shellcheck disable=SC2086
    target/release/examples/ch49_native $MODELS > "$OUT/native.txt"
    sleep 3
    # shellcheck disable=SC2086
    BADAK_GPU=full target/release/examples/ch49_native $MODELS > "$OUT/native_full.txt" 2> "$WORK/native_full.err"
    sleep 3
    target/release/examples/ch49_ramp > "$OUT/ramp.txt"
fi

# 2. 브라우저. 격리 헤더를 붙이는 서버를 띄운다.
python3 scripts/serve_web.py --isolate --port 8048 --reports "$REPORTS" > /dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2> /dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:8048/bench_gpu.html; do sleep 1; done

launch() {
    browser="$1" url="$2" profile="$WORK/profile_$1"
    rm -rf "$profile"
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

list="$(echo "$MODELS" | tr ' ' ',')"

# 페이지를 열고 끝날 때까지 기다린다. 페이지는 줄을 적을 때마다 보고한다.
# 마지막 줄이 done이면 끝난 것이다. 열 분 동안 새 줄이 없으면 브라우저가
# 죽은 것으로 본다.
measure() {
    browser="$1" name="$2" query="$3" page="${4:-bench_gpu.html}"
    rm -f "$REPORTS/$name.txt"
    sleep 3
    launch "$browser" "http://127.0.0.1:8048/$page?name=$name&run=$(date +%s)&$query"
    quiet=0
    seen=""
    until grep -q "^done" "$REPORTS/$name.txt" 2> /dev/null || [ "$quiet" -ge 600 ]; do
        sleep 2
        now="$(wc -c < "$REPORTS/$name.txt" 2> /dev/null || echo 0)"
        if [ "$now" = "$seen" ]; then quiet=$((quiet + 2)); else quiet=0; fi
        seen="$now"
    done
    close "$browser"
    if [ -f "$REPORTS/$name.txt" ]; then
        cp "$REPORTS/$name.txt" "$OUT/$name.txt"
        grep -q "^done" "$OUT/$name.txt" || echo "stalled (열 분 동안 새 줄이 없어 브라우저를 닫았습니다)" >> "$OUT/$name.txt"
    else
        echo "$name: 아무것도 보고하지 않았습니다" | tee "$OUT/$name.txt"
    fi
}

for browser in $BROWSERS; do
    # 48장의 CPU 측정을 이 장의 빌드로 한 번 더 한다. 스레드를 쓰는 빌드의
    # 메모리 한도를 4GB로 올린 것이 무엇을 바꿨는지 본다.
    if [ "$browser" = chrome ]; then
        measure chrome chrome_cpu "exhaust" bench.html
    fi
    if [ "$browser" = firefox ]; then
        # Firefox는 GPU의 답을 한 번 기다리는 데 0.1초가 걸린다(본문). 선형
        # 층마다 답을 기다리는 방식으로는 모델 하나를 재는 데 몇 시간이
        # 든다. 커널과 CPU는 그대로 재고, GPU에 맡긴 모델은 가장 작은 것을
        # 한 번만 돌린다.
        measure firefox firefox "models=$list&modes=cpu"
        measure firefox firefox_gpu "nokernels&models=smollm2_135m&modes=gpu-mlp&repeats=1&new=1&limit=1800"
    else
        measure "$browser" "$browser" "models=$list"
    fi
    # 이 책의 맥에 깔린 Edge(153)는 SmolLM2-360M을 돌리다가 죽곤 한다(본문).
    # 그때를 위해 그 뒤의 모델을 따로도 잰다.
    if [ "$browser" = edge ]; then
        measure edge edge_llama "nokernels&models=llama32_1b"
    fi
done
echo "끝났습니다: $OUT"
