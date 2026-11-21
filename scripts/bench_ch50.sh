#!/bin/sh
# 50장의 측정: badak과 웹 엔진 셋(Transformers.js, ONNX Runtime Web, WebLLM)을
# 같은 브라우저에서 같은 모델로 돌린다.
#
# 쓰는 법: sh scripts/bench_ch50.sh [chrome edge firefox safari 가운데 몇]
# 결과: bench/ch50/<브라우저>_<측정>.txt
#
#   small    SmolLM2-135M에 같은 질문을 한다 (엔진 넷)
#   large    Llama 3.2 1B에 같은 질문을 한다 (badak과 WebLLM)
#   speech   Whisper tiny로 같은 소리를 받아쓴다 (badak과 Transformers.js).
#            그리고 브라우저에 WebNN과 내장 AI가 있는지 묻는다
#
# 필요한 것: sh scripts/download_ch50.sh, download_smollm2.sh,
# download_ch33.sh(Whisper), download_llama32.sh(Llama 3.2 1B),
# rustup target add wasm32-wasip1 wasm32-wasip1-threads, python3.
# 브라우저는 새 프로필로 띄우고 끝나면 닫는다. Safari만은 쓰던 프로필로 열린다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch50"
WORK="$ROOT/target/web_bench"
REPORTS="$ROOT/target/web_reports/ch50"
BROWSERS="${*:-chrome edge firefox safari}"
cd "$ROOT"
mkdir -p "$WORK" "$REPORTS" "$OUT"

sh scripts/build_web.sh target/web_site > "$OUT/sizes.txt"
cp crates/badak_web/web/bench_engines.html crates/badak_web/web/bench_speech.html target/web_site/

python3 scripts/serve_web.py --isolate --port 8048 --reports "$REPORTS" > /dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2> /dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:8048/bench_engines.html; do sleep 1; done

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

# 페이지를 열고 끝날 때까지 기다린다. 페이지는 줄을 적을 때마다 보고한다.
# 마지막 줄이 done이면 끝난 것이다. 열 분 동안 새 줄이 없으면 닫는다.
measure() {
    browser="$1" name="$2" page="$3" query="$4"
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
    measure "$browser" "${browser}_small" bench_engines.html "group=small"
    measure "$browser" "${browser}_large" bench_engines.html "group=large"
    measure "$browser" "${browser}_speech" bench_speech.html ""
done
echo "끝났습니다: $OUT"
