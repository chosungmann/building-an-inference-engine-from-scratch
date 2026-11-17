#!/bin/sh
# 실행 중인 badak을 macOS의 sample 도구로 들여다본다 (14장).
#
# 사용법: sh scripts/profile_sample.sh [모델 디렉터리의 이름] [초]
#
# 글을 이어 쓰는 예제를 띄워 놓고, 정해진 시간 동안 1밀리초마다 "지금 어느
# 함수를 실행 중인가"를 받아 적게 한다. 결과는 scripts/sample_report.py 로
# 줄여서 보여 준다.
# macOS 전용이다. 다른 운영체제의 도구는 14.8절에 적었다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
SECONDS_TO_SAMPLE="${2:-5}"
OUT="$(mktemp -t badak_sample)"

cd "$ROOT"
cargo build -q --release -p badak_models --example ch11_generate
# 표본을 뜨는 동안 끝나지 않도록 토큰을 넉넉히 만들게 한다.
"$ROOT/target/release/examples/ch11_generate" "$NAME" "Once upon a time" 4000 >/dev/null 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
# 가중치를 읽는 동안은 건너뛴다.
sleep 3
sample "$PID" "$SECONDS_TO_SAMPLE" -file "$OUT" >/dev/null 2>&1

python3 "$ROOT/scripts/sample_report.py" "$OUT"
rm -f "$OUT"
