#!/bin/sh
# 20장: 가중치를 두는 방식마다 프로세스가 쓰는 메모리의 최대치를 잰다.
#
# 사용법: sh scripts/measure_ch20_memory.sh [모델 디렉터리의 이름]
#
# macOS의 /usr/bin/time -l이 알려 주는 "peak memory footprint"를 적는다
# (10.3절과 같은 방법). 먼저 cargo build --release -p badak_models --examples.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
TOOL="$ROOT/target/release/examples/ch20_load"

for format in f32 stored q8_0 q4_0; do
    # time의 보고는 표준 오류로 나온다.
    report="$(BADAK_WEIGHTS="$format" /usr/bin/time -l "$TOOL" "$NAME" 2>&1)"
    line="$(echo "$report" | head -1)"
    peak="$(echo "$report" | awk '/peak memory footprint/ {printf "%.2f", $1 / 1e9}')"
    echo "$line, 메모리 최대치 ${peak}GB"
done
