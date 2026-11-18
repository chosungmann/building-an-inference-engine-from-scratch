#!/bin/sh
# 23장: 가중치를 CPU에 둘 때와 GPU에 올릴 때 프로세스의 메모리를 잰다.
#
# 사용법: sh scripts/measure_ch23_memory.sh [모델 디렉터리의 이름]
#
# 22장과 같은 두 숫자를 적는다(올라온 메모리, 지켜야 하는 메모리).
# 먼저 cargo build --release -p badak_models --examples.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
TOOL="$ROOT/target/release/examples/ch23_load"

for place in cpu gpu; do
    report="$(/usr/bin/time -l "$TOOL" "$place" "$NAME" 2>&1)"
    line="$(echo "$report" | head -1)"
    resident="$(echo "$report" | awk '/maximum resident set size/ {printf "%.2f", $1 / 1e9}')"
    peak="$(echo "$report" | awk '/peak memory footprint/ {printf "%.2f", $1 / 1e9}')"
    echo "$line, 올라온 메모리 ${resident}GB, 지켜야 하는 메모리 ${peak}GB"
done
