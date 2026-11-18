#!/bin/sh
# 22장: 가중치를 두는 방식마다 프로세스의 메모리를 두 가지 숫자로 잰다.
#
# 사용법: sh scripts/measure_ch22_memory.sh [모델 디렉터리의 이름]
#
# macOS의 /usr/bin/time -l이 알려 주는 두 숫자를 나란히 적는다.
#   올라온 메모리: maximum resident set size. 파일을 비춘 쪽까지 센다.
#   지켜야 하는 메모리: peak memory footprint. 이 프로세스만의 메모리다.
# 먼저 cargo build --release -p badak_models --examples.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-llama32_1b}"
TOOL="$ROOT/target/release/examples/ch20_load"

for format in f32 stored q8_0 q4_0; do
    report="$(BADAK_WEIGHTS="$format" /usr/bin/time -l "$TOOL" "$NAME" 2>&1)"
    line="$(echo "$report" | head -1)"
    resident="$(echo "$report" | awk '/maximum resident set size/ {printf "%.2f", $1 / 1e9}')"
    peak="$(echo "$report" | awk '/peak memory footprint/ {printf "%.2f", $1 / 1e9}')"
    echo "$line, 올라온 메모리 ${resident}GB, 지켜야 하는 메모리 ${peak}GB"
done
