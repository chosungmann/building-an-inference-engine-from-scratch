#!/bin/sh
# 26장: 그래프 해석기와 LiteRT가 사진 한 장을 추론하는 시간을 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch26.sh [차례 수]
#
# LiteRT는 Python에서 부른다. reference/requirements_ch26.txt 를 설치한
# Python을 PYTHON으로 알려 준다. 실행마다 앞에서 REST초(기본 3초) 쉰다
# (22.12절). 결과는 bench/ch26/ 에 차례마다 한 파일씩 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch26"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch26_bench

round=1
while [ "$round" -le "$rounds" ]; do
    sleep "$REST"
    target/release/examples/ch26_bench > "$OUT/badak_$round.txt"
    sleep "$REST"
    "$PYTHON" reference/ch26_time_litert.py 2> /dev/null > "$OUT/litert_$round.txt"
    echo "차례 $round"
    grep "추론, 한 장" "$OUT/badak_$round.txt"
    grep "invoke" "$OUT/litert_$round.txt"
    round=$(( round + 1 ))
done
