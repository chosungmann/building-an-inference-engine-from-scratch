#!/bin/sh
# 35장: U-Net 한 번과 디코더 한 번의 시간을 PyTorch와 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch35.sh [차례 수]
#
# reference/requirements_ch35.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch35/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch35"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch35_bench

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for threads in 1 4 12; do
        for path in "" fast; do
            sleep "$REST"
            BADAK_THREADS="$threads" target/release/examples/ch35_bench $path profile >> "$result"
        done
    done
    REST="$REST" "$PYTHON" reference/ch35_time.py 2> /dev/null > "$OUT/python_$round.txt"
    echo "차례 $round"
    grep "중앙값" "$result"
    cat "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
