#!/bin/sh
# 37장: 음성 합성의 단계마다의 시간을 PyTorch와 번갈아 잰다.
#
# 사용법: OUTETTS=<OuteTTS 저장소의 경로> sh scripts/bench_ch37.sh [차례 수]
#
# reference/requirements_ch37.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch37/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch37"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch37_bench

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for threads in 1 4 12; do
        for path in "" fast; do
            sleep "$REST"
            BADAK_THREADS="$threads" target/release/examples/ch37_bench $path >> "$result"
        done
    done
    (cd reference && REST="$REST" "$PYTHON" ch37_time.py 2> /dev/null) > "$OUT/python_$round.txt"
    echo "차례 $round"
    cat "$result" "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
sleep "$REST"
BADAK_THREADS=12 target/release/examples/ch37_bench chunks > "$OUT/chunks.txt"
cat "$OUT/chunks.txt"
