#!/bin/sh
# 34장: 임베딩의 처리량과 지연 시간을 ONNX Runtime, PyTorch와 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch34.sh [차례 수]
#
# reference/requirements_ch34.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch34/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch34"
BENCH="target/release/examples/ch34_bench"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch34_bench

# 스레드 수와 나머지 인자를 받아 한 번 잰다.
run() {
    threads="$1"
    shift
    sleep "$REST"
    BADAK_THREADS="$threads" "$BENCH" "$@" >> "$result"
}

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for path in "" fast; do
        # 글 하나의 시간.
        for threads in 1 4 12; do
            run "$threads" latency $path
        done
        # 스레드 하나: 묶는 방법만 바꾼다.
        run 1 bulk 1 ragged given inside $path
        run 1 bulk 32 padded given inside $path
        run 1 bulk 32 padded sorted inside $path
        run 1 bulk 32 ragged given inside $path
        run 1 bulk 32 ragged sorted inside $path
        # 스레드 열둘: 스레드를 어디에 쓰는지와 배치의 크기.
        run 12 bulk 32 ragged given inside $path
        run 12 bulk 32 padded given across $path
        run 12 bulk 32 padded sorted across $path
        run 12 bulk 32 ragged given across $path
        for batch in 1 8 32 128; do
            run 12 bulk "$batch" ragged sorted across $path
        done
        # 스레드의 수.
        for threads in 2 4 8; do
            run "$threads" bulk 32 ragged sorted across $path
        done
    done
    run 1 tokenize
    for threads in 1 12; do
        run "$threads" search
        run "$threads" search 1000000
    done
    REST="$REST" "$PYTHON" reference/ch34_time.py 2> /dev/null > "$OUT/python_$round.txt"
    echo "차례 $round"
    cat "$result" "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
