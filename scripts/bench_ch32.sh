#!/bin/sh
# 32장: ViT-Tiny와 YOLOX-Tiny의 시간을 PyTorch, ONNX Runtime과 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch32.sh [차례 수]
#
# reference/requirements_ch32.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch32/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch32"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch32_bench
cargo build --release -p badak_models --example ch32_vit_bench

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for threads in 1 4 12; do
        for kernel in neon sme; do
            sleep "$REST"
            BADAK_THREADS="$threads" target/release/examples/ch32_bench "$kernel" >> "$result"
        done
        for path in basic fast; do
            sleep "$REST"
            BADAK_THREADS="$threads" target/release/examples/ch32_vit_bench "$path" >> "$result"
        done
    done
    REST="$REST" "$PYTHON" reference/ch32_time.py 2> /dev/null > "$OUT/python_$round.txt"
    echo "차례 $round"
    grep "중앙값" "$result"
    cat "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done

# BASE=1 이면 ViT-B/16도 잰다(`sh scripts/download_ch32.sh base` 뒤에).
# reference/ch32_time.py 는 가중치 파일이 있으면 ViT-B/16을 함께 잰다.
if [ "${BASE:-0}" = "1" ]; then
    result="$OUT/badak_base.txt"
    : > "$result"
    round=1
    while [ "$round" -le "$rounds" ]; do
        for threads in 1 4 12; do
            for path in basic fast; do
                sleep "$REST"
                BADAK_THREADS="$threads" target/release/examples/ch32_vit_bench base "$path" >> "$result"
            done
        done
        round=$(( round + 1 ))
    done
    REST="$REST" "$PYTHON" reference/ch32_time.py 2> /dev/null > "$OUT/python_base.txt"
    cat "$result" "$OUT/python_base.txt"
fi
