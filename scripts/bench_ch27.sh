#!/bin/sh
# 27장: 합성곱 모델의 시간을 우리 엔진, LiteRT, PyTorch로 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch27.sh [차례 수]
#
# LiteRT와 PyTorch는 Python에서 부른다. reference/requirements_ch27.txt 를
# 설치한 Python을 PYTHON으로 알려 준다. 실행마다 앞에서 REST초(기본 3초)
# 쉰다(22.12절). 결과는 bench/ch27/ 에 차례마다 한 파일씩 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch27"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch27_bench

round=1
while [ "$round" -le "$rounds" ]; do
    for model in resnet50 mobilenetv2; do
        sleep "$REST"
        target/release/examples/ch27_bench "$model" > "$OUT/badak_${model}_$round.txt"
    done
    sleep "$REST"
    "$PYTHON" reference/ch27_time.py 2> /dev/null > "$OUT/python_$round.txt"
    echo "차례 $round"
    grep -h "스레드 12개\|스레드  1개" "$OUT/badak_resnet50_$round.txt" "$OUT/badak_mobilenetv2_$round.txt"
    grep "LiteRT\|PyTorch" "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
