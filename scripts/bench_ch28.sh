#!/bin/sh
# 28장: 합성곱 커널을 고친 뒤의 시간을 LiteRT, PyTorch와 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch28.sh [차례 수]
#
# 27장의 scripts/bench_ch27.sh 와 같은 일을 하고, 층마다의 표
# (ch28_layers)를 더한다. 결과는 bench/ch28/ 에 남는다. PYTHON에
# reference/requirements_ch27.txt 를 설치한 Python을 알려 준다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch28"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch27_bench --example ch28_layers

round=1
while [ "$round" -le "$rounds" ]; do
    for model in resnet50 mobilenetv2; do
        sleep "$REST"
        target/release/examples/ch27_bench "$model" > "$OUT/badak_${model}_$round.txt"
        for threads in 1 12; do
            for method in whole tiled indirect winograd auto; do
                sleep "$REST"
                target/release/examples/ch28_layers "$model" "$threads" "$method" \
                    > "$OUT/layers_${model}_${threads}_${method}_$round.txt"
            done
        done
    done
    sleep "$REST"
    "$PYTHON" reference/ch27_time.py 2> /dev/null > "$OUT/python_$round.txt"
    echo "차례 $round"
    grep -h "스레드 12개\|스레드  1개" "$OUT/badak_resnet50_$round.txt" "$OUT/badak_mobilenetv2_$round.txt"
    for file in "$OUT"/layers_resnet50_*_"$round".txt; do
        head -1 "$file"
    done
    grep "LiteRT\|PyTorch" "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
