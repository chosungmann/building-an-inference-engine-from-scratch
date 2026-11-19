#!/bin/sh
# 31장: 우리 엔진과 LiteRT, ONNX Runtime, NCNN, MNN을 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch31.sh [차례 수]
#
# 먼저 reference/ch31_convert.py 로 NCNN과 MNN의 모델을 만들어 둔다.
# reference/requirements_ch31.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch31/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch31"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch31_bench --example ch31_layers --example ch29_bench
cargo build --release -p badak_core --example ch31_sme

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for model in resnet50 mobilenetv2; do
        for threads in 1 4 12; do
            for kernel in neon sme; do
                sleep "$REST"
                BADAK_THREADS="$threads" target/release/examples/ch31_bench "$model" "$kernel" >> "$result"
            done
        done
    done
    for threads in 1 12; do
        sleep "$REST"
        BADAK_THREADS="$threads" target/release/examples/ch31_layers > "$OUT/badak_layers_${threads}_$round.txt"
        sleep "$REST"
        BADAK_THREADS="$threads" target/release/examples/ch29_bench > "$OUT/badak_int8_${threads}_$round.txt"
    done
    sleep "$REST"
    BADAK_THREADS=1 target/release/examples/ch31_sme > "$OUT/kernel_$round.txt"
    REST="$REST" "$PYTHON" reference/ch31_time.py 2> /dev/null | grep -v "^The device\|^MNN use" > "$OUT/engines_$round.txt"
    REST=1 "$PYTHON" reference/ch31_layers.py 2> /dev/null | grep "^층\|^스레드\|^conv\|^depthwise" > "$OUT/layers_$round.txt"
    echo "차례 $round"
    cat "$result" "$OUT/engines_$round.txt"
    round=$(( round + 1 ))
done
