#!/bin/sh
# 30장: 그래프 최적화를 한 단계씩 더해 가며 시간을 잰다.
#
# 사용법: sh scripts/bench_ch30.sh [차례 수]
#
# 단계 하나를 한 번의 실행으로 재고, 실행마다 앞에서 REST초(기본 3초) 쉰다
# (22.12절). ONNX Runtime은 Python에서 부른다. reference/requirements_ch30.txt
# 를 설치한 Python을 PYTHON으로 알려 준다. 결과는 bench/ch30/ 에 차례마다 한
# 파일씩 남는다.
#
# 일부만 더 재려면 MODELS, THREADS, FIRST(첫 차례의 번호)를 준다. 그때는
# ONNX Runtime을 재지 않는다. 2밀리초짜리 측정은 차례마다 흔들림이 커서
# MobileNetV2의 스레드 12개 줄을 그렇게 네 차례 더 쟀다:
#   MODELS=mobilenetv2 THREADS=12 FIRST=4 sh scripts/bench_ch30.sh 7
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
MODELS="${MODELS:-resnet50 mobilenetv2}"
THREADS="${THREADS:-1 12}"
FIRST="${FIRST:-1}"
OUT="$ROOT/bench/ch30"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_graph --example ch30_bench

round="$FIRST"
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for model in $MODELS; do
        for threads in $THREADS; do
            for stage in 0 1 2 3 4 5; do
                sleep "$REST"
                BADAK_THREADS="$threads" target/release/examples/ch30_bench onnx "$model" "$stage" >> "$result"
            done
            for stage in 0 1 2; do
                sleep "$REST"
                BADAK_THREADS="$threads" target/release/examples/ch30_bench tflite "$model" "$stage" >> "$result"
            done
        done
    done
    echo "차례 $round"
    cat "$result"
    if [ "$FIRST" -eq 1 ]; then
        sleep "$REST"
        "$PYTHON" reference/ch30_time.py 2> /dev/null > "$OUT/onnxruntime_$round.txt"
        cat "$OUT/onnxruntime_$round.txt"
    fi
    round=$(( round + 1 ))
done
