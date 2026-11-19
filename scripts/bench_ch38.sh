#!/bin/sh
# 38장: 음성 대화 파이프라인의 구간마다의 시간을 잰다.
#
# 사용법: sh scripts/bench_ch38.sh [차례 수]
#
# reference/requirements_ch38.txt 를 설치한 Python을 PYTHON으로 알려 준다.
# models/llama32_1b 가 있으면 답을 쓰는 모델을 그것으로 바꾼 것도 잰다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch38/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch38"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch38_talk

chats="smollm2_135m"
if [ -f "$MODELS/llama32_1b/model.safetensors" ]; then
    chats="$chats llama32_1b"
fi

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    : > "$result"
    for chat in $chats; do
        for threads in 1 4 12; do
            for mode in sentence whole; do
                flag=""
                [ "$mode" = "whole" ] && flag="--whole"
                sleep "$REST"
                BADAK_THREADS="$threads" target/release/examples/ch38_talk --chat "$chat" $flag \
                    --out "$OUT/answer.wav" | grep "^talk" | sed "s/^talk/talk $chat $mode/" >> "$result"
            done
        done
    done
    rm -f "$OUT/answer.wav"
    (cd reference && REST="$REST" "$PYTHON" ch38_time.py 2> /dev/null) > "$OUT/python_$round.txt"
    echo "차례 $round"
    cat "$result" "$OUT/python_$round.txt"
    round=$(( round + 1 ))
done
