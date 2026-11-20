#!/bin/sh
# 41장: 문법에 묶은 생성, 추측 디코딩, LoRA 어댑터를 잰다.
#
# 사용법: sh scripts/bench_ch41.sh [차례 수]
#
# 먼저 준비할 것:
#   sh scripts/download_ch41.sh                (추측 디코딩의 큰 모델 둘)
#   python reference/ch41_train_lora.py        (LoRA 어댑터와 그 정답)
# models/llama32_1b 가 있으면 문법에 묶은 생성을 그것으로도 잰다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch41/ 에
# 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch41"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch41_json --example ch41_speculate --example ch41_lora
BIN="$ROOT/target/release/examples"

round=1
while [ "$round" -le "$rounds" ]; do
    # 1. 문법에 묶은 생성. 답들은 Python으로 다시 검사할 수 있게 따로 적는다.
    for model in smollm2_135m llama32_1b; do
        [ -f "$MODELS/$model/model.safetensors" ] || continue
        sleep "$REST"
        # 답을 적는 파일은 저장소 안에서의 경로로 준다. 출력에 그대로 찍힌다.
        BADAK_THREADS=12 "$BIN/ch41_json" --model "$model" --out "bench/ch41/answers_$model.jsonl" \
            > "$OUT/json_${model}_$round.txt"
    done

    # 2. 추측 디코딩. 큰 모델을 바꿔 가며, 스레드 12개와 4개로.
    for run in "smollm2_360m 12" "smollm2_1_7b 12" "smollm2_1_7b 4"; do
        set -- $run
        [ -f "$MODELS/$1/model.safetensors" ] || continue
        sleep "$REST"
        BADAK_THREADS="$2" "$BIN/ch41_speculate" --target "$1" --repeats 3 \
            > "$OUT/speculate_$1_$2_$round.txt"
    done

    # 3. LoRA 어댑터.
    if [ -f "$MODELS/smollm2_135m_lora/adapter_model.safetensors" ]; then
        sleep "$REST"
        BADAK_THREADS=12 "$BIN/ch41_lora" > "$OUT/lora_$round.txt"
    fi
    echo "차례 $round"
    round=$(( round + 1 ))
done
