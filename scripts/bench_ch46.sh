#!/bin/sh
# 46장: 줄인 모델들의 품질과 속도를 엔진으로 잰다.
#
# 쓰는 법: sh scripts/bench_ch46.sh [차례의 수]
# 결과: bench/ch46/*.txt
#
# 먼저 `python reference/ch46_train.py`로 모델들을 만들어 둔다. 학습하지
# 않아도 되는 부분(prune_*.txt, kernel_*.txt, speed_*.txt)은 원래 모델만
# 있으면 된다. 전원을 꽂고, 다른 일을 하지 않는 동안 돌린다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS="${1:-3}"
OUT="$ROOT/bench/ch46"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"
cd "$ROOT"

cargo build --quiet --release -p badak_models --example ch46_prune --example ch46_quality
PRUNE="$ROOT/target/release/examples/ch46_prune"
QUALITY="$ROOT/target/release/examples/ch46_quality"

# 손실과 크기는 돌릴 때마다 같다. 한 번만 잰다.
"$PRUNE" loss > "$OUT/prune_loss.txt"
"$PRUNE" blocks > "$OUT/prune_blocks.txt"
"$PRUNE" size > "$OUT/prune_size.txt"

# 학습한 모델이 있으면 품질을 잰다.
TRAINED=""
for method in ft qat qat_kd drop drop_ft drop_kd pick pick_ft pick_kd pick_qat_kd prune_kd; do
    [ -f "$MODELS/smollm2_135m_$method/model.safetensors" ] && TRAINED="$TRAINED smollm2_135m_$method"
done
if [ -f "$MODELS/smollm2_135m_qat/fake_q4_0.safetensors" ]; then
    "$QUALITY" check > "$OUT/check.txt"
fi

# 손실은 한 번만 잰다. 7분쯤 걸린다.
# shellcheck disable=SC2086
"$QUALITY" 64 smollm2_135m $TRAINED > "$OUT/quality.txt"

# 속도는 흔들리므로 여러 차례 잰다.
for round in $(seq "$ROUNDS"); do
    sleep 3
    # shellcheck disable=SC2086
    "$QUALITY" rate smollm2_135m $TRAINED > "$OUT/rate_$round.txt"
    sleep 3
    "$PRUNE" kernel > "$OUT/kernel_$round.txt"
    "$PRUNE" speed > "$OUT/speed_$round.txt"
    echo "차례 $round 끝"
done
