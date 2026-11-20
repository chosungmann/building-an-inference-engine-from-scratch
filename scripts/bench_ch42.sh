#!/bin/sh
# 42장: 서버 여러 대를 띄워 놓고 잰다.
#
# 사용법: sh scripts/bench_ch42.sh [차례 수]
#
# models/llama32_1b 와 models/smollm2_1_7b(scripts/download_ch41.sh)가
# 있으면 그것도 잰다. 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절).
# 결과는 bench/ch42/ 에 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch42"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_cli
cargo build --release -p badak_server --example ch42_bench
BENCH="$ROOT/target/release/examples/ch42_bench"

# 측정 하나를 돌려, 줄마다 앞에 이름표를 붙여 결과에 더한다.
run() {
    label="$1"; shift
    sleep "$REST"
    "$@" > "$OUT/run.tmp"
    sed "s/^/$label /" "$OUT/run.tmp" >> "$result"
    rm -f "$OUT/run.tmp"
}
has() { [ -f "$MODELS/$1/model.safetensors" ]; }

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/round_$round.txt"
    : > "$result"

    # 1. 서버 넷 앞의 라우터. 보내는 방법과, 서버가 남겨 두는 캐시의 수를 바꾼다.
    for policy in round_robin least_busy prefix_hash longest_prefix; do
        for keep in 4 8; do
            run "smollm2_135m keep$keep" "$BENCH" --part route --model smollm2_135m \
                --policy "$policy" --keep "$keep"
        done
        if has llama32_1b; then
            run "llama32_1b keep4" "$BENCH" --part route --model llama32_1b --policy "$policy" --keep 4
        fi
    done

    # 2. 서버를 새로 띄워 첫 요청을 보낸다.
    for model in smollm2_135m smollm2_360m llama32_1b smollm2_1_7b; do
        has "$model" || continue
        for warm in yes no; do
            run "$model" "$BENCH" --part cold --model "$model" --warm "$warm"
        done
    done

    # 3. 글을 읽는 서버와 답을 쓰는 서버를 따로 둔다.
    for model in smollm2_135m llama32_1b; do
        has "$model" || continue
        run "$model" "$BENCH" --part split --model "$model"
    done
    echo "차례 $round"
    cat "$result"
    round=$(( round + 1 ))
done
