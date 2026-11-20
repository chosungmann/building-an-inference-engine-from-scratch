#!/bin/sh
# 45장: 그래프를 Core ML에 통째로 넘길 때와 나눠서 넘길 때를 잰다.
#
# 쓰는 법: [PYTHON=<파이썬>] sh scripts/bench_ch45.sh [차례의 수]
# 결과: bench/ch45/round_<n>.txt, bench/ch45/plan.txt, 그리고 PYTHON을 주면
#       bench/ch45/engines_<n>.txt
#
# macOS에서만 된다. 전원을 꽂고, 다른 일을 하지 않는 동안 돌린다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS="${1:-3}"
OUT="$ROOT/bench/ch45"
mkdir -p "$OUT"
cd "$ROOT"

sh scripts/build_coreml.sh > /dev/null
cargo build --quiet --release -p badak_graph --example ch45_bench
BENCH="$ROOT/target/release/examples/ch45_bench"

# 재기 전에 3초 쉰다.
run() {
    sleep 3
    "$BENCH" "$@"
}

# Core ML이 층마다 고른 장치는 한 번만 묻는다.
{
    for model in mobilenetv2 resnet50; do
        run plan "$model"
    done
} > "$OUT/plan.txt"

for round in $(seq "$ROUNDS"); do
    {
        for model in mobilenetv2 resnet50; do
            run whole "$model"
            run split "$model"
        done
    } > "$OUT/round_$round.txt"
    # 31장의 엔진들. 파이썬 환경(reference/requirements_ch45.txt)이 있을 때만.
    if [ -n "${PYTHON:-}" ]; then
        "$PYTHON" reference/ch45_time.py 2> /dev/null | grep -E '^(#|engine) ' \
            > "$OUT/engines_$round.txt"
    fi
    echo "차례 $round 끝"
done
