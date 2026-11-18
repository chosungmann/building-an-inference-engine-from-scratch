#!/bin/sh
# 22장: 엔진 여럿의 속도를 번갈아 잰다.
#
# 사용법: sh scripts/compare_engines.sh <가중치 방식> <차례 수> <이름=명령> ...
#
#   sh scripts/compare_engines.sh q8_0 6 \
#       "21장=/tmp/ch21/target/release/examples/ch14_baseline" \
#       "아레나없이=env BADAK_ARENA=off target/release/examples/ch14_baseline" \
#       "아레나=target/release/examples/ch14_baseline"
#
# 명령은 14장의 기준 측정(ch14_baseline)을 실행하는 것이어야 한다. 차례마다
# 엔진들을 한 번씩 돌리되 시작하는 엔진을 바꿔 가고, 실행마다 앞에서
# REST초(기본 3초) 쉰다. 쉬지 않으면 앞의 실행이 달궈 놓은 기계에서 재게
# 되고, 읽는 시간이 긴 엔진만 그동안 쉬게 되어 공평하지 않다(22.12절).
#
# 고치기 전의 엔진은 git worktree로 받아 따로 빌드한다.
#   git worktree add /tmp/ch21 <커밋>
#   (cd /tmp/ch21 && cargo build --release -p badak_models --example ch14_baseline)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BADAK_MODELS="${BADAK_MODELS:-$ROOT/models}"
MODEL="${MODEL:-llama32_1b}"
REPETITIONS="${REPETITIONS:-10}"
REST="${REST:-3}"

weights="$1"
rounds="$2"
shift 2
count=$#

echo "가중치 $weights, 실행마다 앞에서 ${REST}초 쉼, 되풀이 ${REPETITIONS}번의 중앙값 (토큰/초)"
printf '%-6s %-14s %14s %12s\n' "차례" "엔진" "프롬프트 처리" "토큰 생성"
results="$(mktemp)"
trap 'rm -f "$results"' EXIT

round=1
while [ "$round" -le "$rounds" ]; do
    # 이번 차례에 처음 돌릴 엔진의 번호. 차례마다 하나씩 밀린다.
    first=$(( (round - 1) % count ))
    step=0
    while [ "$step" -lt "$count" ]; do
        index=$(( (first + step) % count + 1 ))
        eval "entry=\${$index}"
        name="${entry%%=*}"
        command="${entry#*=}"
        sleep "$REST"
        # shellcheck disable=SC2086
        report="$(BADAK_WEIGHTS="$weights" $command "$MODEL" "$REPETITIONS")"
        prompt="$(echo "$report" | awk '/^pp256/ {print $2}')"
        generation="$(echo "$report" | awk '/^tg32/ {print $2}')"
        printf '%-6s %-14s %14.1f %12.1f\n' "$round" "$name" "$prompt" "$generation"
        echo "$name $prompt $generation" >> "$results"
        step=$(( step + 1 ))
    done
    round=$(( round + 1 ))
done

echo
printf '%-6s %-14s %14s %12s\n' "중앙값" "엔진" "프롬프트 처리" "토큰 생성"
for entry in "$@"; do
    name="${entry%%=*}"
    for column in 2 3; do
        grep "^$name " "$results" | awk -v column="$column" '{print $column}' | sort -n |
            awk '{values[NR] = $1} END {
                middle = int((NR + 1) / 2)
                print (NR % 2) ? values[middle] : (values[middle] + values[middle + 1]) / 2
            }'
    done | paste -s -d ' ' - | awk -v name="$name" '{printf "%-6s %-14s %14.1f %12.1f\n", "", name, $1, $2}'
done
