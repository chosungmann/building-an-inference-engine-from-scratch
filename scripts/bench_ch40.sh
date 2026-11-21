#!/bin/sh
# 40장: 요청이 여럿일 때의 처리량과 지연을 잰다.
#
# 사용법: sh scripts/bench_ch40.sh [차례 수]
#
# 견줄 서버는 llama.cpp의 llama-server다(39장과 같은 빌드).
#   sh scripts/build_llama_cpp.sh --plain llama-server
# models/llama32_1b 가 있으면 그것도 잰다. 실행마다 앞에서 REST초(기본
# 3초) 쉰다(22.12절). 결과는 bench/ch40/ 에 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch40"
MODELS="${BADAK_MODELS:-$ROOT/models}"
LLAMA="$ROOT/third_party/llama.cpp/build_plain/bin/llama-server"
PORT="${PORT:-8040}"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_server --example ch40_bench
BENCH="$ROOT/target/release/examples/ch40_bench"

models="smollm2_135m"
if [ -f "$MODELS/llama32_1b/model.safetensors" ]; then
    models="$models llama32_1b"
fi

# 측정 하나를 돌려, 줄마다 앞에 이름표를 붙여 결과에 더한다.
run() {
    label="$1"; shift
    sleep "$REST"
    "$@" > "$OUT/run.tmp"
    sed "s/^/$label /" "$OUT/run.tmp" >> "$result"
    rm -f "$OUT/run.tmp"
}

server=""
stop() {
    if [ -n "$server" ]; then
        kill "$server" 2> /dev/null || true
        wait "$server" 2> /dev/null || true
        server=""
    fi
}
trap stop EXIT

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/round_$round.txt"
    : > "$result"
    for model in $models; do
        # 요청이 도착하는 빠르기(초당). 한 번에 하나씩 처리할 때 감당하는
        # 것보다 조금 느리게, 그리고 그보다 빠르게.
        if [ "$model" = "smollm2_135m" ]; then rates="4 8"; else rates="1 2"; fi
        for threads in 4 12; do
            bench() { label="$1"; shift; run "badak $model $threads $label" env BADAK_THREADS="$threads" "$BENCH" --model "$model" "$@"; }

            # 1. 배치를 키워 간다. 클라이언트의 수가 배치의 크기와 같다.
            bench serial --part closed --mode serial --clients 1 --requests 3
            for size in 1 2 4 8 16; do
                bench "continuous:$size" --part closed --mode "continuous:$size" --clients "$size" --requests 3
            done
            [ "$threads" = 12 ] || continue

            # 2. 클라이언트는 여덟으로 두고 묶는 방법을 바꾼다.
            for mode in serial static:4:100 static:8:100 continuous:2 continuous:4 continuous:8; do
                bench "$mode" --part closed --mode "$mode" --clients 8 --requests 3
            done

            # 3. 요청이 제멋대로 도착한다.
            for rate in $rates; do
                for mode in serial static:4:100 static:4:1000 continuous:8; do
                    bench "$mode" --part open --mode "$mode" --rate "$rate" --requests 30
                done
            done

            # 4. 답을 쓰는 요청 옆에서 긴 글을 읽는다.
            bench serial --part long --mode serial
            for chunk in 16 64 256 100000; do
                bench "chunk:$chunk" --part long --mode continuous:4 --chunk "$chunk"
            done

            # 5. 같은 지시문으로 시작하는 대화 여덟.
            bench serial --part shared --mode serial
            bench continuous:4 --part shared --mode continuous:4

            # 6. 요청 하나가 긴 글을 읽고 답을 쓴다. 이어진 캐시와 페이지
            #    캐시를 견준다. 끊어 읽지 않게 한다.
            bench serial --part deep --mode serial
            bench continuous:1 --part deep --mode continuous:1 --chunk 100000
        done

        # 7. llama.cpp의 서버. 자리를 여덟 두면 스스로 묶어 처리한다.
        if [ -x "$LLAMA" ]; then
            "$LLAMA" -m "$MODELS/$model/model_f32.gguf" -t 12 -np 8 -c 16384 -cram 0 \
                --host 127.0.0.1 --port "$PORT" > /dev/null 2>&1 &
            server=$!
            tries=0
            until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
                tries=$(( tries + 1 ))
                [ "$tries" -lt 300 ] || { echo "서버가 뜨지 않았습니다" >&2; exit 1; }
                sleep 0.1
            done
            for clients in 1 2 4 8; do
                run "llama $model 12 np8" "$BENCH" --address "127.0.0.1:$PORT" --part closed --clients "$clients" --requests 3
            done
            stop
        fi
    done
    echo "차례 $round"
    cat "$result"
    round=$(( round + 1 ))
done
