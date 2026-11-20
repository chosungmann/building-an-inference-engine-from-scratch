#!/bin/sh
# 39장: 추론 서버를 잰다.
#
# 사용법: sh scripts/bench_ch39.sh [차례 수]
#
# 견줄 서버는 llama.cpp의 llama-server다. 먼저 빌드해 둔다.
#   sh scripts/build_llama_cpp.sh --plain llama-server
# models/llama32_1b 가 있으면 그것도 잰다. 실행마다 앞에서 REST초(기본
# 3초) 쉰다(22.12절). 결과는 bench/ch39/ 에 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch39"
MODELS="${BADAK_MODELS:-$ROOT/models}"
LLAMA="$ROOT/third_party/llama.cpp/build_plain/bin/llama-server"
PORT="${PORT:-8039}"
SLOTS="$(mktemp -d)"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_server --example ch39_bench
cargo build --release -p badak_cli
BENCH="$ROOT/target/release/examples/ch39_bench"
BADAK="$ROOT/target/release/badak"

models="smollm2_135m"
if [ -f "$MODELS/llama32_1b/model.safetensors" ]; then
    models="$models llama32_1b"
fi

# 측정 하나를 돌려, 줄마다 앞에 이름표를 붙여 결과에 더한다. 측정이
# 실패하면 스크립트가 멈춘다(관으로 바로 이으면 실패를 놓친다).
run() {
    label="$1"; shift
    "$@" > "$OUT/run.tmp"
    sed "s/^/$label /" "$OUT/run.tmp" >> "$result"
    rm -f "$OUT/run.tmp"
}

# 띄워 둔 서버를 끈다. 스크립트가 중간에 멈춰도 끈다.
server=""
stop() {
    if [ -n "$server" ]; then
        kill "$server" 2> /dev/null || true
        wait "$server" 2> /dev/null || true
        server=""
    fi
}
trap stop EXIT

# 서버가 요청을 받을 때까지 기다린다. llama-server는 모델을 읽는 동안
# 503으로 답하므로, 연결이 되는지가 아니라 200이 오는지를 본다(-f).
ready() {
    tries=0
    until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        tries=$(( tries + 1 ))
        [ "$tries" -lt 300 ] || { echo "서버가 뜨지 않았습니다" >&2; exit 1; }
        sleep 0.1
    done
}

# llama-server를 띄운다. 자리(slot)는 하나, 앞의 캐시를 메모리에 모아 두는
# 기능은 끈다(-cram 0). 이 책의 서버와 같은 조건이다.
llama() {
    model="$1"; threads="$2"; shift 2
    "$LLAMA" -m "$MODELS/$model/model_f32.gguf" -t "$threads" -np 1 -c 16384 -cram 0 \
        --slot-save-path "$SLOTS/" --host 127.0.0.1 --port "$PORT" "$@" > /dev/null 2>&1 &
    server=$!
    ready
}

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/round_$round.txt"
    : > "$result"
    for model in $models; do
        for threads in 4 12; do
            # 1. 이 책의 서버. 프로그램 안에서 띄우고 진짜 연결로 부른다.
            for kv in f32 f16; do
                sleep "$REST"
                part=""
                [ "$kv" = "f16" ] && part="--part persist"
                # shellcheck disable=SC2086
                run "badak-$kv $model $threads" env BADAK_THREADS="$threads" BADAK_KV="$kv" \
                    "$BENCH" --model "$model" --repeats 3 $part
            done
            # 2. llama.cpp의 서버.
            if [ -x "$LLAMA" ]; then
                sleep "$REST"
                llama "$model" "$threads"
                run "llama $model $threads" "$BENCH" --address "127.0.0.1:$PORT" --llama --repeats 3
                stop
            fi
        done

        # 3. 작은 조각을 모았다가 보내는 네이글 알고리즘을 켜 둔 것과 끈 것.
        #    서버를 따로 띄워, 같은 조건에서 견준다.
        for nagle in off on; do
            sleep "$REST"
            if [ "$nagle" = "on" ]; then
                BADAK_NAGLE=1 BADAK_THREADS=12 "$BADAK" serve --model "$model" --port "$PORT" > /dev/null 2>&1 &
            else
                BADAK_THREADS=12 "$BADAK" serve --model "$model" --port "$PORT" > /dev/null 2>&1 &
            fi
            server=$!
            ready
            if [ "$nagle" = "on" ]; then
                run "badak-nagle-$nagle $model 12" env BADAK_NAGLE=1 \
                    "$BENCH" --address "127.0.0.1:$PORT" --repeats 3 --part overhead
            else
                run "badak-nagle-$nagle $model 12" \
                    "$BENCH" --address "127.0.0.1:$PORT" --repeats 3 --part overhead
            fi
            stop
        done

        # 4. 방금 띄운 서버의 첫 요청. 데워 둔 것과 그러지 않은 것.
        for warm in yes no; do
            flag=""
            [ "$warm" = "no" ] && flag="--no-warm"
            sleep "$REST"
            # shellcheck disable=SC2086
            BADAK_THREADS=12 "$BADAK" serve --model "$model" --port "$PORT" $flag > /dev/null 2>&1 &
            server=$!
            ready
            run "badak-warm-$warm $model 12" "$BENCH" --address "127.0.0.1:$PORT" --part first
            stop
            if [ -x "$LLAMA" ]; then
                flag="--warmup"
                [ "$warm" = "no" ] && flag="--no-warmup"
                sleep "$REST"
                llama "$model" 12 "$flag"
                run "llama-warm-$warm $model 12" "$BENCH" --address "127.0.0.1:$PORT" --part first
                stop
            fi
        done
    done
    echo "차례 $round"
    cat "$result"
    round=$(( round + 1 ))
done
rm -rf "$SLOTS"
