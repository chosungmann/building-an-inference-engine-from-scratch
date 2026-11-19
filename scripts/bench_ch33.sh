#!/bin/sh
# 33장: Whisper-tiny의 단계마다의 시간을 whisper.cpp와 번갈아 잰다.
#
# 사용법: sh scripts/bench_ch33.sh [차례 수]
#
# WHISPER_CPP에 whisper.cpp를 빌드한 디렉터리를 알려 준다. 그 안에
# build/bin/whisper-cli 와 ggml-tiny.bin 이 있어야 한다(부록 B). GPU를 끄고
# (-ng) 잰다. 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는
# bench/ch33/ 에 차례마다 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_CPP="${WHISPER_CPP:?whisper.cpp의 디렉터리를 WHISPER_CPP로 알려 주십시오}"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch33"
SOUND="${BADAK_MODELS:-$ROOT/models}/whisper/jfk.wav"
mkdir -p "$OUT"

cd "$ROOT"
cargo build --release -p badak_models --example ch33_bench

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/badak_$round.txt"
    other="$OUT/whisper_cpp_$round.txt"
    : > "$result"
    : > "$other"
    for threads in 1 4 12; do
        for path in basic fast; do
            sleep "$REST"
            BADAK_THREADS="$threads" target/release/examples/ch33_bench "$path" >> "$result"
        done
        for beam in 1 5; do
            sleep "$REST"
            echo "스레드 $threads, 빔 $beam" >> "$other"
            "$WHISPER_CPP/build/bin/whisper-cli" -m "$WHISPER_CPP/ggml-tiny.bin" -f "$SOUND" \
                -t "$threads" -bs "$beam" -bo 1 -nt -l en -ng 2>&1 \
                | grep "mel time\|encode time\|decode time\|batchd time\|total time" >> "$other"
        done
    done
    echo "차례 $round"
    cat "$result"
    grep "스레드\|encode time\|total time" "$other"
    round=$(( round + 1 ))
done
