#!/bin/sh
# 43장: 기기 위의 제약(크기, 시작 시간, 메모리, 작은 코어, 오래 돌리기)을 잰다.
#
# 사용법: sh scripts/bench_ch43.sh [차례 수]
#         ONLY=sizes sh scripts/bench_ch43.sh   (크기만 다시 잰다)
#
# 이 맥에서 잰다. 폰이 아니다. macOS의 `/usr/bin/time -l`, `footprint`,
# `taskpolicy`, `pmset`을 쓴다. 관리자 권한은 필요 없다.
# 실행마다 앞에서 REST초(기본 3초) 쉰다(22.12절). 결과는 bench/ch43/ 에 남는다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REST="${REST:-3}"
rounds="${1:-3}"
OUT="$ROOT/bench/ch43"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 1. 크기. 빌드 설정을 바꿔 가며 실행 파일 셋을 빌드하고 크기와 속도를 잰다.
#    저장소의 Cargo.toml은 건드리지 않고 --config 로 설정을 덮어쓴다.
# ---------------------------------------------------------------------------
sizes="$OUT/sizes.txt"
: > "$sizes"
build() {
    name="$1"; shift
    cargo build --release --target-dir "target/ch43/$name" "$@" \
        -p badak_graph --example ch43_device \
        -p badak_models --example ch43_llm \
        -p badak_cli --bin badak > /dev/null 2>&1
    bin="target/ch43/$name/release"
    for pair in "device $bin/examples/ch43_device" "llm $bin/examples/ch43_llm" "badak $bin/badak"; do
        set -- $pair
        echo "size $name $1 $(wc -c < "$2" | tr -d ' ')" >> "$sizes"
    done
    for run in 1 2 3; do
        sleep "$REST"
        "$bin/examples/ch43_device" | sed "s/^/speed $name /" >> "$sizes"
        sleep "$REST"
        BADAK_THREADS=12 "$bin/examples/ch43_llm" | sed "s/^/speed $name /" >> "$sizes"
    done
}
STRIP='--config profile.release.strip=true'
LTO='--config profile.release.lto="fat" --config profile.release.codegen-units=1'
ABORT='--config profile.release.panic="abort"'
# shellcheck disable=SC2086
{
    build default
    build strip $STRIP
    build lto $STRIP $LTO
    build abort $STRIP $LTO $ABORT
    build size_s $STRIP $LTO $ABORT --config 'profile.release.opt-level="s"'
    build size_z $STRIP $LTO $ABORT --config 'profile.release.opt-level="z"'
}

# ONLY=sizes 를 주면 여기서 끝낸다.
[ "${ONLY:-}" = "sizes" ] && exit 0

# ---------------------------------------------------------------------------
# 2. 시작 시간과 메모리. 프로세스를 새로 띄워 한 번 돌린다.
# ---------------------------------------------------------------------------
cargo build --release -p badak_graph --example ch43_device -p badak_models --example ch43_llm
DEVICE="$ROOT/target/release/examples/ch43_device"
LLM="$ROOT/target/release/examples/ch43_llm"

# 명령 하나를 돌려, 프로그램이 적은 줄 뒤에 운영체제가 잰 메모리를 붙인다.
measure() {
    label="$1"; shift
    sleep "$REST"
    /usr/bin/time -l "$@" > "$OUT/run.out" 2> "$OUT/run.err"
    resident=$(awk '/maximum resident set size/ {print $1}' "$OUT/run.err")
    footprint=$(awk '/peak memory footprint/ {print $1}' "$OUT/run.err")
    wall=$(awk '/real/ {print $1}' "$OUT/run.err")
    sed "s/$/ 프로세스 $wall 상주 $resident 발자국 $footprint/; s/^/$label /" "$OUT/run.out" >> "$result"
    rm -f "$OUT/run.out" "$OUT/run.err"
}

# 프로세스를 붙잡아 두고 메모리가 무엇으로 이루어졌는지 본다.
inside() {
    label="$1"; shift
    sleep "$REST"
    "$@" --hold 4 > /dev/null &
    pid=$!
    sleep 2.5
    footprint "$pid" 2> /dev/null | awk -v label="$label" '
        /mapped file/ { file = $3 " " $4 }
        /TOTAL/ { dirty = $1 " " $2; clean = $3 " " $4 }
        /phys_footprint:/ { now = $2 " " $3 }
        END { print label " 안쪽 더러운페이지 " dirty " 깨끗한페이지 " clean " 비춘파일 " file " 발자국 " now }' >> "$result"
    wait "$pid" || true
}

round=1
while [ "$round" -le "$rounds" ]; do
    result="$OUT/round_$round.txt"
    : > "$result"

    # 사진 분류. 비추기와 읽기, f32와 int8.
    for model in mobilenetv2 resnet50; do
        for how in "" "--read"; do
            # shellcheck disable=SC2086
            measure "p12" "$DEVICE" "$model" $how
            # shellcheck disable=SC2086
            inside "p12 device $model$how" "$DEVICE" "$model" $how
        done
    done
    measure "p12" "$DEVICE" mobilenetv2 --int8
    inside "p12 device mobilenetv2--int8" "$DEVICE" mobilenetv2 --int8

    # 언어 모델. 가중치를 메모리에 두는 방식을 바꿔 가며.
    for model in smollm2_135m llama32_1b; do
        [ -f "$MODELS/$model/model.safetensors" ] || continue
        for weights in stored f32 q8_0 q4_0; do
            measure "p12" env BADAK_THREADS=12 BADAK_WEIGHTS="$weights" "$LLM" "$model"
            inside "p12 llm $model $weights" env BADAK_THREADS=12 BADAK_WEIGHTS="$weights" "$LLM" "$model"
        done
    done

    # 3. 큰 코어와 작은 코어. taskpolicy -b 는 프로세스를 작은 코어에 묶는다.
    for model in smollm2_135m llama32_1b; do
        [ -f "$MODELS/$model/model.safetensors" ] || continue
        measure "p4" env BADAK_THREADS=4 "$LLM" "$model"
        measure "e4" taskpolicy -b env BADAK_THREADS=4 "$LLM" "$model"
    done
    measure "p4" env BADAK_THREADS=4 "$DEVICE" resnet50
    measure "e4" taskpolicy -b env BADAK_THREADS=4 "$DEVICE" resnet50

    # 4. 오래 돌리기. 3분 동안 쉬지 않고 쓰면서 10초마다 속도를 적는다.
    sleep "$REST"
    echo "therm 앞 $(pmset -g therm | tr '\n' ' ')" >> "$result"
    BADAK_THREADS=12 "$LLM" smollm2_135m --seconds 180 | grep sustained >> "$result"
    echo "therm 뒤 $(pmset -g therm | tr '\n' ' ')" >> "$result"

    echo "차례 $round"
    round=$(( round + 1 ))
done
