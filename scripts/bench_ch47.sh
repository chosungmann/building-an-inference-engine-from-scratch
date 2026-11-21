#!/bin/sh
# 47장: 기기 위의 언어 모델을 여러 각도에서 잰다.
#
# 쓰는 법: [PYTHON=<파이썬>] sh scripts/bench_ch47.sh [부분 ...]
# 결과: bench/ch47/*.txt
#
# 부분 (주지 않으면 모두):
#   budget   메모리 예산의 계산과 실제 (badak)
#   system   운영체제에 든 모델 (macOS 26 이상)
#   engines  같은 모델(SmolLM2-135M)을 여러 엔진으로. llama.cpp는
#            scripts/build_llama_cpp.sh 로, LiteRT-LM과 ExecuTorch는
#            reference/requirements_ch47.txt 로 준비하고 PYTHON에 그 환경의
#            파이썬을 준다
#   hybrid   기기의 모델이 먼저 답하고 자신 없을 때만 큰 모델로
#
# 전원을 꽂고, 다른 일을 하지 않는 동안 돌린다. 재기 전마다 3초 쉰다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch47"
MODELS="${BADAK_MODELS:-$ROOT/models}"
mkdir -p "$OUT"
cd "$ROOT"
PARTS="${*:-budget system engines hybrid}"
wants() { case " $PARTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

cargo build --quiet --release -p badak_models --example ch47_budget --example ch47_hybrid
BUDGET="$ROOT/target/release/examples/ch47_budget"

# 명령을 돌리고, 그것이 적은 줄 뒤에 프로세스의 가장 컸던 풋프린트를 붙인다.
with_footprint() {
    sleep 3
    /usr/bin/time -l "$@" 2> "$OUT/.time" | tr '\n' ' '
    awk '/peak memory footprint/ { printf "풋프린트 %d\n", $1 }' "$OUT/.time"
    rm -f "$OUT/.time"
}

if wants budget; then
    {
        "$BUDGET" predict
        # 모델, 가중치의 방식, KV 캐시의 정밀도, 읽을 토큰의 수.
        while read -r model weights cache tokens; do
            [ -n "$model" ] || continue
            with_footprint "$BUDGET" run "$model" "$weights" "$cache" "$tokens"
        done <<'CONFIGS'
smollm2_135m stored f32 512
smollm2_135m stored f32 2048
smollm2_135m stored f16 2048
smollm2_135m q4_0 f32 2048
smollm2_360m stored f32 2048
llama32_1b stored f32 2048
llama32_1b q4_0 f32 2048
llama32_1b q4_0 f16 2048
llama32_1b q4_0 f16 8192
smollm2_1_7b stored f32 2048
smollm2_1_7b stored f16 8192
CONFIGS
    } > "$OUT/budget.txt"
    cat "$OUT/budget.txt"
fi

if wants system; then
    mkdir -p "$ROOT/target/system_model"
    xcrun swiftc -parse-as-library -O apps/system_model/SystemModel.swift \
        -o "$ROOT/target/system_model/system_model"
    {
        for round in 1 2 3; do
            for prompt in "What is the capital of France?" \
                "Explain in two sentences why the sky is blue." \
                "Write a haiku about the sea."; do
                with_footprint "$ROOT/target/system_model/system_model" "$prompt" | head -n 1 \
                    | sed "s/^system/system $round/"
            done
        done
    } > "$OUT/system.txt"
    cat "$OUT/system.txt"
fi

if wants engines; then
    LLAMA="$ROOT/third_party/llama.cpp/build/bin/llama-bench"
    {
        for round in 1 2 3; do
            for weights in stored q8_0 q4_0; do
                with_footprint "$BUDGET" speed smollm2_135m "$weights"
            done
            if [ -x "$LLAMA" ]; then
                for file in model_q8_0 model_q4_0; do
                    for layers in 0 99; do
                        sleep 3
                        "$LLAMA" -m "$MODELS/smollm2_135m/$file.gguf" -p 256 -n 128 -r 3 \
                            -ngl "$layers" -o jsonl 2> /dev/null \
                            | awk -v file="$file" -v layers="$layers" '
                                /"n_prompt": 256/ { match($0, /"avg_ts": [0-9.]+/); reading = substr($0, RSTART + 10, RLENGTH - 10) }
                                /"n_gen": 128/ { match($0, /"avg_ts": [0-9.]+/); writing = substr($0, RSTART + 10, RLENGTH - 10) }
                                END { printf "speed llama.cpp smollm2_135m %s gpu층 %s 읽기 %.0f 쓰기 %.0f\n", file, layers, reading, writing }'
                    done
                done
            fi
            if [ -n "${PYTHON:-}" ]; then
                BIN="$(dirname "$PYTHON")"
                for backend in cpu gpu; do
                    sleep 3
                    "$BIN/litert-lm" benchmark "$MODELS/smollm2_135m_litert/SmolLM2_135M_Instruct.litertlm" \
                        -p 256 -d 128 --backend "$backend" 2> /dev/null \
                        | awk -v backend="$backend" '
                            /Prefill speed/ { reading = $3 }
                            /Decode speed/ { writing = $3 }
                            /Init time/ { init = $3 * 1000 }
                            END { printf "speed litert-lm smollm2_135m %s 올리기 %.0f 읽기 %.0f 쓰기 %.0f\n", backend, init, reading, writing }'
                done
                sleep 3
                "$PYTHON" reference/ch47_executorch.py 2> /dev/null | grep '^speed'
            fi
        done
    } > "$OUT/engines.txt"
    cat "$OUT/engines.txt"
fi

if wants hybrid; then
    HYBRID="$ROOT/target/release/examples/ch47_hybrid"
    sleep 3
    "$HYBRID" smollm2_360m smollm2_1_7b > "$OUT/hybrid.txt"
    sleep 3
    env CH47_SYSTEM=0 "$HYBRID" smollm2_360m smollm2_1_7b > "$OUT/hybrid_no_system.txt"
    sleep 3
    "$HYBRID" smollm2_135m smollm2_1_7b > "$OUT/hybrid_135m.txt"
    grep -h '^summary' "$OUT"/hybrid*.txt
    # Llama 3.2의 틀이 적는 날짜 두 줄을 넣을 때와 뺄 때(47.3절).
    if [ -f "$MODELS/llama32_1b/model.safetensors" ]; then
        sleep 3
        cargo run --quiet --release -p badak_models --example ch47_template > "$OUT/template_llama32.txt"
        grep -h '^summary' "$OUT/template_llama32.txt"
    fi
fi
