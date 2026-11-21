#!/bin/sh
# 51장의 측정: PyTorch를 그대로 쓸 때의 값과, 내보내는 길들.
#
# 쓰는 법: PYTHON=파이썬 sh scripts/bench_ch51.sh
# 결과: bench/ch51/*.txt
#
# PYTHON은 reference/requirements_ch51.txt 를 설치한 환경의 파이썬이다.
# 필요한 것: sh scripts/download_smollm2.sh, download_mnist.sh,
# download_ch50.sh(SmolLM2의 ONNX 파일), 7장의 models/mnist_mlp/.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch51"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
mkdir -p "$OUT"
cargo build --quiet --release -p badak_models --example ch11_generate
cargo build --quiet --release -p badak_graph --example ch51_onnx
cargo build --quiet --release -p badak_web --example ch49_native

# 시작부터 끝까지의 시간과 메모리. 프로그램이 스스로 적는 것이 아니라
# 밖에서 잰다. real이 걸린 시간, maximum resident set size가 가장 많이 쥔
# 메모리(바이트)다.
whole() {
    label="$1"
    shift
    sleep 3
    /usr/bin/time -l "$@" > "$OUT/whole.tmp" 2> "$OUT/whole.err" || true
    real="$(awk '/ real /{print $1}' "$OUT/whole.err")"
    resident="$(awk '/maximum resident set size/{print $1}' "$OUT/whole.err")"
    footprint="$(awk '/peak memory footprint/{print $1}' "$OUT/whole.err")"
    echo "whole $label real-s $real resident-mb $((resident / 1000000)) footprint-mb $((footprint / 1000000))"
    rm -f "$OUT/whole.tmp" "$OUT/whole.err"
}
{
    # 한 번씩 미리 돌려 파일을 운영체제의 캐시에 올린다.
    "$PYTHON" -c "import torch" > /dev/null 2>&1
    target/release/examples/ch11_generate smollm2_135m "The capital of France is" 40 > /dev/null
    whole python-만 "$PYTHON" -c "pass"
    whole python-import-torch "$PYTHON" -c "import torch"
    whole python-import-transformers "$PYTHON" -c "import torch, transformers"
    whole pytorch-글-한-줄 "$PYTHON" reference/ch51_eager.py once
    whole badak-글-한-줄 target/release/examples/ch11_generate smollm2_135m "The capital of France is" 40
} > "$OUT/whole.txt"

sleep 3
"$PYTHON" reference/ch51_eager.py size op > "$OUT/eager.txt" 2> /dev/null
sleep 3
"$PYTHON" reference/ch51_eager.py model >> "$OUT/eager.txt" 2> /dev/null
sleep 3
"$PYTHON" reference/ch51_eager.py profile 2> /dev/null | grep "^profile" >> "$OUT/eager.txt"
sleep 3
"$PYTHON" reference/ch51_eager.py compile 2> /dev/null | grep "^compile" >> "$OUT/eager.txt"
sleep 3
target/release/examples/ch49_native smollm2_135m | grep "llm-speed smollm2_135m cpu" > "$OUT/badak.txt"
sleep 3
BADAK_THREADS=1 target/release/examples/ch49_native smollm2_135m | grep "llm-speed smollm2_135m cpu" | sed 's/cpu/cpu-1-thread/' >> "$OUT/badak.txt"

"$PYTHON" reference/ch51_export.py branch export onnx 2> /dev/null | grep -E "^(branch|export|onnx) " > "$OUT/export.txt"
sleep 3
"$PYTHON" reference/ch51_export.py executorch 2> /dev/null | grep "^executorch" >> "$OUT/export.txt"
sleep 3
target/release/examples/ch51_onnx >> "$OUT/export.txt"
sleep 3
"$PYTHON" reference/ch51_export.py ort 2> /dev/null | grep "^ort" >> "$OUT/export.txt"
echo "끝났습니다: $OUT"
