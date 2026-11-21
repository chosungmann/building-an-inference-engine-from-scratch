#!/bin/sh
# 52장의 측정: 해석기와, 그래프를 Rust 소스로 옮겨 컴파일한 것.
#
# 쓰는 법: sh scripts/bench_ch52.sh
# 결과: bench/ch52/compile.txt, bench/ch52/ir.txt
#
# 필요한 것: 길에 있는 rustc, 7장의 models/mnist_mlp/,
# sh scripts/download_mnist.sh. 전원을 연결하고, 다른 일을 하지 않는 동안
# 돌린다. 5분쯤 걸린다(소스 쉰여 개를 rustc로 컴파일한다).
#
# bench/ch52/shaders.txt 는 따로 만든다. 50장의 download_ch50.sh 가 받은
# MLC의 모델 라이브러리를 읽는다:
#   python3 reference/ch52_shaders.py > bench/ch52/shaders.txt
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/ch52"
cd "$ROOT"
mkdir -p "$OUT"
cargo build --quiet --release -p badak_graph --example ch52_compile
sleep 3
target/release/examples/ch52_compile kernels mlp glue > "$OUT/compile.txt"
target/release/examples/ch52_compile ir > "$OUT/ir.txt"
cat "$OUT/compile.txt"
