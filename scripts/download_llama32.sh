#!/bin/sh
# 2부의 주 모델 Llama 3.2 1B Instruct를 내려받아 models/llama32_1b/ 에 둔다.
#
# 사용법: sh scripts/download_llama32.sh
#
# 이 모델은 내려받기 전에 Hugging Face에서 Meta의 라이선스에 동의하고 승인을
# 받아야 한다. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# 승인받은 계정의 토큰이 필요하다. 환경 변수 HF_TOKEN이 있으면 그것을,
# 없으면 `hf auth login`이 저장해 둔 토큰을 쓴다.
#
# 승인 없이 따라 하려면 scripts/download_smollm2.sh 를 대신 쓴다.
# 가중치 파일은 약 2.5GB다.
set -eu

REPO="meta-llama/Llama-3.2-1B-Instruct"
REVISION="9213176726f574b556790deb65791e0c5aa438b6"
BASE="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/llama32_1b"
mkdir -p "$OUT"

TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN=$(cat "$TOKEN_FILE")
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Hugging Face 토큰이 없습니다. HF_TOKEN을 설정하거나 hf auth login을 실행하십시오." >&2
    exit 1
fi

# 이 책이 PyTorch로 계산해 올려 둔 정답 파일. 9장부터 12장까지의 검증에 쓴다.
# 승인이 필요 없는 공개 저장소이므로 토큰을 보내지 않는다.
REFERENCE_REPO="building-an-inference-engine-from-scratch/llama32-1b-reference"
REFERENCE_REVISION="d3ea9e9e06c68a72b57f2c957dc9f9c3905a1dcb"
REFERENCE_BASE="https://huggingface.co/$REFERENCE_REPO/resolve/$REFERENCE_REVISION"

# 받을 곳, 파일 이름, SHA-256.
FILES="
$BASE config.json 2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb
$BASE tokenizer.json 79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4
$BASE tokenizer_config.json 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e
$BASE model.safetensors 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
$REFERENCE_BASE reference.safetensors 98a497873969797e8824d1226479be76636eb29319e447eafb53ff0303653acb
$REFERENCE_BASE tokenizer_vectors.json da36c2247d575417c265d6ef5689c60df25beb045331183f5348034c5a30a0bb
$REFERENCE_BASE generation.json c7c86c8ac280058c225ab020c28f16d6cd29515ba42c871616a31eeefb5a0e60
$REFERENCE_BASE sampling.safetensors 801387502d9bd4c71ab75931a02005b0737df9bd281f2d1c5ac16a1768a4eb8d
"

echo "$FILES" | while read -r base name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        if [ "$base" = "$BASE" ]; then
            curl -fSL --progress-bar -H "Authorization: Bearer $HF_TOKEN" "$base/$name" -o "$OUT/$name.partial"
        else
            curl -fSL --progress-bar "$base/$name" -o "$OUT/$name.partial"
        fi
        mv "$OUT/$name.partial" "$OUT/$name"
    fi
    actual=$(shasum -a 256 "$OUT/$name" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "해시가 다릅니다: $name" >&2
        echo "  기대: $expected" >&2
        echo "  실제: $actual" >&2
        exit 1
    fi
    echo "확인: $name"
done
echo "모델이 $OUT 에 준비되었습니다."
