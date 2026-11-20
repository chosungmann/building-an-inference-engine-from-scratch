#!/bin/sh
# 47장에서 다른 엔진들이 쓰는 SmolLM2-135M-Instruct의 파일을 내려받는다.
#
# 사용법: sh scripts/download_ch47.sh
#
# 같은 모델을 엔진마다 자기 형식으로 바꿔 올려 둔 것이다. 엔진을 만드는
# 쪽이 운영하는 Hugging Face의 저장소에서 판(revision)을 고정해 받고
# SHA-256 해시를 확인한다. 둘 다 Apache 2.0이며 동의 절차가 없다.
#
#   litert-community/SmolLM2-135M-Instruct   LiteRT-LM의 .litertlm (143MB)
#   executorch-community/SmolLM2-135M        ExecuTorch의 .pte (543MB)
#
# badak과 llama.cpp가 쓰는 원래 모델은 scripts/download_smollm2.sh 가 받는다.
set -eu

MODELS="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"

# 저장소, 판, 둘 곳, 파일 이름, SHA-256.
FILES="
litert-community/SmolLM2-135M-Instruct f7e76881373410af03873a1d92e939495ba98f7b smollm2_135m_litert SmolLM2_135M_Instruct.litertlm ccdc5c85735743f081b7d44ca309cab569f76c0f2f0e8e163449a63721969c37
executorch-community/SmolLM2-135M dc98569fae30e15e218b94b9dcbfb1912130ec35 smollm2_135m_executorch model.pte ed9f393aadff548ab8ab12bc6afb6b66a7538bf009a132306270abd509761bb5
"

echo "$FILES" | while read -r repo revision directory name expected; do
    [ -n "$name" ] || continue
    out="$MODELS/$directory"
    mkdir -p "$out"
    if [ ! -f "$out/$name" ]; then
        echo "내려받는 중: $directory/$name"
        curl -fSL --progress-bar "https://huggingface.co/$repo/resolve/$revision/$name" \
            -o "$out/$name.partial"
        mv "$out/$name.partial" "$out/$name"
    fi
    actual=$(shasum -a 256 "$out/$name" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "해시가 다릅니다: $directory/$name" >&2
        echo "  기대: $expected" >&2
        echo "  실제: $actual" >&2
        exit 1
    fi
    echo "확인: $directory/$name"
done
echo "모델이 $MODELS 에 준비되었습니다."
