#!/bin/sh
# 2부에서 쓰는 언어 모델 SmolLM2-135M-Instruct를 내려받아
# models/smollm2_135m/ 에 둔다.
#
# 사용법: sh scripts/download_smollm2.sh
#
# 모델을 만든 Hugging Face TB 팀의 저장소에서 버전(revision)을 고정해 받고
# SHA-256 해시를 확인한다. 라이선스는 Apache 2.0이며 동의 절차가 없다.
# 가중치 파일은 약 269MB다.
set -eu

REPO="HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
BASE="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/smollm2_135m"
mkdir -p "$OUT"

# 이 책이 PyTorch로 계산해 올려 둔 정답 파일. 9장부터 13장까지의 검증에 쓴다.
REFERENCE_REPO="building-an-inference-engine-from-scratch/smollm2-135m-reference"
REFERENCE_REVISION="1a490305d063c639f46cdb8ecdd6ba0fcd113c95"
REFERENCE_BASE="https://huggingface.co/$REFERENCE_REPO/resolve/$REFERENCE_REVISION"

# 받을 곳, 파일 이름, SHA-256.
FILES="
$BASE config.json 8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a
$BASE tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
$BASE tokenizer_config.json 4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
$BASE model.safetensors 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c
$REFERENCE_BASE reference.safetensors f283a989fd9fb0f33814427f649220c0199152ff9d3327a2bdcb4e7eea079d29
$REFERENCE_BASE tokenizer_vectors.json 5850ca876a4b09305e1e11d7e7375dfc629fe27363a9d30eda5070bbca6fb759
$REFERENCE_BASE generation.json 474634205aa2bb54866f18576fbf42a4cbbd14a1f7212a6a1ef6949b476ce322
$REFERENCE_BASE sampling.safetensors dbac90d38b6cbf8ffc9c72e466f79d45a157d49201fd893b033152d604b2d91e
$REFERENCE_BASE logits.safetensors 5981e5ab0bcb2728419a4d1d201248e974467d5bd8c94f1830abdc00294974e0
"

echo "$FILES" | while read -r base name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fSL --progress-bar "$base/$name" -o "$OUT/$name.partial"
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
