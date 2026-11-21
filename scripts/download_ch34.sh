#!/bin/sh
# 34장의 모델, 글, 정답을 models/minilm/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch34.sh [onnx]
#
# 모델과 글은 원래 배포처에서 받는다.
#   model.safetensors 등   sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0)
#   stsbenchmark.tsv      STS 벤치마크. sentence-transformers 프로젝트가
#                         sbert.net 에 올려 둔, 한 파일로 합친 것이다
# `onnx`를 붙이면 같은 저장소의 onnx/model.onnx (90MB)도 받는다. ONNX
# Runtime과 견줄 때(reference/ch34_time.py)만 쓴다.
# 정답(reference.safetensors)은 이 책이 reference/ch34_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 버전을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/minilm"
mkdir -p "$OUT"

MODEL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SENTENCES="https://sbert.net/datasets/stsbenchmark.tsv.gz"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/d0cc98f26772db65e4eb7f15feab4ba2b003bd0b/reference_ch34.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
model.safetensors $MODEL/model.safetensors 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
config.json $MODEL/config.json 953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
vocab.txt $MODEL/vocab.txt 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
stsbenchmark.tsv.gz $SENTENCES aa3ae9b0060fdab8a7239df4bed7668086b4181e7cf6e921d086a70f9e42c17e
reference.safetensors $REFERENCE be57779d48323043a3a0555748728bc435d4396a97c279ea9b1402207f2207c8
"
if [ "${1:-}" = "onnx" ]; then
    FILES="$FILES
model.onnx $MODEL/onnx/model.onnx 6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
"
fi

echo "$FILES" | while read -r name url expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        if ! curl -fsSL --connect-timeout 20 "$url" -o "$OUT/$name.partial"; then
            # 글 파일을 주는 서버는 닿지 않는 곳이 있다(GitHub의 CI가
            # 그렇다). 그 파일이 없어도 토큰과 임베딩의 대조는 할 수 있으므로
            # 건너뛴다. 나머지 파일은 받지 못하면 멈춘다.
            if [ "$name" = "stsbenchmark.tsv.gz" ]; then
                echo "건너뜀: $name 을 받지 못했습니다" >&2
                rm -f "$OUT/$name.partial"
                continue
            fi
            exit 1
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
# 글은 압축을 풀어 둔다.
if [ -f "$OUT/stsbenchmark.tsv.gz" ]; then
    gunzip -c "$OUT/stsbenchmark.tsv.gz" > "$OUT/stsbenchmark.tsv"
fi
echo "34장의 모델이 $OUT 에 준비되었습니다."
