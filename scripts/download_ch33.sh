#!/bin/sh
# 33장의 모델, 소리, 정답을 models/whisper/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch33.sh
#
# 모델과 소리는 원래 배포처에서 받는다.
#   model.safetensors 등   openai/whisper-tiny (Apache-2.0)
#   jfk.wav               whisper.cpp 저장소의 samples/ (케네디 대통령의 1961년
#                         취임 연설. 미국 연방 정부의 저작물이다)
# 정답(reference.safetensors)은 이 책이 reference/ch33_make_reference.py 로
# 만들어 Hugging Face Hub에 올려 둔 것이다. 판을 고정해 받고 해시를 확인한다.
set -eu

OUT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}/whisper"
mkdir -p "$OUT"

MODEL="https://huggingface.co/openai/whisper-tiny/resolve/169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
SOUND="https://raw.githubusercontent.com/ggml-org/whisper.cpp/v1.8.7/samples/jfk.wav"
REFERENCE="https://huggingface.co/building-an-inference-engine-from-scratch/imagenet-tflite/resolve/310b423c9eb90279d8f78485d47d7b11d58f258f/reference_ch33.safetensors"

# 받아서 둘 이름, 주소, 해시.
FILES="
model.safetensors $MODEL/model.safetensors 7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395
config.json $MODEL/config.json ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98
generation_config.json $MODEL/generation_config.json a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512
vocab.json $MODEL/vocab.json 8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9
added_tokens.json $MODEL/added_tokens.json 9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1
jfk.wav $SOUND 59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e
reference.safetensors $REFERENCE ae2789abe38f10e0448a30ebace80115f2ea4680a717940c24f3ae796b891db1
"

echo "$FILES" | while read -r name url expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$url" -o "$OUT/$name.partial"
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
echo "33장의 모델이 $OUT 에 준비되었습니다."
