#!/bin/sh
# 27장의 시험 사진 다섯 장을 models/images/ 에, 모델 둘과 정답을
# models/vision_tflite/ 에 내려받는다.
#
# 사용법: sh scripts/download_ch27.sh
#
# 사진은 위키미디어 공용(Wikimedia Commons)의 원본이고, 모두 올린 사람이
# 권리를 포기한 것(CC0)이다. 저장소에는 넣지 않고 원래 위치에서 받는다.
# SHA-256 해시를 확인한다. 출처는 사진마다 아래에 적었다.
#
# 모델은 남이 공개한 가중치를 이 책이 .tflite로 바꿔 Hugging Face Hub에 올려
# 둔 것이다(ResNet-50은 BSD-3-Clause, MobileNetV2는 Apache-2.0. 출처와
# 라이선스는 그 저장소에 있다). 버전을 고정해 받고 해시를 확인한다. 직접
# 만들려면 reference/ch27_make_tflite.py 를 실행한다.
set -eu

ROOT="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"

# 받은 파일의 해시를 확인한다.
verify() {
    actual=$(shasum -a 256 "$1" | cut -d ' ' -f 1)
    if [ "$actual" != "$2" ]; then
        echo "해시가 다릅니다: $1" >&2
        echo "  기대: $2" >&2
        echo "  실제: $actual" >&2
        exit 1
    fi
    echo "확인: $(basename "$1")"
}

OUT="$ROOT/images"
BASE="https://upload.wikimedia.org/wikipedia/commons"
mkdir -p "$OUT"

# 이름, 해시, 주소. 설명 쪽은 https://commons.wikimedia.org/wiki/File:<파일 이름> 이다.
#   cup              File:Demitasse Cup and Saucer MET 270970.jpg
#   golden_retriever File:Golden Retriever Chewing A Stick.jpg
#   guitar           File:Fender California Series Acoustic Guitar (serial no. CSC10001615) (2018-04-26 13.21.46 Piqsels.com id olkmb).jpg
#   pelican          File:Brown pelican (Pelecanus occidentalis), Costa Rica.jpg
#   tabby_cat        File:Tabby Cat at a Cat Cafe (Taipei).jpg
FILES="
cup.jpg 624c3ecabc3e5ea3f6748ed0845afb3fb3116b7ddb92fe81e9b7268bc09f3fce 7/70/Demitasse_Cup_and_Saucer_MET_270970.jpg
golden_retriever.jpg de6e52f6b151fedf58992d3c715af0de0d1d8f61ba12ca2190c12d17768a3b1b f/fd/Golden_Retriever_Chewing_A_Stick.jpg
guitar.jpg d05c40ec9b467f6ecf7b7ff66a17116d90444304dee90bfa2d925378d1674c08 6/67/Fender_California_Series_Acoustic_Guitar_%28serial_no._CSC10001615%29_%282018-04-26_13.21.46_Piqsels.com_id_olkmb%29.jpg
pelican.jpg 09a951970a5ae68a4c7184be1765c5de60261bd22101aab9bbddbca66b5eaf02 a/a0/Brown_pelican_%28Pelecanus_occidentalis%29%2C_Costa_Rica.jpg
tabby_cat.jpg 83be297b693ca434188a9b8dea08787a6848b1105b157741371c71a7cfd79623 2/2b/Tabby_Cat_at_a_Cat_Cafe_%28Taipei%29.jpg
"

echo "$FILES" | while read -r name expected path; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        # 위키미디어는 누가 받는지 밝히지 않은 요청을 거절한다.
        curl -fsSL -A "badak-book-download/0.1" "$BASE/$path" -o "$OUT/$name.partial"
        mv "$OUT/$name.partial" "$OUT/$name"
    fi
    verify "$OUT/$name" "$expected"
done
echo "사진이 $OUT 에 준비되었습니다."

REPO="building-an-inference-engine-from-scratch/imagenet-tflite"
REVISION="fd225d824adbf228475b29ad9e0bd880d19f89ab"
HUB="https://huggingface.co/$REPO/resolve/$REVISION"
OUT="$ROOT/vision_tflite"
mkdir -p "$OUT"

MODELS="
resnet50.tflite 512cdd534abb0e5e6cab6da34d13b3f00683ac9e33990db9fa1292516d7bbbeb
mobilenetv2.tflite 4467c815c8f78c6a829d7e18b63c80fa817f8eaa30fe4bede369070acd25d40f
labels.txt 4eb3da435cf544e4a6f390f62c84cb9c9bb68cf8b14e97f8a063452382e5efd2
reference.safetensors a248c2baac968b01891ebdba45f9a7e0f4d3e903078befb603bf498ee5593bb5
"

echo "$MODELS" | while read -r name expected; do
    [ -n "$name" ] || continue
    if [ ! -f "$OUT/$name" ]; then
        echo "내려받는 중: $name"
        curl -fsSL "$HUB/$name" -o "$OUT/$name.partial"
        mv "$OUT/$name.partial" "$OUT/$name"
    fi
    verify "$OUT/$name" "$expected"
done
echo "모델이 $OUT 에 준비되었습니다."
