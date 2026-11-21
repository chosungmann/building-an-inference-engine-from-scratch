#!/bin/sh
# 41장의 추측 디코딩에 쓰는 큰 모델 둘을 내려받는다.
#
# 사용법: sh scripts/download_ch41.sh
#
# SmolLM2-360M-Instruct(0.7GB)와 SmolLM2-1.7B-Instruct(3.4GB)다. 2부의
# SmolLM2-135M-Instruct와 토크나이저가 같아서, 작은 모델이 쓴 초안을 큰
# 모델이 채점할 수 있다. 모델을 만든 Hugging Face TB 팀의 저장소에서
# 버전(revision)을 고정해 받고 SHA-256 해시를 확인한다. 라이선스는 둘 다
# Apache 2.0이며 동의 절차가 없다.
#
# LoRA 어댑터는 내려받지 않는다. reference/ch41_train_lora.py 가 이
# 기계에서 학습시켜 models/smollm2_135m_lora/ 에 둔다.
set -eu

MODELS="${BADAK_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"

# 저장소, 버전, 둘 곳, 파일 이름, SHA-256.
FILES="
SmolLM2-360M-Instruct a10cc1512eabd3dde888204e902eca88bddb4951 smollm2_360m config.json 224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee
SmolLM2-360M-Instruct a10cc1512eabd3dde888204e902eca88bddb4951 smollm2_360m tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
SmolLM2-360M-Instruct a10cc1512eabd3dde888204e902eca88bddb4951 smollm2_360m tokenizer_config.json 4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
SmolLM2-360M-Instruct a10cc1512eabd3dde888204e902eca88bddb4951 smollm2_360m model.safetensors e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e
SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674 smollm2_1_7b config.json 994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea
SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674 smollm2_1_7b tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674 smollm2_1_7b tokenizer_config.json 4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674 smollm2_1_7b model.safetensors f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8
"

echo "$FILES" | while read -r repo revision directory name expected; do
    [ -n "$name" ] || continue
    out="$MODELS/$directory"
    mkdir -p "$out"
    if [ ! -f "$out/$name" ]; then
        echo "내려받는 중: $directory/$name"
        curl -fSL --progress-bar "https://huggingface.co/HuggingFaceTB/$repo/resolve/$revision/$name" \
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
