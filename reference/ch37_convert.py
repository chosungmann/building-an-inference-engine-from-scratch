"""37장: WavTokenizer의 디코더 가중치를 safetensors로 옮긴다.

실행:
    sh scripts/download_ch37.sh        (원본 체크포인트까지 받는다)
    pip install -r reference/requirements_ch37.txt
    python reference/ch37_convert.py

입력: models/outetts/original/wavtokenizer.ckpt
      (novateur/WavTokenizer-large-speech-75token, MIT)
출력: models/outetts/wavtokenizer.safetensors

원본은 PyTorch의 피클 파일이다. 피클은 읽을 때 코드를 실행할 수 있으므로
`weights_only=True`로만 연다. 이 옵션은 텐서와 기본 자료형만 풀고 그 밖의
것은 거절한다. 1.75GB 가운데 소리를 만드는 데 쓰는 것만 골라 낸다(코드북,
backbone, head). 학습에만 쓰는 판별기와 인코더는 버린다.

값은 바꾸지 않는다. 이름을 줄이고 모양만 다듬는다.
- 창이 1인 합성곱의 필터 (출력, 입력, 1)은 (출력, 입력)의 행렬로 편다.
- 조건에 따라 달라지는 층 정규화(`scale`, `shift`가 네 줄)는 이 모델이 쓰는
  첫 줄(대역폭 번호 0)만 남긴다.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/outetts/original/wavtokenizer.ckpt"
OUT = ROOT / "models/outetts/wavtokenizer.safetensors"
CODEBOOK = "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed"


def main():
    state = torch.load(SOURCE, map_location="cpu", weights_only=True)["state_dict"]
    out = {"codebook": state[CODEBOOK]}
    for name, value in state.items():
        if not name.startswith(("backbone.", "head.out.")):
            continue
        if name.endswith((".scale.weight", ".shift.weight")):
            # (대역폭 번호의 수, 폭) 가운데 첫 줄.
            name, value = name.removesuffix(".weight"), value[0]
        elif value.dim() == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        out[name] = value.to(torch.float32).contiguous()
    save_file(out, str(OUT))
    count = sum(value.numel() for value in out.values())
    print(f"{OUT}: 텐서 {len(out)}개, 값 {count:,}개")


if __name__ == "__main__":
    main()
