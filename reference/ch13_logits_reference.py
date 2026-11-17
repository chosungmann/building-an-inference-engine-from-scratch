"""13장: 엔진들의 점수를 견줄 기준을 f64로 계산한다.

실행:
    python reference/ch13_logits_reference.py llama32_1b
    python reference/ch13_logits_reference.py smollm2_135m

출력: models/<모델 디렉터리>/logits.safetensors

"The capital of France is"의 마지막 토큰 다음 점수를 두 가지로 계산한다.
가중치를 f64로 바꿔 계산한 것이 기준이고, 평소처럼 f32로 계산한 것은 견줄
대상 가운데 하나다(12.6절: 기준 구현도 f32에서는 오차가 있다).

safetensors의 f64를 badak이 읽지 못하므로 기준은 f32 둘로 나누어 적는다.
`truth_hi`는 f32로 반올림한 값이고 `truth_lo`는 그때 잘려 나간 나머지다.
둘을 f64로 더하면 기준이 된다.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "The capital of France is"


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    directory = ROOT / "models" / sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        single = model(ids).logits[0, -1].numpy()
        double = model.double()(ids).logits[0, -1].numpy()

    hi = double.astype(np.float32)
    lo = (double - hi.astype(np.float64)).astype(np.float32)
    out = directory / "logits.safetensors"
    save_file(
        {"ids": ids[0].numpy().astype(np.int32), "truth_hi": hi, "truth_lo": lo, "pytorch_f32": single},
        out,
        metadata={"prompt": PROMPT},
    )
    scale = np.abs(double).max()
    error = np.abs(single.astype(np.float64) - double).max()
    print(f"{sys.argv[1]}: 토큰 {ids.shape[1]}개, 가장 큰 점수 {scale:.3f}")
    print(f"  PyTorch f32와 f64의 가장 큰 차이 {error:.3e} (상대 {error / scale:.3e})")
    print(f"-> {out.name} ({out.stat().st_size}바이트)")


if __name__ == "__main__":
    main()
