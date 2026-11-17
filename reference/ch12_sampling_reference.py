"""12장: 점수를 손질한 뒤의 확률분포를 transformers로 계산해 기록한다.

실행:
    python reference/ch12_sampling_reference.py llama32_1b
    python reference/ch12_sampling_reference.py smollm2_135m

출력: models/<모델 디렉터리>/sampling.safetensors

입력은 9장의 정답 파일에 든 마지막 줄의 점수다. 모델을 다시 실행하지 않는다.
설정마다 transformers의 구현(반복 벌점, 온도, top-k, top-p의 순서)을 차례로
적용하고 소프트맥스를 한 결과를 `case0`, `case1`, …로 저장한다. 걸러진
토큰의 확률은 0이다. 설정은 메타데이터의 `cases`에 JSON으로 적는다.

계산은 f64로 한다. f32로 계산하면 값 10만여 개를 더하는 소프트맥스의 합이
틀어져 정답으로 쓸 수 없다. 얼마나 틀어지는지도 함께 출력한다(12.6절).
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.numpy import load_file, save_file
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {"temperature": 0.7},
    {"temperature": 1.5},
    {"top_k": 40},
    {"top_p": 0.9},
    {"repetition_penalty": 1.3},
    {"temperature": 0.8, "top_k": 40, "top_p": 0.95},
    {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1},
]


def distribution(case, ids, logits):
    """badak의 `Sampler::distribution`과 같은 순서로 손질한다."""
    scores = logits.clone()
    if "repetition_penalty" in case:
        scores = RepetitionPenaltyLogitsProcessor(case["repetition_penalty"])(ids, scores)
    if "temperature" in case:
        scores = TemperatureLogitsWarper(case["temperature"])(ids, scores)
    if "top_k" in case:
        scores = TopKLogitsWarper(case["top_k"])(ids, scores)
    if "top_p" in case:
        scores = TopPLogitsWarper(case["top_p"])(ids, scores)
    return scores.softmax(dim=-1)[0]


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    directory = ROOT / "models" / sys.argv[1]
    reference = load_file(directory / "reference.safetensors")
    ids = torch.from_numpy(reference["ids"]).long()[None]
    logits = torch.from_numpy(reference["logits"][-1:].copy())

    tensors = {}
    for index, case in enumerate(CASES):
        probabilities = distribution(case, ids, logits.double())
        tensors[f"case{index}"] = probabilities.float().numpy()
        kept = int((probabilities > 0).sum())
        # 같은 계산을 f32로 했을 때 확률의 합이 1에서 얼마나 벗어나는가.
        drift = distribution(case, ids, logits).double().sum().item() - 1.0
        print(
            f"case{index} {json.dumps(case)}: 남은 토큰 {kept}개, "
            f"가장 큰 확률 {probabilities.max():.4f}, f32로 계산한 확률의 합 - 1 = {drift:+.1e}"
        )

    out = directory / "sampling.safetensors"
    save_file(tensors, out, metadata={"cases": json.dumps(CASES)})
    print(f"-> {out.name} ({out.stat().st_size}바이트)")


if __name__ == "__main__":
    main()
