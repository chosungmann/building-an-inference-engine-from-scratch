"""47장: ExecuTorch로 SmolLM2-135M을 돌려 속도를 적는다.

실행:
    sh scripts/download_ch47.sh
    pip install -r reference/requirements_ch47.txt
    python reference/ch47_executorch.py

executorch-community가 올려 둔 .pte 파일을 파이썬 꾸러미에 든 실행기로
돌린다. 그 파일은 XNNPACK용으로 내보낸 32비트 실수 모델이다(저장소의
Export_Recipe.ipynb). 꾸러미의 기본 설정 그대로 돌리고, 스레드의 수는
건드리지 않았다. 실행기가 스스로 잰 시간을 읽어 적는다.
"""

import json
import statistics
import time
from pathlib import Path

from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/smollm2_135m_executorch/model.pte"
TOKENIZER = ROOT / "models/smollm2_135m/tokenizer.json"
# 다른 엔진들과 같은 일: 긴 글을 읽고, 토큰 128개를 쓴다.
PROMPT = "Once upon a time, in a small village by the sea, " * 20


def main():
    started = time.perf_counter()
    runner = TextLLMRunner(str(MODEL), str(TOKENIZER))
    loaded = (time.perf_counter() - started) * 1e3
    reading, writing, counts = [], [], []
    for _ in range(3):
        stats = []
        config = GenerationConfig(echo=False, max_new_tokens=128, temperature=0.0, seq_len=512)
        runner.reset()
        runner.generate(PROMPT, config, lambda piece: None, lambda one: stats.append(json.loads(one.to_json_string())))
        reading.append(stats[-1]["prefill_token_per_sec"])
        writing.append(stats[-1]["decode_token_per_sec"])
        counts.append(stats[-1]["prompt_tokens"])
    # 실행기가 쓴 글을 줄바꿈 없이 표준 출력에 적으므로, 새 줄에서 시작한다.
    print(
        f"\nspeed executorch smollm2_135m f32 올리기 {loaded:.0f} 읽기 {statistics.median(reading):.0f} "
        f"쓰기 {statistics.median(writing):.0f} 프롬프트토큰 {counts[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
