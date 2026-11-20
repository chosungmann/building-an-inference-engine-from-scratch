"""41장: SmolLM2-135M-Instruct에 붙일 작은 LoRA 어댑터를 학습시키고 정답을 만든다.

실행:
    python reference/ch41_train_lora.py

출력: models/smollm2_135m_lora/
    adapter_config.json, adapter_model.safetensors   PEFT가 쓴 어댑터
    reference.safetensors                            어댑터를 붙인 모델의 점수와 쓴 토큰

가르치는 것은 단순하다. **답을 모두 대문자로 쓰기.** 질문 마흔 개에
원래 모델이 쓴 답을 대문자로 바꿔 정답으로 삼는다. 어댑터가 붙었는지를
눈으로 볼 수 있다는 것 말고는 쓸모가 없는 어댑터다.

CPU에서 2~3분 걸린다. 씨앗을 고정했지만, 학습은 기계와 라이브러리의
버전에 따라 끝자리가 달라질 수 있다. 그래서 어댑터를 Hub에 올려 두지
않고, 어댑터와 **그 어댑터로 계산한 정답**을 한 번에 만든다. badak은
같은 디렉터리의 어댑터를 읽어 그 정답과 견준다.
"""

from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "smollm2_135m"
OUT = ROOT / "models" / "smollm2_135m_lora"

TOPICS = [
    "the sea", "a cat", "bread", "the moon", "rain", "a bicycle", "tea", "a clock", "snow", "a bridge",
    "music", "a train", "bees", "a river", "paper", "a lighthouse", "salt", "a mountain", "glass", "a garden",
]
QUESTIONS = [f"Tell me one fact about {topic}." for topic in TOPICS] + [
    f"Describe {topic} in one sentence." for topic in TOPICS
]
TEST = "Tell me one fact about the sun."


def chat(question, answer=None):
    """8장의 ChatML 틀. badak의 `ChatFormat::ChatMl`과 같은 글을 만든다."""
    text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return text if answer is None else text + answer + "<|im_end|>\n"


def write(model, tokenizer, question, limit):
    ids = tokenizer(chat(question), return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=limit, do_sample=False, repetition_penalty=1.0)
    return ids, out[0, ids.shape[1]:]


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model.eval()
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.generation_config.eos_token_id = end
    model.generation_config.pad_token_id = end

    # 1. 원래 모델의 답을 대문자로 바꿔 정답으로 삼는다.
    pairs = []
    for question in QUESTIONS:
        _, answer = write(model, tokenizer, question, 24)
        text = tokenizer.decode(answer, skip_special_tokens=True).strip()
        pairs.append((question, text.upper()))
    print("가르칠 것의 보기:", pairs[0])

    # 2. 질의와 값의 층에만 어댑터를 붙여 학습시킨다.
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(model, config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    for epoch in range(6):
        total = 0.0
        for question, answer in pairs:
            prompt = tokenizer(chat(question), add_special_tokens=False).input_ids
            whole = tokenizer(chat(question, answer), add_special_tokens=False).input_ids
            ids = torch.tensor([whole])
            labels = ids.clone()
            labels[0, : len(prompt)] = -100  # 질문은 채점하지 않는다.
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"  {epoch + 1}번째 돌기: 손실 {total / len(pairs):.3f}")
    model.eval()

    # 3. 어댑터를 적고, 그 어댑터로 계산한 정답을 함께 적는다.
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    ids, written = write(model, tokenizer, TEST, 24)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].numpy()
    print("어댑터를 붙인 답:", repr(tokenizer.decode(written, skip_special_tokens=True)))
    save_file(
        {
            "ids": ids[0].numpy().astype(np.int32),
            "logits": logits.astype(np.float32),
            "written": written.numpy().astype(np.int32),
        },
        OUT / "reference.safetensors",
        metadata={"question": TEST},
    )
    for name in ("README.md",):
        (OUT / name).unlink(missing_ok=True)
    adapter = OUT / "adapter_model.safetensors"
    print(f"-> {OUT.relative_to(ROOT)} (어댑터 {adapter.stat().st_size}바이트)")


if __name__ == "__main__":
    main()
