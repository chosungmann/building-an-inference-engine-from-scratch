"""9장과 10장: 언어 모델의 순전파를 PyTorch로 실행해 정답을 만든다.

실행:
    sh scripts/download_llama32.sh       (또는 download_smollm2.sh)
    python reference/ch09_llama_reference.py llama32_1b
    python reference/ch09_llama_reference.py smollm2_135m

출력: models/<모델 디렉터리>/reference.safetensors

짧은 글 하나를 모델에 넣고, 첫 블록 안의 부품마다의 출력과 모든 블록의
출력, 최종 점수를 저장한다. 가중치는 f32로 펼쳐서 계산한다.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "The capital of France is Paris. 추론 엔진"


def main():
    if len(sys.argv) != 2:
        sys.exit("모델 디렉터리의 이름을 주십시오. 예: llama32_1b")
    MODEL = ROOT / "models" / sys.argv[1]
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    block = model.model.layers[0]

    # 부품의 출력을 가로채 모아 둔다.
    captured = {}

    def keep(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach()[0].numpy()

        return hook

    hooks = [
        block.input_layernorm.register_forward_hook(keep("l0.norm1")),
        block.self_attn.q_proj.register_forward_hook(keep("l0.q")),
        block.self_attn.k_proj.register_forward_hook(keep("l0.k")),
        block.self_attn.v_proj.register_forward_hook(keep("l0.v")),
        block.self_attn.register_forward_hook(keep("l0.attn")),
        block.post_attention_layernorm.register_forward_hook(keep("l0.norm2")),
        block.mlp.register_forward_hook(keep("l0.mlp")),
        model.model.norm.register_forward_hook(keep("final_norm")),
    ]
    with torch.no_grad():
        result = model(ids, output_hidden_states=True, output_attentions=True)
    for hook in hooks:
        hook.remove()

    tensors = {
        "ids": ids[0].numpy().astype(np.int32),
        "embed": result.hidden_states[0][0].numpy(),
        "l0.attn_weights": result.attentions[0][0].numpy(),
        "l0.out": result.hidden_states[1][0].numpy(),
        # 임베딩과 모든 블록 각각의 출력: (블록 수 + 1, 토큰 수, 벡터 길이)
        "hidden_states": torch.stack(result.hidden_states)[:, 0].numpy(),
        "logits": result.logits[0].numpy(),
        **captured,
    }
    tensors = {name: np.ascontiguousarray(value) for name, value in tensors.items()}
    save_file(
        tensors,
        MODEL / "reference.safetensors",
        metadata={
            "prompt": PROMPT,
            "tokens": " ".join(tokenizer.convert_ids_to_tokens(ids[0])),
            "made_with": f"transformers {transformers.__version__}, torch {torch.__version__}",
        },
    )

    print(f"글: {PROMPT!r}")
    print(f"토큰 {ids.shape[1]}개: {ids[0].tolist()}")
    for name in sorted(tensors):
        print(f"  {name:<18} {str(tensors[name].dtype):<8} {tensors[name].shape}")
    next_id = int(result.logits[0, -1].argmax())
    print(f"다음 토큰 예측: {next_id} {tokenizer.decode([next_id])!r}")
    print(f"reference.safetensors: {(MODEL / 'reference.safetensors').stat().st_size}바이트")


if __name__ == "__main__":
    main()
