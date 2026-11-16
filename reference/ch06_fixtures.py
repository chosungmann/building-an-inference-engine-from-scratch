"""6장: 공식 라이브러리로 작은 모델 파일을 만든다.

우리 파서가 실제 도구가 쓴 파일을 읽을 수 있는지 확인하는 데 쓴다.
만들어진 파일은 저장소에 들어 있으므로 이 스크립트를 실행하지 않아도 된다.

실행: python reference/ch06_fixtures.py
출력: crates/badak_formats/tests/data/tiny.safetensors, tiny.gguf
"""

from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parents[1] / "crates/badak_formats/tests/data"

weight = np.arange(6, dtype=np.float32).reshape(2, 3)
bias = np.array([0.5, -1.5, 2.25], dtype=np.float32)
ids = np.array([3, 1, 4, 1], dtype=np.int32)
pixels = np.array([[0, 64], [128, 255]], dtype=np.uint8)
half = np.array([1.0, -2.0], dtype=np.float16)


def write_safetensors():
    save_file(
        {"weight": weight, "bias": bias, "ids": ids, "pixels": pixels, "half": half},
        OUT / "tiny.safetensors",
        metadata={"format": "pt", "note": "밑바닥 badak"},
    )


def write_gguf():
    writer = gguf.GGUFWriter(OUT / "tiny.gguf", arch="badak-test")
    writer.add_name("tiny")
    writer.add_uint32("test.uint32", 7)
    writer.add_int64("test.int64", -9_000_000_000)
    writer.add_float32("test.float32", 0.5)
    writer.add_bool("test.bool", True)
    writer.add_string("test.string", "밑바닥 badak")
    writer.add_array("test.int_array", [1, 2, 3])
    writer.add_array("test.string_array", ["<s>", "</s>", "안녕"])
    writer.add_tensor("weight", weight)
    writer.add_tensor("ids", ids)
    writer.add_tensor("half", half)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_safetensors()
    write_gguf()
    for name in ("tiny.safetensors", "tiny.gguf"):
        print(f"{name}: {(OUT / name).stat().st_size} bytes")
