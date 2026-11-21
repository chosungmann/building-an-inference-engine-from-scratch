"""52장: 컴파일러가 만든 것을 열어 본다.

실행:
    sh scripts/download_ch50.sh
    python3 reference/ch52_shaders.py

WebLLM이 받는 "모델의 코드"(MLC와 TVM이 모델 하나를 컴파일해 낸 .wasm 파일)
안에는 WebGPU의 셰이더가 글로 들어 있다. 그것들을 꺼내 세고, 이름과 그룹의
크기, 그리고 셰이더에 숫자로 박힌 형상을 적는다. 표준 라이브러리만 쓴다.
"""

import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = [
    "SmolLM2-135M-Instruct-q0f16_cs1k-webgpu.wasm",
    "Llama-3.2-1B-Instruct-q0f16_cs1k-webgpu.wasm",
    "Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm",
]


def shaders(data):
    """파일에서 셰이더 하나하나의 (이름, 글)을 꺼낸다."""
    found = []
    for match in re.finditer(rb"// Function: ([A-Za-z0-9_]+)\n", data):
        start = data.rfind(b"\x00", 0, match.start()) + 1
        end = data.find(b"\x00", match.end())
        found.append((match.group(1).decode(), data[start:end].decode("utf-8", "replace")))
    return found


def family(name):
    """이름에서 번호를 떼고 갈래만 남긴다."""
    name = re.sub(r"_kernel(_\d+)?$", "", name)
    return re.sub(r"\d+$", "", name)


def main():
    for library in LIBRARIES:
        path = ROOT / "models/ch50/mlc" / library
        if not path.exists():
            print(f"{library}: 없습니다 (scripts/download_ch50.sh)")
            continue
        found = shaders(path.read_bytes())
        text = sum(len(source) for _, source in found)
        print(f"library {library} 파일 {path.stat().st_size} 셰이더 {len(found)}개 글 {text}바이트")
        families = collections.Counter(family(name) for name, _ in found)
        print("families " + " ".join(f"{name}×{count}" for name, count in families.most_common(12)))
        sizes = collections.Counter(re.search(r"@workgroup_size\(([0-9, ]+)\)", source).group(1).replace(" ", "") for _, source in found)
        print("workgroups " + " ".join(f"({size})×{count}" for size, count in sizes.most_common()))
        # 행렬 곱 커널에 숫자로 박힌 큰 정수들. 모델의 형상이다.
        baked = collections.Counter()
        for name, source in found:
            if "matmul" in name:
                baked.update(int(number) for number in re.findall(r"\b(\d{3,6})i\b", source))
        print("matmul-constants " + " ".join(f"{number}×{count}" for number, count in sorted(baked.items())[:16]))
        longest = max(found, key=lambda pair: len(pair[1]))
        print(f"longest {longest[0]} {len(longest[1])}바이트 {longest[1].count(chr(10))}줄")
        # 가장 짧은 행렬 곱 커널 하나를 숫자로 적는다.
        name, source = min((pair for pair in found if "matmul" in pair[0]), key=lambda pair: len(pair[1]))
        buffers = re.findall(r"var<storage, \w+> (\w+) : array<(\w+)>", source)
        group = re.search(r"@workgroup_size\(([0-9, ]+)\)", source).group(1)
        loops = len(re.findall(r"\b(?:for|loop|while)\b", source))
        print(
            f"shortest-matmul {name} {len(source)}바이트 {source.count(chr(10))}줄 그룹 ({group})"
            f" fma {source.count('fma(')}번 workgroupBarrier {source.count('workgroupBarrier()')}번 반복문 {loops}개"
            " 버퍼 " + " ".join(f"{buffer}:{kind}" for buffer, kind in buffers)
        )


if __name__ == "__main__":
    main()
