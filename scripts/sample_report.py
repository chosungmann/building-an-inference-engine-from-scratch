"""macOS의 sample 도구가 낸 보고서를 읽기 좋게 줄인다 (14장).

실행: python3 scripts/sample_report.py <sample이 만든 파일>

sample은 Rust의 함수 이름을 컴파일러가 바꿔 놓은 그대로(예:
`_RNvNtCs..._10badak_core3ops6linear`) 보여 준다. 이 스크립트는 그 안에서
크레이트와 모듈과 함수의 이름만 골라 `badak_core::ops::linear`처럼 적는다.
제네릭 인자는 버리는 약식 풀이다. 표본이 전체의 2%에 못 미치는 줄은 뺀다.
"""

import re
import sys


def short_name(symbol):
    """바꿔 놓은 이름에서 앞쪽의 경로만 뽑는다."""
    if not symbol.startswith("_R"):
        return symbol
    # 크레이트 이름 앞의 해시(`Cs…_`)를 지운다.
    text = re.sub(r"Cs[0-9A-Za-z]+_", "", symbol[2:])
    names, i = [], 0
    while i < len(text) and len(names) < 5:
        match = re.match(r"(\d+)", text[i:])
        if match:
            length = int(match[1])
            start = i + len(match[1])
            name = text[start : start + length]
            if len(name) == length and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                names.append(name)
                i = start + length
                continue
        i += 1
    return "::".join(names) if names else symbol


def main():
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Call graph:"))
    end = next(i for i, line in enumerate(lines) if line.startswith("Total number in stack"))
    rows = []
    for line in lines[start + 1 : end]:
        match = re.match(r"^([ +!:|]*?)(\d+) (\S+)", line)
        if match:
            rows.append((len(match[1]), int(match[2]), match[3]))
    total = rows[0][1]

    print(f"표본 {total}개. 호출 경로 (전체의 2% 이상인 줄만)")
    base = rows[0][0]
    for indent, count, symbol in rows:
        if count >= total * 0.02 and not symbol.startswith("Thread_"):
            depth = (indent - base) // 2
            print(f"{count:>6} {count / total * 100:5.1f}%  {'  ' * depth}{short_name(symbol)}")

    print()
    print("맨 안쪽 함수별 표본 수")
    start = next(i for i, line in enumerate(lines) if line.startswith("Sort by top of stack"))
    for line in lines[start + 1 :]:
        match = re.match(r"^\s+(\S+).*?(\d+)\s*$", line)
        if not match:
            break
        print(f"{int(match[2]):>6} {int(match[2]) / total * 100:5.1f}%  {short_name(match[1])}")


if __name__ == "__main__":
    main()
