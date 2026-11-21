"""43장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch43.py
출력: book/src/images/ch43/*.svg

bench/ch43/ 에 남긴 측정값의 중앙값을 그린다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch43"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch43"  # noqa: F405


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def rows():
    out = []
    for path in sorted(BENCH.glob("round_*.txt")):
        out += [row.split() for row in path.read_text(encoding="utf-8").splitlines()]
    return out


def megabytes(parts, key):
    at = parts.index(key)
    value, unit = float(parts[at + 1]), parts[at + 2]
    return {"B": 0.0, "KB": value / 1000, "MB": value, "GB": value * 1000}[unit]


def fig_memory(data):
    """가중치를 두는 방식마다: 더러운 페이지와 깨끗한 페이지."""
    body = [text(20, 28, "Llama 3.2 1B를 올렸을 때의 메모리: 운영체제가 앱의 것으로 세는 페이지와 세지 않는 페이지", INK, 14, "start", "bold")]  # noqa: F405
    body.append(rect(20, 44, 14, 10, ORANGE))  # noqa: F405
    body.append(text(40, 54, "더러운 페이지 (프로그램이 써 넣었다. 풋프린트에 든다)", INK, 12, "start"))  # noqa: F405
    body.append(rect(380, 44, 14, 10, BLUE_BG, BLUE))  # noqa: F405
    body.append(text(400, 54, "깨끗한 페이지 (파일을 비췄다. 모자라면 버릴 수 있다)", INK, 12, "start"))  # noqa: F405
    cases = [("stored", "파일 그대로 비추기"), ("f32", "f32로 펼치기"), ("q8_0", "8비트 파일 비추기"), ("q4_0", "4비트 파일 비추기")]
    found = []
    for weights, label in cases:
        inside = [p for p in data if p[:4] == ["p12", "llm", "llama32_1b", weights] and "안쪽" in p]
        if not inside:
            return
        dirty = statistics.median(megabytes(p, "더러운페이지") for p in inside)
        clean = statistics.median(megabytes(p, "깨끗한페이지") for p in inside)
        found.append((label, dirty, clean))
    scale = 620 / max(dirty + clean for _, dirty, clean in found)
    for index, (label, dirty, clean) in enumerate(found):
        y = 74 + index * 36
        body.append(text(190, y + 17, label, INK, 12, "end"))  # noqa: F405
        body.append(rect(200, y, max(dirty * scale, 2), 24, ORANGE))  # noqa: F405
        body.append(rect(200 + dirty * scale, y, clean * scale, 24, BLUE_BG, BLUE))  # noqa: F405
        body.append(text(200 + (dirty + clean) * scale + 8, y + 17, f"더러운 페이지 {dirty:,.0f}MB, 깨끗한 페이지 {clean:,.0f}MB", INK, 12, "start"))  # noqa: F405
    note = 74 + 4 * 36 + 16
    body.append(text(20, note, "같은 모델이다. 가중치를 프로그램이 만들면 더러운 페이지가 되고, 파일에 있는 그대로 쓰면 깨끗한 페이지가 된다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "memory.svg", 1100, note + 16, body)  # noqa: F405


if __name__ == "__main__":
    data = rows()
    if data:
        fig_memory(data)
