"""16장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch16.py
출력: book/src/images/ch16/*.svg

측정값은 book/src/part3/output/ch16_linear.txt와 bench/의 기록에서 읽는다.
다시 측정하면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch16"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"


def fig_chain():
    """합 하나에 차례로 더하기와, 합 네 개에 나누어 더하기."""
    body = []
    body.append(text(20, 30, "덧셈의 줄 서기", INK, 14, "start", "bold"))

    def node(x, y, label, fill, stroke):
        return [
            f"<rect x='{x}' y='{y}' width='44' height='30' rx='5' fill='{fill}' stroke='{stroke}'/>",
            text(x + 22, y + 20, label, INK, 12),
        ]

    # 위: 합 하나
    y = 70
    body.append(text(20, y + 20, "합 하나", INK, 13, "start", "bold"))
    for i in range(8):
        x = 130 + i * 78
        body += node(x, y, f"+p{i}", ORANGE_BG, ORANGE)
        if i < 7:
            body.append(line(x + 44, y + 15, x + 76, y + 15, ORANGE, 1.5, arrow=True))
    body.append(text(130, y + 54, "덧셈 하나가 끝나야 다음 덧셈을 시작한다. 여덟 번이 한 줄로 선다.", MUTED, 12, "start"))

    # 아래: 합 넷
    y0 = 170
    body.append(text(20, y0 + 20, "합 넷", INK, 13, "start", "bold"))
    for lane in range(4):
        y = y0 + lane * 38
        for i in range(2):
            x = 130 + i * 78
            body += node(x, y, f"+p{lane + i * 4}", BLUE_BG, BLUE)
        body.append(line(174, y + 15, 206, y + 15, BLUE, 1.5, arrow=True))
        body.append(line(252, y + 15, 296, y0 + 72, MUTED, 1))
    body += node(300, y0 + 57, "모음", GREEN_BG, GREEN)
    body.append(text(370, y0 + 62, "줄이 넷이다. 네 줄은 서로 기다리지 않는다.", MUTED, 12, "start"))
    body.append(text(370, y0 + 82, "같은 시간에 덧셈을 네 배로 한다.", MUTED, 12, "start"))
    body.append(text(20, y0 + 176, "p0, p1, …은 곱셈의 결과다. 곱셈은 처음부터 서로 독립이다. 줄을 세우는 것은 덧셈이다.", MUTED, 12, "start"))
    svg(OUT / "chain.svg", 800, y0 + 194, body)


def fig_rows():
    """벡터 하나를 가중치 네 행과 한꺼번에."""
    body = []
    body.append(text(20, 30, "벡터 하나와 가중치 네 행을 한꺼번에", INK, 14, "start", "bold"))
    x0, y0, w = 150, 70, 480
    body.append(f"<rect x='{x0}' y='{y0}' width='{w}' height='30' rx='4' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
    body.append(text(x0 - 14, y0 + 20, "입력 벡터", INK, 12, "end", "bold"))
    for r in range(4):
        y = y0 + 64 + r * 36
        body.append(f"<rect x='{x0}' y='{y}' width='{w}' height='28' rx='4' fill='{BLUE_BG}' stroke='{BLUE}'/>")
        body.append(text(x0 - 14, y + 19, f"가중치 행 {r}", INK, 12, "end"))
        body.append(text(x0 + w + 16, y + 19, f"→ 내적 {r}", MUTED, 12, "start"))
    # 지금 읽고 있는 조각
    cx = x0 + 150
    body.append(f"<rect x='{cx}' y='{y0 - 6}' width='60' height='{64 + 4 * 36 + 8}' rx='6' fill='none' stroke='{INK}' stroke-width='2' stroke-dasharray='6 4'/>")
    body.append(text(cx + 30, y0 + 64 + 4 * 36 + 22, "값 16개", INK, 12, weight="bold"))
    body.append(text(20, y0 + 64 + 4 * 36 + 52, "입력 벡터의 조각은 한 번 읽어서 네 번 쓴다. 가중치 네 행은 가장 빠른 캐시에 머문 채로 다음 벡터를 기다린다.", MUTED, 12, "start"))
    svg(OUT / "rows.svg", 800, y0 + 64 + 4 * 36 + 70, body)


def read_steps():
    """{벡터 수: [(방법, 사이클)]}."""
    result, current = {}, None
    for row in (OUTPUT / "ch16_linear.txt").read_text(encoding="utf-8").splitlines():
        head = re.match(r"벡터 (\d+)개 × 가중치 \(8192", row)
        entry = re.match(r"(\d)\. (.+?)\s+[\d.]+\s+([\d.]+)\s+[\d.]+$", row)
        if head:
            current = result.setdefault(int(head[1]), [])
        elif entry and current is not None:
            current.append((entry[2].strip(), float(entry[3])))
        elif not row.strip():
            current = None
    return result


def fig_steps():
    body = []
    steps = read_steps()
    x0, w, lo, hi, row = 300, 440, -1.0, 0.5, 54
    body.append(text(20, 28, "선형 층 하나, 곱셈 하나에 드는 사이클 (로그 눈금, 왼쪽이 빠르다)", INK, 14, "start", "bold"))

    def xpos(cycles):
        return x0 + (math.log10(cycles) - lo) / (hi - lo) * w

    labels = ["1. 2장의 matvec", "2. 합을 16개로 나눈다", "3. 가중치 한 행으로 모든 벡터를", "4. 가중치 네 행을 한꺼번에"]
    top = 64
    bottom = top + len(labels) * row
    for value in (0.1, 0.2, 0.5, 1, 2):
        xx = xpos(value)
        body.append(line(xx, top - 10, xx, bottom - 10, GRID, 1))
        body.append(text(xx, bottom + 8, f"{value:g}", MUTED, 11))
    for i, label in enumerate(labels):
        y = top + i * row + 8
        body.append(text(x0 - 20, y + 5, label, INK, 12, "end"))
        one, many = steps[1][i][1], steps[256][i][1]
        body.append(line(xpos(many), y, xpos(one), y, MUTED, 2))
        for value, color, dy in ((one, BLUE, 22), (many, ORANGE, -12)):
            body.append(f"<circle cx='{xpos(value):.1f}' cy='{y}' r='6' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
            body.append(text(xpos(value), y + dy, f"{value:.2f}", color, 11))
    for i, (label, color) in enumerate((("벡터 256개 (프롬프트 처리)", ORANGE), ("벡터 1개 (토큰 생성)", BLUE))):
        body.append(f"<circle cx='{x0 + 8 + i * 220}' cy='{bottom + 36}' r='5' fill='{color}'/>")
        body.append(text(x0 + 20 + i * 220, bottom + 40, label, MUTED, 12, "start"))
    svg(OUT / "steps.svg", 800, bottom + 58, body)


def speed(path, test):
    text_ = (ROOT / path).read_text(encoding="utf-8")
    return float(re.search(rf"{test}\s+([\d.]+)", text_)[1])


def fig_result():
    body = []
    llama = (ROOT / "bench/ch13/llama_cpp_llama32.txt").read_text(encoding="utf-8").split("## ")[1]
    groups = [
        ("토큰 생성", "tg32", float(re.search(r"tg32 \|\s+([\d.]+)", llama)[1])),
        ("프롬프트 처리", "pp256", float(re.search(r"pp256 \|\s+([\d.]+)", llama)[1])),
    ]
    x0, w, top_value = 250, 480, 30.0
    body.append(text(20, 28, "Llama 3.2 1B, 스레드 1개: 1초에 처리하는 토큰", INK, 14, "start", "bold"))
    y = 60
    for title, test, theirs in groups:
        before = speed("bench/baseline/badak_llama32_ch14.txt", test)
        after = speed("bench/baseline/badak_llama32_ch16.txt", test)
        body.append(text(20, y + 16, title, INK, 13, "start", "bold"))
        for label, value, color in (("badak, 15장까지", before, MUTED), ("badak, 이 장", after, BLUE), ("llama.cpp (13.5절)", theirs, ORANGE)):
            width = value / top_value * w
            body.append(text(x0 - 12, y + 15, label, INK, 12, "end"))
            body.append(
                f"<path d='M{x0},{y + 2} h{max(width - 3, 1):.1f} a3,3 0 0 1 3,3 v12 a3,3 0 0 1 -3,3 h-{max(width - 3, 1):.1f} z' fill='{color}'/>"
            )
            body.append(text(x0 + width + 8, y + 16, f"{value:.2f}" if value < 10 else f"{value:.1f}", MUTED, 12, "start"))
            y += 28
        y += 18
    svg(OUT / "result.svg", 800, y + 4, body)


if __name__ == "__main__":
    for figure in (fig_chain, fig_rows, fig_steps, fig_result):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
