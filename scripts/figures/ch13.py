"""13장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch13.py
출력: book/src/images/ch13/*.svg

속도는 bench/ch13/의 측정 기록에서, 오차는
book/src/part2/output/ch13_compare_llama32.txt에서 읽는다. 다시 측정하면
그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch13"  # noqa: F405
ROOT = IMAGES.parents[2]  # noqa: F405
GRID = "#e5e7eb"


def read_speeds():
    """[(단계 이름, pp 토큰/초, tg 토큰/초)]. 맨 앞이 badak이다."""
    badak = (ROOT / "bench/ch13/badak_llama32.txt").read_text(encoding="utf-8")
    speed = lambda test, text: float(re.search(rf"{test}\s+([\d.]+) ±", text)[1])  # noqa: E731
    steps = [("badak", speed("pp256", badak), speed("tg32", badak))]
    labels = ["llama.cpp, 직접 짠 CPU 코드, 스레드 1개", "+ 스레드 12개", "+ Accelerate", "+ GPU", "4비트 양자화, CPU", "4비트 양자화, GPU"]
    sections = (ROOT / "bench/ch13/llama_cpp_llama32.txt").read_text(encoding="utf-8").split("## ")[1:]
    for label, section in zip(labels, sections):
        steps.append((label, speed(r"pp256 \|", section), speed(r"tg32 \|", section)))
    return steps


def fig_ladder():
    body = []
    steps = read_speeds()
    x0, w, lo, hi, row = 300, 500, 0, 4, 44
    body.append(text(20, 28, "Llama 3.2 1B: 1초에 처리하는 토큰 (로그 눈금)", INK, 14, "start", "bold"))

    def xpos(value):
        return x0 + (math.log10(value) - lo) / (hi - lo) * w

    top = 64
    bottom = top + len(steps) * row
    for exponent in range(lo, hi + 1):
        xx = x0 + (exponent - lo) / (hi - lo) * w
        body.append(line(xx, top - 10, xx, bottom - 12, GRID, 1))
        body.append(text(xx, bottom + 6, f"{10**exponent:,}", MUTED, 11))
    for i, (label, prompt, generation) in enumerate(steps):
        y = top + i * row + 8
        if i == 5:
            body.append(line(20, y - 22, x0 + w + 20, y - 22, GRID, 1, dash="4 4"))
        body.append(text(x0 - 20, y + 5, label, INK, 12, "end", "bold" if i == 0 else "normal"))
        body.append(line(xpos(generation), y, xpos(prompt), y, MUTED, 2))
        for value, color, dy in ((generation, BLUE, 22), (prompt, ORANGE, -12)):
            body.append(f"<circle cx='{xpos(value):.1f}' cy='{y}' r='6' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
            shown = f"{value:,.0f}" if value >= 100 else f"{value:.1f}"
            body.append(text(xpos(value), y + dy, shown, color, 11))
    for i, (label, color) in enumerate((("프롬프트 처리 (256토큰을 한꺼번에)", ORANGE), ("토큰 생성 (하나씩 32개)", BLUE))):
        body.append(f"<circle cx='{x0 + 8 + i * 250}' cy='{bottom + 34}' r='5' fill='{color}'/>")
        body.append(text(x0 + 20 + i * 250, bottom + 38, label, MUTED, 12, "start"))
    svg(OUT / "ladder.svg", 840, bottom + 56, body)


def read_errors():
    """[(엔진과 설정, 상대 오차)]."""
    rows = []
    text_ = (ROOT / "book/src/part2/output/ch13_compare_llama32.txt").read_text(encoding="utf-8")
    for row in text_.splitlines():
        match = re.match(r"(.+?)\s{2,}[\d.e+-]+\s+([\d.]+e-?\d+)\s+\"", row)
        if match:
            rows.append((match[1].strip(), float(match[2])))
    return rows


def fig_precision():
    body = []
    rows = read_errors()
    x0, w, lo, hi, row = 330, 440, -7, 0, 38
    body.append(text(20, 28, "Llama 3.2 1B: 점수가 기준(f64)에서 벗어난 정도 (상대 오차, 로그 눈금)", INK, 14, "start", "bold"))
    top = 60
    bottom = top + len(rows) * row
    for exponent in range(lo, hi + 1):
        xx = x0 + (exponent - lo) / (hi - lo) * w
        body.append(line(xx, top - 8, xx, bottom - 10, GRID, 1))
        body.append(text(xx, bottom + 8, "1" if exponent == 0 else f"1e{exponent}", MUTED, 11))
    limit = x0 + (-4 - lo) / (hi - lo) * w
    body.append(line(limit, top - 8, limit, bottom - 10, ORANGE, 1.5, dash="6 4"))
    body.append(text(limit + 6, top - 12, "이 책의 허용 오차 1e-4", ORANGE, 11, "start"))
    for i, (label, error) in enumerate(rows):
        y = top + i * row + 8
        xx = x0 + (math.log10(error) - lo) / (hi - lo) * w
        color = GREEN if error < 1e-4 else BLUE
        body.append(text(x0 - 20, y + 5, label, INK, 12, "end"))
        body.append(line(x0, y, xx, y, GRID, 2))
        body.append(f"<circle cx='{xx:.1f}' cy='{y}' r='6' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(xx + 12, y + 4, f"{error:.1e}", MUTED, 11, "start"))
    svg(OUT / "precision.svg", 840, bottom + 26, body)


def fig_layout():
    """질의 벡터 하나(길이 8로 줄여 그림)의 짝짓기와 행 순서."""
    body = []
    size, gap = 44, 6
    colors = [(BLUE_BG, BLUE), (ORANGE_BG, ORANGE), (GREEN_BG, GREEN), (GRAY_BG, MUTED)]
    body.append(text(20, 28, "회전의 짝짓기와 가중치의 행 순서 (헤드 하나, 길이를 8로 줄인 그림)", INK, 14, "start", "bold"))

    def row_of(x, y, order, title, sub):
        out = [text(x, y - 26, title, INK, 13, "start", "bold"), text(x, y - 9, sub, MUTED, 12, "start")]
        for i, source in enumerate(order):
            fill, stroke = colors[source % 4]
            cx = x + i * (size + gap)
            out.append(f"<rect x='{cx}' y='{y}' width='{size}' height='{size}' rx='5' fill='{fill}' stroke='{stroke}'/>")
            out.append(text(cx + size / 2, y + size / 2 + 5, f"{'ab'[source // 4]}{source % 4}", INK, 13))
        return out

    body += row_of(40, 96, list(range(8)), "원본 (safetensors)", "앞 절반 a와 뒤 절반 b. 같은 번호끼리 짝: (a0, b0), (a1, b1), …")
    body += row_of(40, 216, [0, 4, 1, 5, 2, 6, 3, 7], "변환된 파일 (GGUF)", "이웃한 둘이 짝. 변환기가 행을 번갈아 놓았다")
    # 같은 색이 한 짝이다. 짝끼리 이은 선.
    for pair in range(4):
        x1 = 40 + pair * (size + gap) + size / 2
        x2 = 40 + (pair + 4) * (size + gap) + size / 2
        height = 12 + pair * 5
        body.append(
            f"<path d='M{x1},{96 + size} v{height} h{x2 - x1} v-{height}' fill='none' stroke='{colors[pair][1]}' stroke-width='1.5'/>"
        )
    body.append(text(480, 124, "짝이 멀리 떨어져 있다", MUTED, 12, "start"))
    body.append(text(480, 244, "짝이 붙어 있다", MUTED, 12, "start"))
    body.append(text(40, 300, "행의 순서만 다르고 값은 같다. 질의와 키를 똑같이 바꾸므로 둘의 내적은 변하지 않는다.", MUTED, 12, "start"))
    svg(OUT / "rope_layout.svg", 720, 320, body)


if __name__ == "__main__":
    for figure in (fig_ladder, fig_precision, fig_layout):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
