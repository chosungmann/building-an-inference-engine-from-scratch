"""12장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch12.py
출력: book/src/images/ch12/*.svg

확률과 후보 수는 book/src/part2/output/ch12_distribution_llama32.txt에서
읽는다. 예제를 다시 실행해 출력 파일이 바뀌면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch12"  # noqa: F405
SOURCE = IMAGES.parent / "part2/output/ch12_distribution_llama32.txt"  # noqa: F405
GRID = "#e5e7eb"


def read_sections():
    """출력 파일을 {설정 이름: [(토큰, 확률)]}로 읽는다."""
    sections, current = {}, None
    for row in SOURCE.read_text(encoding="utf-8").splitlines():
        header = re.match(r"(.+): 후보 (\d+)개$", row)
        entry = re.match(r'\s+"(.*)"\s+([\d.]+)%$', row)
        if header:
            current = sections.setdefault(header[1], [])
        elif entry and current is not None:
            current.append((entry[1], float(entry[2])))
        elif not row.strip():
            current = None
    return sections


def read_nucleus():
    """자리마다 (고른 토큰, 후보 수)를 읽는다."""
    rows = []
    for row in SOURCE.read_text(encoding="utf-8").splitlines():
        match = re.match(r'\s*\d+\s+"(.*)"\s+(\d+)\s+[\d.]+%$', row)
        if match:
            rows.append((match[1], int(match[2])))
    return rows


def label(token):
    return token.replace("&", "&amp;").replace("<", "&lt;").replace(" ", "␣")


def bar(x, y, width, height, color):
    """왼쪽은 곧고 오른쪽 끝만 둥근 막대."""
    if width <= 6:
        return f"<rect x='{x}' y='{y}' width='{max(width, 1.5):.1f}' height='{height}' fill='{color}'/>"
    return (
        f"<path d='M{x},{y} h{width - 3:.1f} a3,3 0 0 1 3,3 v{height - 6} a3,3 0 0 1 -3,3 "
        f"h-{width - 3:.1f} z' fill='{color}'/>"
    )


def fig_pipeline():
    body = []
    steps = [
        ("점수", "(128256)", GRAY_BG, MUTED),
        ("반복 벌점", "나온 토큰을 깎는다", ORANGE_BG, ORANGE),
        ("온도", "분포를 좁히거나 편다", ORANGE_BG, ORANGE),
        ("top-k", "높은 k개만", ORANGE_BG, ORANGE),
        ("top-p", "합이 p가 될 때까지만", ORANGE_BG, ORANGE),
        ("뽑기", "난수 하나로 고른다", GREEN_BG, GREEN),
    ]
    w, h, gap, x, y = 124, 56, 18, 20, 44
    for i, (title, sub, fill, stroke) in enumerate(steps):
        body += box(x, y, w, h, title, sub, fill, stroke)
        if i + 1 < len(steps):
            body.append(line(x + w, y + h / 2, x + w + gap - 2, y + h / 2, INK, 1.5, arrow=True))
        x += w + gap
    first, last = 20 + w + gap, 20 + 5 * (w + gap) - gap
    body.append(line(first, 26, last, 26, MUTED, 1))
    body.append(text((first + last) / 2, 18, "손질: 뽑힐 수 있는 후보와 확률을 정한다", MUTED, 12))
    body.append(text(20 + 5 * (w + gap) + w / 2, 18, "선택", MUTED, 12))
    body.append(text(20, y + h + 30, "모델은 점수까지만 낸다. 그 뒤는 모두 엔진이 정하는 일이고, 여기에는 가중치가 없다.", MUTED, 12, "start"))
    svg(OUT / "pipeline.svg", 20 + 6 * (w + gap), y + h + 46, body)


def fig_temperature():
    body = []
    sections = read_sections()
    panels = [("온도 0.5", "온도 0.5"), ("온도 1 (손질 없음)", "손질 없음"), ("온도 1.5", "온도 1.5")]
    scale = 1.75  # 1%의 폭
    for i, (title, key) in enumerate(panels):
        x = 20 + i * 290
        body.append(text(x, 30, title, INK, 14, "start", "bold"))
        for row, (token, probability) in enumerate(sections[key]):
            y = 48 + row * 28
            body.append(text(x + 84, y + 15, label(token), INK, 12, "end"))
            body.append(bar(x + 92, y + 2, probability * scale, 18, BLUE))
            body.append(text(x + 92 + max(probability * scale, 1.5) + 6, y + 16, f"{probability:.1f}%", MUTED, 11, "start"))
    body.append(
        text(20, 48 + 8 * 28 + 22, '␣는 빈칸. "Once upon a time, there was a" 다음 토큰의 확률, 높은 순서로 여덟 개. 순서는 변하지 않고 격차만 변한다.', MUTED, 12, "start")
    )
    svg(OUT / "temperature.svg", 880, 48 + 8 * 28 + 40, body)


def fig_truncation():
    body = []
    rows = read_sections()["손질 없음"]
    x0, y0, bw, gap, scale = 70, 60, 64, 14, 5.0
    base = y0 + 190
    body.append(text(x0, 30, "top-k와 top-p는 어디서 자르는가", INK, 14, "start", "bold"))
    cumulative = 0.0
    for i, (token, probability) in enumerate(rows):
        x = x0 + i * (bw + gap)
        height = probability * scale
        cumulative += probability
        body.append(
            f"<path d='M{x},{base} v-{height - 3:.1f} a3,3 0 0 1 3,-3 h{bw - 6} a3,3 0 0 1 3,3 v{height - 3:.1f} z' fill='{BLUE}'/>"
        )
        body.append(text(x + bw / 2, base - height - 8, f"{probability:.1f}%", MUTED, 11))
        body.append(text(x + bw / 2, base + 18, label(token), INK, 12))
        body.append(text(x + bw / 2, base + 36, f"누적 {cumulative:.0f}%", MUTED, 11))
    body.append(line(x0 - 10, base, x0 + 8 * (bw + gap), base, MUTED, 1))
    # top-p 0.5: 누적이 50%를 넘는 셋째까지. top-k 5: 다섯째까지.
    for count, name, color, dy in ((3, "top-p 0.5: 누적이 50%에 이를 때까지 (3개)", ORANGE, 0), (5, "top-k 5: 개수로 (5개)", GREEN, 22)):
        x = x0 + count * (bw + gap) - gap / 2
        body.append(line(x, y0 + dy, x, base + 44, color, 1.5, dash="6 4"))
        body.append(text(x + 8, y0 + dy + 4, name, color, 12, "start", "bold"))
    body.append(text(x0, base + 66, "자른 선의 왼쪽만 남기고, 남은 것들의 확률을 합이 1이 되게 다시 나눈다.", MUTED, 12, "start"))
    svg(OUT / "truncation.svg", 760, base + 84, body)


def fig_nucleus():
    body = []
    rows = read_nucleus()
    x0, y0, w, h, lo, hi = 70, 50, 720, 200, 0, 3
    body.append(text(x0, 28, "자리마다 top-p 0.9가 남기는 후보의 수 (로그 눈금)", INK, 13, "start", "bold"))

    def ypos(count):
        return y0 + h - (math.log10(count) - lo) / (hi - lo) * h

    for exponent in range(lo, hi + 1):
        yy = y0 + h - (exponent - lo) / (hi - lo) * h
        body.append(line(x0, yy, x0 + w, yy, GRID, 1))
        body.append(text(x0 - 8, yy + 4, f"{10**exponent}개", MUTED, 11, "end"))
    y40 = ypos(40)
    body.append(line(x0, y40, x0 + w, y40, GREEN, 1.5, dash="6 4"))
    body.append(text(x0 + w, y40 - 6, "top-k 40이라면 늘 40개", GREEN, 11, "end"))
    step = w / len(rows)
    points = [(x0 + (i + 0.5) * step, ypos(count)) for i, (_, count) in enumerate(rows)]
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    body.append(f"<polyline points='{path}' fill='none' stroke='{BLUE}' stroke-width='2'/>")
    for (px, py), (token, count) in zip(points, rows):
        body.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{BLUE}' stroke='#ffffff'/>")
        if count >= 12:
            body.append(text(px + 8, py - 8, str(count), BLUE, 11, "start", "bold"))
        body.append(
            f"<text x='{px:.1f}' y='{y0 + h + 14}' fill='{MUTED}' font-size='11' text-anchor='end' "
            f"transform='rotate(-50 {px:.1f} {y0 + h + 14})'>{label(token)}</text>"
        )
    body.append(text(x0, y0 + h + 92, "가로: 이어 쓴 토큰(늘 1위를 골랐다). 모델이 확신하는 자리에서는 하나만 남고, 망설이는 자리에서는 수백 개가 남는다.", MUTED, 12, "start"))
    svg(OUT / "nucleus_size.svg", 860, y0 + h + 110, body)


def fig_pick():
    body = []
    rows = read_sections()["top-k 5"]
    x0, y0, w, h = 40, 70, 720, 44
    body.append(text(x0, 30, "난수 하나로 후보 고르기", INK, 14, "start", "bold"))
    fills = [BLUE_BG, ORANGE_BG, GREEN_BG, GRAY_BG, BLUE_BG]
    strokes = [BLUE, ORANGE, GREEN, MUTED, BLUE]
    x, cumulative = x0, 0.0
    for (token, probability), fill, stroke in zip(rows, fills, strokes):
        width = probability / 100 * w
        body.append(f"<rect x='{x:.1f}' y='{y0}' width='{width:.1f}' height='{h}' fill='{fill}' stroke='{stroke}'/>")
        if width > 60:
            body.append(text(x + width / 2, y0 + 20, label(token), INK, 12, weight="bold"))
            body.append(text(x + width / 2, y0 + 36, f"{probability:.0f}%", MUTED, 11))
        else:
            body.append(text(x + width / 2, y0 - 8, label(token), INK, 11))
        cumulative += probability
        x += width
        body.append(line(x, y0 + h, x, y0 + h + 6, MUTED, 1))
        body.append(text(x, y0 + h + 20, f"{min(cumulative, 100) / 100:.2f}", MUTED, 11))
    body.append(text(x0, y0 + h + 20, "0", MUTED, 11))
    u = 0.72
    ux = x0 + u * w
    body.append(line(ux, y0 + h + 62, ux, y0 + h + 4, INK, 2, arrow=True))
    body.append(text(ux, y0 + h + 80, f"난수 u = {u}", INK, 13, weight="bold"))
    body.append(text(ux + 60, y0 + h + 80, "→ ␣small을 고른다", MUTED, 12, "start"))
    body.append(text(x0, y0 + h + 110, "후보의 확률을 길이로 삼아 0과 1 사이에 늘어놓는다. 난수가 떨어진 칸이 뽑힌 토큰이다. 넓은 칸일수록 잘 뽑힌다.", MUTED, 12, "start"))
    svg(OUT / "pick.svg", 800, y0 + h + 128, body)


if __name__ == "__main__":
    for figure in (fig_pipeline, fig_temperature, fig_truncation, fig_nucleus, fig_pick):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
