"""14장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch14.py
출력: book/src/images/ch14/*.svg

측정값은 book/src/part3/output/ch14_*.txt에서 읽는다. 예제를 다시 실행해
출력 파일이 바뀌면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch14"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"


def fig_metrics():
    """요청 하나의 시간 축 위에 세 지표를 그린다."""
    body = []
    x0, y, h = 60, 110, 44
    body.append(text(x0, 30, "요청 하나의 시간과 세 가지 지표", INK, 14, "start", "bold"))
    prompt_w, token_w, tokens = 250, 46, 8
    body.append(f"<rect x='{x0}' y='{y}' width='{prompt_w}' height='{h}' rx='5' fill='{ORANGE_BG}' stroke='{ORANGE}'/>")
    body.append(text(x0 + prompt_w / 2, y + 27, "프롬프트 처리", INK, 13, weight="bold"))
    for i in range(tokens):
        x = x0 + prompt_w + i * token_w
        body.append(f"<rect x='{x + 3}' y='{y}' width='{token_w - 6}' height='{h}' rx='5' fill='{BLUE_BG}' stroke='{BLUE}'/>")
        body.append(text(x + token_w / 2, y + 27, f"t{i + 1}", INK, 12))
    end = x0 + prompt_w + tokens * token_w
    body.append(line(x0, y + h + 44, end + 30, y + h + 44, MUTED, 1.5, arrow=True))
    body.append(text(end + 36, y + h + 48, "시간", MUTED, 12, "start"))
    body.append(line(x0, y - 8, x0, y + h + 50, MUTED, 1, dash="4 3"))
    body.append(text(x0, y + h + 66, "요청 도착", MUTED, 12))
    # 첫 토큰까지의 시간
    first = x0 + prompt_w + token_w
    body.append(line(x0, y - 22, first, y - 22, ORANGE, 2))
    body.append(line(first, y - 28, first, y + h + 50, ORANGE, 1, dash="4 3"))
    body.append(text((x0 + first) / 2, y - 32, "첫 토큰까지의 시간 (TTFT)", ORANGE, 13, weight="bold"))
    body.append(text(first, y + h + 66, "첫 토큰", MUTED, 12))
    # 토큰 사이의 간격
    a, b = first + 2 * token_w, first + 3 * token_w
    body.append(line(a, y + h + 14, b, y + h + 14, BLUE, 2))
    body.append(line(a, y + h + 8, a, y + h + 20, BLUE, 2))
    body.append(line(b, y + h + 8, b, y + h + 20, BLUE, 2))
    body.append(text((a + b) / 2 + 4, y + h + 34, "토큰 사이의 간격 (ITL)", BLUE, 12, "start", "bold"))
    # 생성 속도
    body.append(line(first, y - 56, end, y - 56, GREEN, 2))
    body.append(text((first + end) / 2, y - 66, "생성 속도 (TPS) = 이 구간의 토큰 수 ÷ 이 구간의 시간", GREEN, 13, weight="bold"))
    body.append(line(end, y - 62, end, y + h + 50, GREEN, 1, dash="4 3"))
    body.append(text(end, y + h + 66, "마지막 토큰", MUTED, 12))
    svg(OUT / "metrics.svg", 800, y + h + 84, body)


def read_noise():
    """소음 실험의 두 줄을 {이름: [최소, 중앙값, 평균, p90, p99, 최대]}로 읽는다."""
    rows = {}
    for row in (OUTPUT / "ch14_pitfalls.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(조용할 때|다른 일이 CPU를 다툴 때)\s+([\d.\s]+)$", row)
        if match:
            rows[match[1]] = [float(value) for value in match[2].split()]
    return rows


def fig_noise():
    body = []
    rows = read_noise()
    x0, w, lo, hi = 250, 520, 1.5, 5.0
    body.append(text(20, 28, "같은 일을 2,000번 쟀을 때의 요약값 (마이크로초, 로그 눈금)", INK, 14, "start", "bold"))

    def xpos(value):
        return x0 + (math.log10(value) - lo) / (hi - lo) * w

    top = 70
    for exponent in (2, 3, 4, 5):
        xx = xpos(10**exponent)
        body.append(line(xx, top - 14, xx, top + 150, GRID, 1))
        body.append(text(xx, top + 168, f"{10**exponent:,}", MUTED, 11))
    marks = [("최소", MUTED, 0), ("중앙값", BLUE, 1), ("평균", ORANGE, 2), ("p99", GREEN, 4), ("최대", INK, 5)]
    for i, (label, values) in enumerate(rows.items()):
        y = top + 20 + i * 80
        body.append(text(x0 - 20, y + 5, label, INK, 13, "end", "bold"))
        body.append(line(xpos(values[0]), y, xpos(values[5]), y, GRID, 3))
        for j, (name, color, index) in enumerate(marks):
            xx = xpos(values[index])
            body.append(f"<circle cx='{xx:.1f}' cy='{y}' r='6' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
            if i == 1:
                dy = -14 if j % 2 == 0 else 24
                shown = f"{values[index]:,.0f}"
                body.append(text(xx, y + dy, f"{name} {shown}", color, 11))
        if i == 0:
            body.append(text(xpos(values[5]) + 14, y + 4, f"모두 {values[0]:.0f}에서 {values[5]:.0f} 사이", MUTED, 11, "start"))
    for j, (name, color, _) in enumerate(marks):
        body.append(f"<circle cx='{x0 + 8 + j * 100}' cy='{top + 198}' r='5' fill='{color}'/>")
        body.append(text(x0 + 20 + j * 100, top + 202, name, MUTED, 12, "start"))
    svg(OUT / "noise.svg", 820, top + 220, body)


def read_profile():
    """프로파일 출력의 경로 표를 [(경로, 직접 쓴 시간)]으로 읽는다."""
    rows = []
    for row in (OUTPUT / "ch14_profile_llama32.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\S+)\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+%$", row)
        if match:
            rows.append((match[1], float(match[2])))
    return rows


def fig_profile():
    body = []
    rows = dict(read_profile())
    total = sum(rows.values())
    parts = [
        ("MLP의 선형 층", rows["mlp/linear"], BLUE),
        ("어텐션의 선형 층", rows["attention/linear"], ORANGE),
        ("출력 층의 선형 층", rows["output_layer/linear"], GREEN),
    ]
    rest = total - sum(value for _, value, _ in parts)
    parts.append(("나머지 전부", rest, MUTED))
    x0, y, w, h = 30, 70, 760, 46
    body.append(text(x0, 30, "Llama 3.2 1B: 순전파의 시간이 쓰인 곳 (프롬프트 64토큰 + 16토큰)", INK, 14, "start", "bold"))
    x = x0
    for label, value, color in parts:
        width = value / total * w
        body.append(f"<rect x='{x:.1f}' y='{y}' width='{max(width - 2, 1):.1f}' height='{h}' rx='4' fill='{color}'/>")
        if width > 90:
            body.append(text(x + width / 2, y + 21, label, "#ffffff", 13, weight="bold"))
            body.append(text(x + width / 2, y + 38, f"{value / total * 100:.1f}%", "#ffffff", 12))
        x += width
    ly = y + h + 30
    for i, (label, value, color) in enumerate(parts):
        lx = x0 + i * 195
        body.append(f"<rect x='{lx}' y='{ly - 11}' width='14' height='14' rx='3' fill='{color}'/>")
        body.append(text(lx + 22, ly, f"{label} {value / total * 100:.1f}%", MUTED, 12, "start"))
    svg(OUT / "profile.svg", 820, ly + 22, body)


def fig_profilers():
    """계측과 표본: 같은 실행을 보는 두 가지 방법."""
    body = []
    x0, w = 150, 600
    body.append(text(20, 30, "같은 실행을 보는 두 가지 방법", INK, 14, "start", "bold"))
    spans = [("linear", 0.00, 0.44, BLUE_BG, BLUE), ("map", 0.44, 0.50, GREEN_BG, GREEN), ("linear", 0.50, 0.94, BLUE_BG, BLUE), ("zip", 0.94, 1.0, ORANGE_BG, ORANGE)]

    def lane(y):
        out = []
        for name, a, b, fill, stroke in spans:
            out.append(f"<rect x='{x0 + a * w:.1f}' y='{y}' width='{(b - a) * w - 2:.1f}' height='34' rx='4' fill='{fill}' stroke='{stroke}'/>")
            if b - a > 0.1:
                out.append(text(x0 + (a + b) / 2 * w, y + 22, name, INK, 12))
        return out

    # 계측
    y = 70
    body.append(text(x0 - 16, y + 14, "계측", INK, 13, "end", "bold"))
    body.append(text(x0 - 16, y + 32, "엔진이 직접 적는다", MUTED, 11, "end"))
    body += lane(y)
    for _, a, b, _, stroke in spans:
        for edge in (a, b):
            body.append(line(x0 + edge * w - (1 if edge == b else 0), y - 10, x0 + edge * w - (1 if edge == b else 0), y, stroke, 1.5))
    body.append(text(x0 + w + 14, y + 22, "구간의 시작과 끝에서 시각을 읽는다", MUTED, 11, "start"))
    # 표본
    y = 170
    body.append(text(x0 - 16, y + 14, "표본", INK, 13, "end", "bold"))
    body.append(text(x0 - 16, y + 32, "바깥에서 들여다본다", MUTED, 11, "end"))
    body += lane(y)
    for i in range(1, 25):
        xx = x0 + i * w / 25
        body.append(line(xx, y - 14, xx, y - 2, INK, 1.2, arrow=True))
    body.append(text(x0 + w + 14, y + 22, "일정한 간격으로 \"지금 어디인가\"를 본다", MUTED, 11, "start"))
    body.append(text(x0, y + 62, "짧은 구간(map, zip)은 표본에 거의 걸리지 않는다. 계측은 모든 구간을 세지만 구간을 미리 정해 두어야 한다.", MUTED, 12, "start"))
    svg(OUT / "profilers.svg", 1010, y + 80, body)


if __name__ == "__main__":
    for figure in (fig_metrics, fig_noise, fig_profile, fig_profilers):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
