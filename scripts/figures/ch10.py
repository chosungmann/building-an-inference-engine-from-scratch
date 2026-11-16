"""10장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch10.py
출력: book/src/images/ch10/*.svg

오차와 확률 그림의 숫자는 book/src/part2/output/ch10_*.txt에서 읽는다.
예제를 다시 실행해 출력 파일이 바뀌면 그림도 다시 만들어야 한다.
"""

import math
import re

from common import *  # noqa: F403

OUT = IMAGES / "ch10"  # noqa: F405
OUTPUT = IMAGES.parent / "part2/output"  # noqa: F405
GRID = "#e5e7eb"


def read_verify(name):
    """대조표를 (단계 이름, 숫자 여섯 개)의 목록으로 읽는다."""
    rows = []
    for row in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if len(fields) >= 8 and fields[-1] in ("통과", "실패"):
            rows.append((" ".join(fields[:-7]), [float(value) for value in fields[-7:-1]]))
    return rows


def read_candidates(name, count):
    """다음 토큰의 후보를 (토큰, 확률)의 목록으로 읽는다."""
    found = []
    for row in (OUTPUT / name).read_text(encoding="utf-8").splitlines():
        match = re.match(r'\s*\d+\s+\d+\s+"(.*)"\s+([\d.]+)%', row)
        if match:
            found.append((match.group(1), float(match.group(2))))
    return found[:count]


def fig_pipeline():
    body = []
    x0, w, h, gap = 150, 330, 46, 22
    steps = [
        ("글", '"The capital of France is"', None, GRAY_BG, MUTED),
        ("토크나이저", "8장, 10.3절", "(6)  토큰 번호", ORANGE_BG, ORANGE),
        ("임베딩", "표에서 행을 고른다", "(6, 2048)", BLUE_BG, BLUE),
        ("블록 × 16", "어텐션과 MLP. 형상은 그대로", "(6, 2048)", BLUE_BG, BLUE),
        ("마지막 줄만 고른다", "다음 토큰에는 이 줄만 필요하다", "(1, 2048)", GRAY_BG, MUTED),
        ("마지막 정규화", "RMS 정규화", "(1, 2048)", BLUE_BG, BLUE),
        ("출력 층", "임베딩 표의 모든 행과 내적", "(128256)  점수", BLUE_BG, BLUE),
        ("소프트맥스", "점수를 확률로", "(128256)  확률", GREEN_BG, GREEN),
    ]
    y = 24
    for i, (title, sub, shape, fill, stroke) in enumerate(steps):
        body += box(x0, y, w, h, title, sub, fill, stroke)
        if shape:
            body.append(text(x0 + w + 18, y + h + gap / 2 + 4, shape, MUTED, 12, "start"))
        if i + 1 < len(steps):
            body.append(line(x0 + w / 2, y + h, x0 + w / 2, y + h + gap - 2, INK, 1.5, arrow=True))
        y += h + gap
    body += box(x0, y, w, h, '" Paris"  70%', "확률이 가장 높은 토큰", GREEN_BG, GREEN)
    body.append(line(x0 + w / 2, y - gap, x0 + w / 2, y - 2, INK, 1.5, arrow=True))
    # 이 장에서 만든 부분을 묶는다.
    top, bottom = 24 + 2 * (h + gap) - 8, 24 + 7 * (h + gap) - gap + 8
    body.append(
        f"<rect x='{x0 - 14}' y='{top}' width='{w + 28}' height='{bottom - top}' rx='10' fill='none' "
        f"stroke='{MUTED}' stroke-dasharray='6 4'/>"
    )
    body.append(text(x0 - 26, (top + bottom) / 2 - 8, "LlamaModel", INK, 13, "end", "bold"))
    body.append(text(x0 - 26, (top + bottom) / 2 + 10, "(10.1, 10.2절)", MUTED, 12, "end"))
    svg(OUT / "pipeline.svg", 700, y + h + 24, body)


def fig_tied():
    body = []
    tx, ty, rows, rh, tw = 300, 70, 6, 30, 220
    body.append(text(tx + tw / 2, 30, "임베딩 표  (128256, 2048)", INK, 14, weight="bold"))
    body.append(text(tx + tw / 2, 50, "행 하나가 토큰 하나의 벡터", MUTED, 12))
    labels = ["0", "1", "…", "12366", "…", "128255"]
    for i, label in enumerate(labels):
        picked = label == "12366"
        body.append(
            f"<rect x='{tx}' y='{ty + i * rh}' width='{tw}' height='{rh}' "
            f"fill='{BLUE_BG if picked else GRAY_BG}' stroke='{BLUE if picked else MUTED}'/>"
        )
        caption = label if label == "…" else f"{label}번 토큰의 벡터"
        body.append(text(tx + tw / 2, ty + i * rh + 20, caption, INK, 12))
    mid = ty + 3 * rh + rh / 2
    # 왼쪽: 입력으로 쓸 때
    body.append(text(130, 96, "입력으로 쓸 때", INK, 14, weight="bold"))
    body.append(text(130, 116, "번호로 행을 고른다", MUTED, 12))
    body += box(60, mid - 22, 140, 44, "토큰 12366", None, ORANGE_BG, ORANGE)
    body.append(line(200, mid, tx - 4, mid, ORANGE, 1.5, arrow=True))
    body.append(text(130, mid + 48, "곱셈 0번", MUTED, 12))
    # 오른쪽: 출력으로 쓸 때
    ox = tx + tw + 100
    body.append(text(ox + 80, 96, "출력으로 쓸 때", INK, 14, weight="bold"))
    body.append(text(ox + 80, 116, "모든 행과 내적한다", MUTED, 12))
    body += box(ox, mid - 22, 160, 44, "블록을 지난 벡터", None, GREEN_BG, GREEN)
    for i in range(rows):
        body.append(line(ox - 2, mid, tx + tw + 4, ty + i * rh + rh / 2, GREEN, 1.2, arrow=True))
    body.append(text(ox + 80, mid + 48, "곱셈 128256 × 2048번", MUTED, 12))
    body.append(text(ox + 80, mid + 66, "→ 토큰마다 점수 하나", MUTED, 12))
    svg(OUT / "tied_embedding.svg", 800, ty + rows * rh + 30, body)


def log_chart(body, x, y, w, h, lo, hi, title, steps, y_label):
    """로그 눈금의 틀을 그리고, (단계, 값)을 좌표로 바꾸는 함수를 돌려준다."""
    body.append(text(x, y - 16, title, INK, 13, "start", "bold"))
    for exponent in range(lo, hi + 1):
        yy = y + h - (exponent - lo) / (hi - lo) * h
        body.append(line(x, yy, x + w, yy, GRID, 1))
        body.append(text(x - 8, yy + 4, f"1e{exponent}" if exponent else "1", MUTED, 11, "end"))
    body.append(line(x, y + h, x + w, y + h, MUTED, 1))
    for step in range(0, steps + 1, 5 if steps > 20 else 4):
        xx = x + step / steps * w
        body.append(line(xx, y + h, xx, y + h + 5, MUTED, 1))
        body.append(text(xx, y + h + 19, str(step), MUTED, 11))
    body.append(text(x + w / 2, y + h + 38, "지나온 블록의 수", MUTED, 12))
    body.append(text(x - 8, y - 16, y_label, MUTED, 11, "end"))

    def point(step, value):
        value = min(max(math.log10(value), lo), hi)
        return x + step / steps * w, y + h - (value - lo) / (hi - lo) * h

    return point


def polyline(points, color):
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    marks = "".join(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' fill='{color}' stroke='#ffffff'/>" for px, py in points)
    return f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>" + marks


def fig_error():
    body = []
    panels = [("Llama 3.2 1B", "ch10_verify_llama32.txt", 40), ("SmolLM2 135M", "ch10_verify_smollm2.txt", 450)]
    for title, name, x in panels:
        # 임베딩(오차 0)과 맨 끝의 점수는 뺀다. 마지막 블록은 정규화까지 거친 값이다.
        rows = read_verify(name)[1:-1]
        steps = len(rows)
        point = log_chart(body, x + 40, 60, 320, 220, -8, -3, title, steps, "상대 오차")
        for exponent, label, color in ((-4, "이 장의 허용 오차 1e-4", ORANGE), (-5, "연산 하나의 허용 오차 1e-5", MUTED)):
            _, yy = point(0, 10**exponent)
            body.append(line(x + 40, yy, x + 360, yy, color, 1.5, dash="6 4"))
            body.append(text(x + 356, yy - 6, label, color, 11, "end"))
        body.append(polyline([point(i + 1, values[5]) for i, (_, values) in enumerate(rows)], BLUE))
    body.append(f"<circle cx='52' cy='20' r='4' fill='{BLUE}'/>")
    body.append(text(62, 24, "첫 줄을 뺀 나머지의 상대 오차 (최대 오차 ÷ 최대 크기)", INK, 12, "start"))
    svg(OUT / "error_growth.svg", 860, 340, body)


def fig_first_token():
    body = []
    rows = read_verify("ch10_verify_llama32.txt")[:-2]
    steps = len(rows) - 1
    point = log_chart(body, 80, 60, 520, 220, -2, 3, "Llama 3.2 1B: 단계마다 가장 큰 값의 크기", steps, "크기")
    whole = [point(i, values[0]) for i, (_, values) in enumerate(rows)]
    rest = [point(i, values[3]) for i, (_, values) in enumerate(rows)]
    body.append(polyline(whole, ORANGE))
    body.append(polyline(rest, BLUE))
    body.append(text(whole[-1][0] + 10, whole[-1][1] + 4, "전체 (첫 토큰 포함)", ORANGE, 12, "start", "bold"))
    body.append(text(rest[-1][0] + 10, rest[-1][1] + 4, "첫 토큰을 뺀 나머지", BLUE, 12, "start", "bold"))
    svg(OUT / "first_token.svg", 780, 340, body)


def fig_next_token():
    body = []
    panels = [
        ("Llama 3.2 1B", "ch10_next_token_llama32.txt", 30),
        ("SmolLM2 135M", "ch10_next_token_smollm2.txt", 440),
    ]
    scale = 2.6  # 1%가 차지하는 폭
    for title, name, x in panels:
        body.append(text(x, 30, title, INK, 14, "start", "bold"))
        for i, (token, probability) in enumerate(read_candidates(name, 8)):
            y = 50 + i * 30
            label = token.replace("&", "&amp;").replace("<", "&lt;").replace(" ", "␣")
            body.append(text(x + 96, y + 16, label, INK, 13, "end"))
            width = max(probability * scale, 1.5)
            body.append(
                f"<path d='M{x + 106},{y + 2} h{width - 3:.1f} a3,3 0 0 1 3,3 v14 a3,3 0 0 1 -3,3 h-{width - 3:.1f} z' fill='{BLUE}'/>"
                if width > 6
                else f"<rect x='{x + 106}' y='{y + 2}' width='{width:.1f}' height='20' fill='{BLUE}'/>"
            )
            body.append(text(x + 106 + width + 8, y + 17, f"{probability:.1f}%", MUTED, 12, "start"))
    body.append(text(30, 50 + 8 * 30 + 22, '␣는 빈칸. "The capital of France is" 다음에 올 토큰의 확률, 높은 순서로 여덟 개.', MUTED, 12, "start"))
    svg(OUT / "next_token.svg", 860, 50 + 8 * 30 + 40, body)


if __name__ == "__main__":
    for figure in (fig_pipeline, fig_tied, fig_error, fig_first_token, fig_next_token):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
