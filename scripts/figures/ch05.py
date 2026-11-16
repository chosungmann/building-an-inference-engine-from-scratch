"""5장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch05.py
출력: book/src/images/ch05/*.svg

오차 그래프는 예제의 실행 결과 파일에서 그린다.
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch05"  # noqa: F405
OUTPUT = IMAGES.parent / "part1/output"  # noqa: F405

SERIES = ["#2a78d6", "#eb6834"]
GRID = "#e5e7eb"


def fig_paths():
    body = []
    body += box(40, 30, 200, 56, "① 검사", "형상, 자료형, 출력 형상", GRAY_BG, MUTED)
    body.append(line(240, 58, 290, 58, MUTED, 2, arrow=True))
    body += box(294, 30, 200, 56, "② 읽는 길 고르기", "입력이 연속인가?", ORANGE_BG, ORANGE)

    body.append(line(394, 86, 250, 150, BLUE, 2, arrow=True))
    body.append(line(394, 86, 540, 150, MUTED, 2, arrow=True))
    body.append(text(300, 118, "예", BLUE, 13, weight="bold"))
    body.append(text(490, 118, "아니오", MUTED, 13, weight="bold"))

    body += box(130, 154, 240, 60, "빠른 길", "슬라이스를 단순한 반복문으로", BLUE_BG, BLUE)
    body += box(420, 154, 240, 60, "일반적인 길", "스트라이드를 따라가는 반복자로", GRAY_BG, MUTED)

    body.append(line(250, 214, 380, 272, BLUE, 2, arrow=True))
    body.append(line(540, 214, 410, 272, MUTED, 2, arrow=True))
    body += box(294, 276, 200, 56, "③ 계산", "결과를 새 텐서에 담는다", GREEN_BG, GREEN)
    body.append(text(395, 362, "두 길은 같은 답을 낸다. 다른 것은 속도뿐이다.", MUTED, 13))
    svg(OUT / "paths.svg", 700, 380, body)


def fig_reduce():
    body = []
    w, h = 50, 40
    x0, y0 = 250, 70
    values = ["0", "1", "2", "3", "4", "5"]
    body.append(text(x0 + 1.5 * w, y0 - 16, "x: 형상 (2, 3)", INK, 14, weight="bold"))
    body += grid(x0, y0, 2, 3, w, h, values, [GRAY_BG] * 6)
    body.append(text(x0 - 14, y0 + h + 5, "축 0 ↓", MUTED, 12, "end"))
    body.append(text(x0 + 1.5 * w, y0 + 2 * h + 0, "", MUTED, 12))

    # 축 0을 따라: 세로로 접는다
    body.append(line(x0 + 1.5 * w, y0 + 2 * h + 8, x0 + 1.5 * w, y0 + 2 * h + 44, BLUE, 2, arrow=True))
    body += grid(x0, y0 + 2 * h + 50, 1, 3, w, h, ["3", "5", "7"], [BLUE_BG] * 3)
    body.append(text(x0 + 1.5 * w, y0 + 3 * h + 72, "sum(x, 0): 형상 (3)", BLUE, 13, weight="bold"))
    body.append(text(x0 + 1.5 * w, y0 + 3 * h + 92, "세로로 접는다. 열마다 값 하나", MUTED, 12))

    # 축 1을 따라: 가로로 접는다
    body.append(line(x0 + 3 * w + 8, y0 + h, x0 + 3 * w + 60, y0 + h, ORANGE, 2, arrow=True))
    body += grid(x0 + 3 * w + 66, y0, 2, 1, w, h, ["3", "12"], [ORANGE_BG] * 2)
    body.append(text(x0 + 3 * w + 130, y0 + 16, "sum(x, 1): 형상 (2)", ORANGE, 13, "start", "bold"))
    body.append(text(x0 + 3 * w + 130, y0 + 36, "keepdim이면 형상 (2, 1)", MUTED, 12, "start"))
    body.append(text(x0 + 3 * w + 130, y0 + 56, "가로로 접는다. 행마다 값 하나", MUTED, 12, "start"))
    body.append(text(x0 + 1.5 * w, y0 - 40, "축 1 →", MUTED, 12))
    svg(OUT / "reduce.svg", 700, 290, body)


def fig_pairwise():
    body = []
    w, h, gap = 44, 30, 8

    def cells(x, y, labels, fill):
        out = []
        for i, label in enumerate(labels):
            out.append(cell(x + i * (w + gap), y, w, h, label, fill, MUTED, 13))
        return out

    names = [f"x{i}" for i in "₀₁₂₃₄₅₆₇"]

    # 왼쪽: 단순 합산은 긴 사슬
    lx = 30
    body.append(text(lx + 110, 28, "단순 합산", INK, 14, weight="bold"))
    body.append(text(lx + 110, 48, "합이 점점 커져, 뒤의 수는 큰 합에 더해진다", MUTED, 12))
    y = 70
    for i in range(8):
        body.append(cell(lx, y + i * 34, w, h, names[i], GRAY_BG, MUTED, 13))
        width = 30 + i * 24
        body.append(
            f"<rect x='{lx + 70}' y='{y + i * 34 + 5}' width='{width}' height='20' rx='4' "
            f"fill='{ORANGE_BG}' stroke='{ORANGE}'/>"
        )
        if i:
            body.append(line(lx + w + 2, y + i * 34 + 15, lx + 66, y + i * 34 + 15, MUTED, 1.5, arrow=True))
    body.append(text(lx + 70, y + 8 * 34 + 12, "지금까지의 합 (막대 길이 = 크기)", ORANGE, 12, "start"))

    # 오른쪽: 쌍별 합산은 나무
    rx = 350
    body.append(text(rx + 204, 28, "쌍별 합산", INK, 14, weight="bold"))
    body.append(text(rx + 204, 48, "크기가 비슷한 합끼리 더한다. 깊이는 log₂ n", MUTED, 12))
    levels = [(8, BLUE_BG, 70), (4, BLUE_BG, 150), (2, BLUE_BG, 230), (1, GREEN_BG, 310)]
    centers = []
    for count, fill, ly in levels:
        span = 8 * (w + gap) - gap
        step = span / count
        row = [rx + step * (i + 0.5) for i in range(count)]
        centers.append((row, ly))
        for i, cx in enumerate(row):
            bw = w if count == 8 else min(step - 10, w + (8 // count - 1) * 22)
            label = names[i] if count == 8 else ("합" if count > 1 else "전체 합")
            body.append(
                f"<rect x='{cx - bw / 2}' y='{ly}' width='{bw}' height='{h}' "
                f"fill='{GRAY_BG if count == 8 else fill}' stroke='{MUTED if count == 8 else BLUE}'/>"
            )
            body.append(text(cx, ly + h / 2 + 5, label, INK, 13))
    for (upper, uy), (lower, ly) in zip(centers, centers[1:]):
        for i, cx in enumerate(lower):
            for child in (upper[2 * i], upper[2 * i + 1]):
                body.append(line(child, uy + h + 2, cx, ly - 3, MUTED, 1.5, arrow=True))
    svg(OUT / "pairwise.svg", 780, 370, body)


def fig_sum_error():
    rows = []
    for row in (OUTPUT / "ch05_sum_error.txt").read_text(encoding="utf-8").splitlines()[1:]:
        n, _, naive, pairwise = row.split()
        rows.append((float(n), float(naive), float(pairwise)))

    left, top, width, height = 80, 40, 470, 300
    x_min, x_max = 2.0, 8.0
    y_min, y_max = -9.0, 0.0

    def px(n):
        return left + (math.log10(n) - x_min) / (x_max - x_min) * width

    def py(error):
        return top + (y_max - math.log10(error)) / (y_max - y_min) * height

    body = []
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    for exponent in range(2, 9):
        x = left + (exponent - x_min) / (x_max - x_min) * width
        body.append(line(x, top, x, top + height, GRID, 1))
        body.append(text(x, top + height + 20, "10" + str(exponent).translate(superscripts), MUTED, 12))
    for exponent in range(-9, 1):
        y = top + (y_max - exponent) / (y_max - y_min) * height
        body.append(line(left, y, left + width, y, GRID, 1))
        if exponent % 2 == 0 or exponent == -9:
            label = "1" if exponent == 0 else "10" + str(exponent).translate(superscripts)
            body.append(text(left - 10, y + 4, label, MUTED, 12, "end"))
    body.append(line(left, top + height, left + width, top + height, MUTED, 1))
    body.append(line(left, top, left, top + height, MUTED, 1))
    body.append(text(left + width / 2, top + height + 44, "더한 숫자의 개수 n", MUTED, 13))
    body.append(text(left - 52, top - 14, "상대 오차", MUTED, 13, "start"))

    series = [("단순 합산", 1, SERIES[0]), ("쌍별 합산", 2, SERIES[1])]
    for name, column, color in series:
        points = [(px(row[0]), py(row[column])) for row in rows]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        body.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in points:
            body.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        # 선 끝에 이름을 직접 적는다. 글자는 본문 색이다.
        end_x, end_y = points[-1]
        body.append(text(end_x + 14, end_y + 5, name, INK, 13, "start"))

    # 범례
    for i, (name, _, color) in enumerate(series):
        y = top + 18 + i * 22
        body.append(line(left + 16, y - 4, left + 40, y - 4, color, 2))
        body.append(f"<circle cx='{left + 28}' cy='{y - 4}' r='4.5' fill='{color}' stroke='#ffffff' stroke-width='2'/>")
        body.append(text(left + 48, y, name, INK, 13, "start"))

    last = rows[-1]
    body.append(text(px(last[0]), top - 14, "합이 2²⁴에서 멈춘다 ↓", MUTED, 12, "end"))
    svg(OUT / "sum_error.svg", 680, 400, body)


if __name__ == "__main__":
    for figure in (fig_paths, fig_reduce, fig_pairwise, fig_sum_error):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
