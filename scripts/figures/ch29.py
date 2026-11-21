"""29장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch29.py
출력: book/src/images/ch29/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch29"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_scale():
    """실수의 구간을 정수 256개에 대응시킨다."""
    body = [text(20, 28, "스케일과 영점: 실수의 한 구간을 정수 256개에 나눠 담는다", INK, 14, "start", "bold")]  # noqa: F405
    x0, x1 = 110, 830
    scale, zero = 6.0 / 255.0, -128

    def place(real):
        return x0 + (real - 0.0) / 6.0 * (x1 - x0)

    # 실수의 축.
    y = 80
    y2 = 170
    # 구간 밖은 잘린다. 축보다 먼저 그려 글자를 가리지 않게 한다.
    body.append(rect(x1 + 18, y - 22, 34, y2 - y + 44, ORANGE_BG, None))  # noqa: F405
    body.append(rect(x0 - 52, y - 22, 34, y2 - y + 44, ORANGE_BG, None))  # noqa: F405
    body.append(line(x0 - 52, y, x1 + 56, y, INK, 1.5, arrow=True))  # noqa: F405
    body.append(text(x0 - 60, y + 5, "실수", INK, 12, "end"))  # noqa: F405
    for real in range(0, 7):
        x = place(real)
        body.append(line(x, y - 5, x, y + 5, INK, 1.5))  # noqa: F405
        body.append(text(x, y - 12, f"{real}.0", INK, 11))  # noqa: F405
    # 정수의 축.
    body.append(line(x0 - 52, y2, x1 + 56, y2, PURPLE, 1.5, arrow=True))
    body.append(text(x0 - 60, y2 + 5, "정수", PURPLE, 12, "end"))
    for q in (-128, -64, 0, 64, 127):
        real = scale * (q - zero)
        x = place(real)
        body.append(line(x, y2 - 5, x, y2 + 5, PURPLE, 1.5))
        body.append(text(x, y2 + 22, str(q), PURPLE, 11))
        body.append(line(x, y + 6, x, y2 - 6, MUTED, 0.8, dash="3 3"))  # noqa: F405
    # 잘리는 곳.
    body.append(text(x1 + 30, y2 + 46, "6보다 큰 값은", ORANGE, 10))  # noqa: F405
    body.append(text(x1 + 30, y2 + 58, "127로 잘린다", ORANGE, 10))  # noqa: F405
    body.append(text(x0 - 22, y2 + 46, "0보다 작은 값은", ORANGE, 10))  # noqa: F405
    body.append(text(x0 - 22, y2 + 58, "−128로 잘린다", ORANGE, 10))  # noqa: F405
    # 한 칸의 폭.
    a, b = place(scale * 96), place(scale * 97)
    body.append(text(place(2.3), y + 44, "한 칸의 폭이 스케일: 6 / 255 = 0.0235", INK, 12))  # noqa: F405
    y3 = y2 + 84
    body.append(text(20, y3, "ReLU6 뒤의 값(0에서 6 사이)을 담는 경우다. 스케일 0.0235, 영점 −128. 실수 = 스케일 × (정수 − 영점).", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y3 + 18, "영점은 실수 0이 놓이는 정수다. 0을 오차 없이 나타낼 수 있어야 패딩과 ReLU가 정확하다.", MUTED, 12, "start"))  # noqa: F405
    _ = (a, b)
    svg(OUT / "scale.svg", 930, y3 + 34, body)  # noqa: F405


def fig_multiply():
    """실수 배율을 정수의 곱셈과 자리 옮기기로."""
    body = [text(20, 28, "실수를 곱하는 일을 정수로: 배율 = (31비트 정수) × 2의 거듭제곱", INK, 14, "start", "bold")]  # noqa: F405
    y = 70
    steps = [
        (30, 190, "누산값", "32비트 정수", BLUE_BG, BLUE),  # noqa: F405
        (270, 220, "× 배율의 가수", "0.5 이상 1 미만을 2³¹배한 정수", PURPLE_BG, PURPLE),
        (540, 170, "64비트 곱", "값이 2³¹배 커져 있다", GRAY_BG, MUTED),  # noqa: F405
        (760, 140, "오른쪽으로 옮김", "31 − 지수 칸", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    for x, w, title, sub, fill, stroke in steps:
        body += box(x, y, w, 56, title, sub, fill, stroke)  # noqa: F405
    for left, right in [(220, 268), (490, 538), (710, 758)]:
        body.append(line(left, y + 28, right, y + 28, INK, 1.5, arrow=True))  # noqa: F405
    y2 = y + 96
    body.append(text(20, y2, "예: 배율 0.000731 = 0.7485 × 2⁻¹⁰. 가수는 0.7485 × 2³¹ = 1,607,486,000, 지수는 −10.", INK, 12, "start"))  # noqa: F405
    body.append(text(20, y2 + 20, "누산값 12,345에 곱하면 12,345 × 1,607,486,000을 41칸(31 + 10) 오른쪽으로 옮겨 9가 된다. 12,345 × 0.000731 = 9.02.", INK, 12, "start"))  # noqa: F405
    y3 = y2 + 56
    body.append(text(20, y3, "옮기면서 버려지는 자리를 어떻게 반올림하느냐가 세 가지다", INK, 12, "start", "bold"))  # noqa: F405
    rows = [
        ("한 번", "64비트 곱에 절반을 더하고 한 번에 옮긴다", "합성곱, 덧셈"),
        ("두 번", "위쪽 32비트를 반올림해 얻고, 다시 옮기며 반올림한다", "깊이별 합성곱, 평균"),
        ("두 번, 큰 쪽으로", "두 번 모두 절반을 큰 쪽으로 보낸다", "선형 층"),
    ]
    for index, (name, how, where) in enumerate(rows):
        yy = y3 + 14 + index * 26
        body.append(rect(20, yy, 150, 22, PURPLE_BG, PURPLE, 4, 1))
        body.append(text(95, yy + 15, name, INK, 11))  # noqa: F405
        body.append(text(184, yy + 15, how, INK, 11, "start"))  # noqa: F405
        body.append(text(900, yy + 15, where, MUTED, 11, "end"))  # noqa: F405
    y4 = y3 + 14 + 3 * 26 + 18
    body.append(text(20, y4, "수학으로는 같은 계산이고 결과가 드물게 1씩 다르다. 오른쪽은 LiteRT의 자체 커널이 그 방법을 쓰는 연산이다(29.7절).", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "multiply.svg", 930, y4 + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_scale()
    fig_multiply()
    print("그림을", OUT, "에 썼습니다")
