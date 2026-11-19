"""36장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch36.py
출력: book/src/images/ch36/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch36"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_pipeline():
    """사진이 토큰의 자리에 들어가기까지."""
    body = [text(20, 28, "사진이 언어 모델에 들어가기까지", INK, 14, "start", "bold")]  # noqa: F405
    y = 60
    steps = [
        ("사진", "512 × 512 × 3", GRAY_BG, MUTED),  # noqa: F405
        ("비전 인코더", "ViT, 블록 12개", BLUE_BG, BLUE),  # noqa: F405
        ("조각의 벡터", "1,024 × 768", GRAY_BG, MUTED),  # noqa: F405
        ("이웃 16개를 잇기", "64 × 12,288", ORANGE_BG, ORANGE),  # noqa: F405
        ("선형 층", "64 × 576", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    w, gap = 150, 26
    for index, (title, sub, fill, stroke) in enumerate(steps):
        x = 20 + index * (w + gap)
        body += box(x, y, w, 50, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(x - gap + 2, y + 25, x - 3, y + 25, INK, 1.5, True))  # noqa: F405
    body.append(text(20 + 3 * (w + gap) + w + gap / 2, y - 10, "커넥터", ORANGE, 12, "middle", "bold"))  # noqa: F405
    # 토큰들의 줄.
    y2 = 190
    body.append(text(20, y2 - 18, "언어 모델이 받는 벡터들 (폭 576)", INK, 13, "start", "bold"))  # noqa: F405
    cells = [("&lt;|im_start|&gt;", 86, GRAY_BG, MUTED), ("User", 44, GRAY_BG, MUTED), (":", 20, GRAY_BG, MUTED), ("표시", 40, PURPLE_BG, PURPLE), ("표시", 40, PURPLE_BG, PURPLE)]  # noqa: F405
    cells += [("", 9, ORANGE_BG, ORANGE)] * 24 + [("표시", 40, PURPLE_BG, PURPLE), ("Describe", 66, GRAY_BG, MUTED), ("this", 38, GRAY_BG, MUTED), ("image", 48, GRAY_BG, MUTED), (".", 20, GRAY_BG, MUTED), ("…", 26, GRAY_BG, MUTED), ("Assistant", 70, GRAY_BG, MUTED), (":", 20, GRAY_BG, MUTED)]  # noqa: F405
    x = 20
    first = last = None
    for label, width, fill, stroke in cells:
        body.append(rect(x, y2, width, 30, fill, stroke, 3))
        if label:
            body.append(text(x + width / 2, y2 + 20, label, INK, 11))  # noqa: F405
        else:
            first = x if first is None else first
            last = x + width
        x += width + 3
    body.append(line(first, y2 + 38, last, y2 + 38, ORANGE, 1.5))  # noqa: F405
    body.append(text((first + last) / 2, y2 + 56, "사진의 벡터 64개 (토큰 번호가 없다)", ORANGE, 12))  # noqa: F405
    x_from = 20 + 4 * (w + gap) + w / 2
    body.append(f"<path d='M{x_from},{y + 52} L{x_from},{y + 86} L{(first + last) / 2},{y + 86} L{(first + last) / 2},{y2 - 4}' fill='none' stroke='{ORANGE}' stroke-width='1.5' marker-end='url(#arrow-{ORANGE[1:]})'/>")  # noqa: F405
    body.append(text(20, y2 + 86, "글의 토큰은 표에서 벡터를 꺼내고, 사진의 자리에는 커넥터가 낸 벡터를 그대로 놓는다. 그 뒤로 언어 모델은 둘을 가리지 않는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 920, y2 + 104, body)  # noqa: F405


def fig_shuffle():
    """이웃한 조각 16개를 하나로."""
    body = [text(20, 28, "커넥터: 이웃한 조각 4 × 4개의 벡터를 이어 하나로", INK, 14, "start", "bold")]  # noqa: F405
    s, x0, y0 = 16, 40, 56
    colors = [(BLUE, BLUE_BG), (ORANGE, ORANGE_BG), (GREEN, GREEN_BG), (PURPLE, PURPLE_BG)]  # noqa: F405
    for row in range(8):
        for column in range(8):
            stroke, fill = colors[(row // 4) * 2 + column // 4]
            body.append(rect(x0 + column * s, y0 + row * s, s - 2, s - 2, fill, stroke, 2))
    body.append(text(x0 + 4 * s, y0 + 8 * s + 20, "조각 8 × 8개, 폭 768", MUTED, 12))  # noqa: F405
    body.append(text(x0 + 4 * s, y0 + 8 * s + 38, "(실제로는 32 × 32개)", MUTED, 11))  # noqa: F405
    body.append(line(x0 + 8 * s + 16, y0 + 4 * s, x0 + 8 * s + 76, y0 + 4 * s, INK, 1.5, True))  # noqa: F405
    x1 = x0 + 8 * s + 96
    for index, (stroke, fill) in enumerate(colors):
        y = y0 + 6 + index * 30
        body.append(rect(x1, y, 320, 20, fill, stroke, 3))
        body.append(text(x1 + 160, y + 15, "조각 16개의 값을 줄 순서로 이은 벡터 하나 (폭 12,288)", INK, 11))  # noqa: F405
    body.append(text(x1 + 160, y0 + 8 * s + 20, "긴 벡터 2 × 2개", MUTED, 12))  # noqa: F405
    body.append(line(x1 + 336, y0 + 4 * s, x1 + 396, y0 + 4 * s, INK, 1.5, True))  # noqa: F405
    body.append(text(x1 + 366, y0 + 4 * s - 10, "선형 층", MUTED, 11))  # noqa: F405
    x2 = x1 + 416
    for index, (stroke, fill) in enumerate(colors):
        y = y0 + 6 + index * 30
        body.append(rect(x2, y, 60, 20, fill, stroke, 3))
        body.append(text(x2 + 30, y + 15, "576", INK, 11))  # noqa: F405
    body.append(text(x2 + 30, y0 + 8 * s + 20, "토큰 4개", MUTED, 12))  # noqa: F405
    body.append(text(20, y0 + 8 * s + 66, "값은 버리지 않는다. 토큰의 수만 16분의 1이 된다. 언어 모델이 읽을 것이 1,024개에서 64개로 준다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "shuffle.svg", 760, y0 + 8 * s + 84, body)  # noqa: F405


if __name__ == "__main__":
    fig_pipeline()
    fig_shuffle()
