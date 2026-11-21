"""32장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch32.py
출력: book/src/images/ch32/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch32"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_patches():
    """사진을 조각으로 잘라 토큰의 줄로 만든다."""
    body = [text(20, 28, "사진 한 장이 토큰 197개가 되기까지", INK, 14, "start", "bold")]  # noqa: F405
    # 사진과 격자.
    x0, y0, cell, n = 30, 60, 11, 14
    body.append(rect(x0, y0, n * cell, n * cell, GRAY_BG, MUTED, 0, 1.2))  # noqa: F405
    for index in range(1, n):
        body.append(line(x0 + index * cell, y0, x0 + index * cell, y0 + n * cell, MUTED, 0.5))  # noqa: F405
        body.append(line(x0, y0 + index * cell, x0 + n * cell, y0 + index * cell, MUTED, 0.5))  # noqa: F405
    body.append(rect(x0 + 3 * cell, y0 + 2 * cell, cell, cell, BLUE_BG, BLUE, 0, 1.6))  # noqa: F405
    body.append(text(x0 + n * cell / 2, y0 + n * cell + 18, "224 × 224 사진", INK, 12))  # noqa: F405
    body.append(text(x0 + n * cell / 2, y0 + n * cell + 34, "16 × 16 조각 196개", MUTED, 11))  # noqa: F405
    # 조각 하나.
    x1 = 230
    body.append(line(x0 + 4 * cell + 2, y0 + 2.5 * cell, x1 - 6, 96, BLUE, 1.2, arrow=True))  # noqa: F405
    body.append(rect(x1, 76, 40, 40, BLUE_BG, BLUE, 0, 1.4))  # noqa: F405
    body.append(text(x1 + 20, 132, "값 768개", INK, 11))  # noqa: F405
    body.append(text(x1 + 20, 146, "(3 × 16 × 16)", MUTED, 10))  # noqa: F405
    # 선형 층.
    x2 = 330
    body.append(line(x1 + 44, 96, x2 - 6, 96, INK, 1.2, arrow=True))  # noqa: F405
    body += box(x2, 74, 120, 44, "선형 층", "768 → 192", ORANGE_BG, ORANGE)  # noqa: F405
    # 토큰의 줄.
    x3, y3, w, h = 520, 70, 34, 52
    body.append(line(x2 + 124, 96, x3 - 6, 96, INK, 1.2, arrow=True))  # noqa: F405
    labels = ["분류", "1", "2", "3", "…", "196"]
    for index, label in enumerate(labels):
        fill, stroke = (PURPLE_BG, PURPLE) if index == 0 else (BLUE_BG, BLUE)  # noqa: F405
        if label == "…":
            body.append(text(x3 + index * (w + 6) + w / 2, y3 + 30, "…", INK, 14))  # noqa: F405
            continue
        body.append(rect(x3 + index * (w + 6), y3, w, h, fill, stroke, 3, 1.2))
        body.append(text(x3 + index * (w + 6) + w / 2, y3 + 30, label, INK, 11))  # noqa: F405
    body.append(text(x3 + 3 * (w + 6), y3 - 10, "토큰 197개, 폭 192", INK, 12))  # noqa: F405
    body.append(text(x3 + w / 2, y3 + h + 16, "학습된", PURPLE, 10))
    body.append(text(x3 + w / 2, y3 + h + 28, "벡터", PURPLE, 10))
    # 위치.
    y4 = y3 + h + 44
    for index, label in enumerate(labels):
        if label == "…":
            continue
        body.append(rect(x3 + index * (w + 6), y4, w, 20, GREEN_BG, GREEN, 3, 1))  # noqa: F405
        body.append(text(x3 + index * (w + 6) + w / 2, y4 - 4, "+", INK, 11))  # noqa: F405
    body.append(text(x3 + 6 * (w + 6) + 6, y4 + 14, "자리마다의 위치 벡터를 더한다", GREEN, 11, "start"))  # noqa: F405
    # 블록과 머리.
    x4 = 520
    y5 = y4 + 48
    body += box(x4, y5, 240, 40, "블록 12개", "정규화·어텐션, 정규화·MLP", GRAY_BG, MUTED)  # noqa: F405
    body.append(line(x4 + 120, y4 + 22, x4 + 120, y5 - 4, INK, 1.2, arrow=True))  # noqa: F405
    body += box(x4 + 290, y5, 150, 40, "점수 1,000개", "분류 토큰만 꺼내서", PURPLE_BG, PURPLE)  # noqa: F405
    body.append(line(x4 + 244, y5 + 20, x4 + 286, y5 + 20, INK, 1.2, arrow=True))  # noqa: F405
    note = y5 + 70
    body.append(text(20, note, "조각 하나가 단어 하나의 자리를 맡는다. 그 뒤는 2부의 트랜스포머와 같고, 마스크가 없어 모든 조각이 모든 조각을 본다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "patches.svg", 1000, note + 18, body)  # noqa: F405


def fig_grids():
    """격자 셋, 칸 하나의 예측, 비최대 억제."""
    body = [text(20, 28, "검출 모델의 출력: 칸마다의 예측을 상자로, 겹친 상자를 하나로", INK, 14, "start", "bold")]  # noqa: F405
    # 격자 셋.
    y0 = 64
    specs = [(52, 8, 20), (26, 16, 190), (13, 32, 360)]
    for cells, stride, x in specs:
        side = 130
        body.append(rect(x, y0, side, side, GRAY_BG, MUTED, 0, 1))  # noqa: F405
        step = side / cells
        thin = 0.25 if cells == 52 else 0.4
        for index in range(1, cells):
            body.append(line(x + index * step, y0, x + index * step, y0 + side, MUTED, thin))  # noqa: F405
            body.append(line(x, y0 + index * step, x + side, y0 + index * step, MUTED, thin))  # noqa: F405
        body.append(text(x + side / 2, y0 + side + 18, f"{cells} × {cells}칸 (칸 하나 {stride}픽셀)", INK, 11))  # noqa: F405
    body.append(text(20 + 235, y0 - 10, "격자 셋: 2,704 + 676 + 169 = 3,549칸", INK, 12))  # noqa: F405
    body.append(text(20 + 65, y0 + 166, "작은 물체를 맡는다", MUTED, 10))  # noqa: F405
    body.append(text(360 + 65, y0 + 166, "큰 물체를 맡는다", MUTED, 10))  # noqa: F405
    # 칸 하나의 예측 85개.
    x1, y1 = 540, 74
    body.append(text(x1, y1 - 20, "칸 하나의 예측: 값 85개", INK, 12, "start"))  # noqa: F405
    parts = [("중심 2", 60, BLUE_BG, BLUE), ("크기 2", 60, BLUE_BG, BLUE), ("물체 1", 60, ORANGE_BG, ORANGE), ("종류 80", 230, GREEN_BG, GREEN)]  # noqa: F405
    x = x1
    for label, width, fill, stroke in parts:
        body.append(rect(x, y1, width, 28, fill, stroke, 0, 1.2))
        body.append(text(x + width / 2, y1 + 19, label, INK, 11))  # noqa: F405
        x += width
    body.append(text(x1, y1 + 50, "상자 = (칸의 위치 + 중심) × 칸의 크기,  exp(크기) × 칸의 크기", INK, 11, "start"))  # noqa: F405
    body.append(text(x1, y1 + 68, "점수 = 물체 × 종류.  임계값(0.3)을 넘는 것만 남긴다", INK, 11, "start"))  # noqa: F405
    # 비최대 억제.
    y2 = 270
    body.append(text(20, y2 - 12, "비최대 억제", INK, 12, "start", "bold"))  # noqa: F405
    def scene(x, kept_only):
        out = [rect(x, y2, 250, 150, "#ffffff", MUTED, 0, 1)]  # noqa: F405
        boxes = [(40, 30, 110, 90, 0.90, True), (48, 36, 108, 92, 0.72, False), (34, 26, 118, 96, 0.61, False), (44, 22, 104, 86, 0.55, False), (150, 60, 70, 70, 0.48, True)]
        for order, (bx, by, bw, bh, score, keep) in enumerate(boxes):
            if kept_only and not keep:
                continue
            color = BLUE if keep else ORANGE  # noqa: F405
            dash = "" if keep or kept_only else " stroke-dasharray='4 3'"
            out.append(f"<rect x='{x + bx}' y='{y2 + by}' width='{bw}' height='{bh}' fill='none' stroke='{color}' stroke-width='1.6'{dash}/>")
            # 겹친 상자들의 점수는 왼쪽에 줄지어 적는다.
            if keep:
                out.append(text(x + bx + bw - 2, y2 + by - 3, f"{score:.2f}", color, 10, "end"))  # noqa: F405
            else:
                out.append(text(x + 28, y2 + 44 + order * 16, f"{score:.2f}", color, 10, "end"))  # noqa: F405
        return out
    body += scene(20, False)
    body += scene(330, True)
    body.append(line(276, y2 + 75, 324, y2 + 75, INK, 1.4, arrow=True))  # noqa: F405
    body.append(text(145, y2 + 168, "임계값을 넘은 상자들. 한 물체에 여럿이다", MUTED, 11))  # noqa: F405
    body.append(text(455, y2 + 168, "점수가 높은 것부터 남기고, 많이 겹치는 것은 버린다", MUTED, 11))  # noqa: F405
    body.append(text(610, y2 + 40, "겹친 정도 = 겹친 넓이 ÷ 합친 넓이", INK, 11, "start"))  # noqa: F405
    body.append(text(610, y2 + 60, "같은 종류이고 0.45보다 많이 겹치면", INK, 11, "start"))  # noqa: F405
    body.append(text(610, y2 + 78, "같은 물체로 본다", INK, 11, "start"))  # noqa: F405
    body.append(text(610, y2 + 106, "떨어져 있는 상자는 점수가 낮아도 남는다", MUTED, 11, "start"))  # noqa: F405
    svg(OUT / "grids.svg", 1000, y2 + 186, body)  # noqa: F405


if __name__ == "__main__":
    fig_patches()
    fig_grids()
