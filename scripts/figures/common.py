"""그림 생성 스크립트가 함께 쓰는 SVG 도우미.

모든 그림은 흰 배경을 깔아 mdBook의 어떤 테마에서도 읽히게 한다.
"""

from pathlib import Path

IMAGES = Path(__file__).resolve().parents[2] / "book/src/images"
INK, MUTED, BLUE, ORANGE, GREEN = "#1f2933", "#6b7280", "#2563eb", "#d97706", "#059669"
BLUE_BG, ORANGE_BG, GREEN_BG, GRAY_BG = "#dbeafe", "#fef3c7", "#d1fae5", "#f3f4f6"
FONT = "font-family='-apple-system, Segoe UI, Noto Sans KR, sans-serif'"


def svg(path, width, height, body):
    head = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' {FONT} font-size='14'>"
        "<defs>"
        + "".join(
            f"<marker id='arrow-{color[1:]}' viewBox='0 0 10 10' refX='9' refY='5' "
            "markerWidth='7' markerHeight='7' orient='auto'>"
            f"<path d='M0,0 L10,5 L0,10 z' fill='{color}'/></marker>"
            for color in (INK, MUTED, BLUE, ORANGE, GREEN)
        )
        + "</defs>"
        f"<rect width='{width}' height='{height}' rx='8' fill='#ffffff'/>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + "".join(body) + "</svg>\n", encoding="utf-8")


def text(x, y, s, fill=INK, size=14, anchor="middle", weight="normal"):
    return (
        f"<text x='{x}' y='{y}' fill='{fill}' font-size='{size}' "
        f"text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"
    )


def cell(x, y, w, h, label, fill=GRAY_BG, stroke=MUTED, size=14):
    return (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='{stroke}'/>"
        + text(x + w / 2, y + h / 2 + 5, label, size=size)
    )


def line(x1, y1, x2, y2, stroke=INK, width=2, arrow=False, dash=None):
    extra = f" marker-end='url(#arrow-{stroke[1:]})'" if arrow else ""
    extra += f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' "
        f"stroke-width='{width}'{extra}/>"
    )


def grid(x, y, rows, cols, w, h, labels, fills=None):
    out = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            fill = fills[i] if fills else GRAY_BG
            out.append(cell(x + c * w, y + r * h, w, h, labels[i], fill))
    return out


def box(x, y, w, h, title, sub=None, fill=GRAY_BG, stroke=MUTED):
    """모서리가 둥근 상자. 제목 한 줄과 선택적인 설명 한 줄을 넣는다."""
    out = [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' "
        f"stroke='{stroke}' stroke-width='1.5'/>"
    ]
    if sub is None:
        out.append(text(x + w / 2, y + h / 2 + 5, title, weight="bold"))
    else:
        out.append(text(x + w / 2, y + h / 2 - 4, title, weight="bold"))
        out.append(text(x + w / 2, y + h / 2 + 16, sub, MUTED, 12))
    return out
