"""7장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch07.py
출력: book/src/images/ch07/*.svg

숫자 그림은 예제의 직접 그린 숫자를 그대로 쓴다. MNIST의 사진이 아니다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch07"  # noqa: F405
EXAMPLE = IMAGES.parents[2] / "crates/badak_models/examples/ch07_classify.rs"  # noqa: F405
GRID = "#e5e7eb"


def drawn_digit():
    """예제 소스에서 직접 그린 숫자의 28줄을 읽는다."""
    source = EXAMPLE.read_text(encoding="utf-8")
    art = re.search(r'const SEVEN: &str = "\\\n(.*?)";', source, re.S)[1]
    return art.split("\n")


def digit(x, y, cell, rows):
    shade = {"#": "#f9fafb", "+": "#9ca3af", ".": "#111827"}
    out = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            out.append(
                f"<rect x='{x + c * cell}' y='{y + r * cell}' width='{cell}' height='{cell}' "
                f"fill='{shade.get(ch, shade['.'])}' stroke='#374151' stroke-width='0.3'/>"
            )
    return out


def fig_pipeline():
    rows = drawn_digit()
    body = []
    body += digit(30, 60, 5, rows)
    body.append(text(100, 46, "사진", INK, 13, weight="bold"))
    body.append(text(100, 220, "u8 (28, 28)", MUTED, 12))

    steps = [
        ("전처리", "0~1로, 평균 빼고, 나누고, 펴기", "f32 (784)", ORANGE_BG, ORANGE),
        ("선형 층 1", "784 → 128", "f32 (128)", BLUE_BG, BLUE),
        ("ReLU", "음수를 0으로", "f32 (128)", GREEN_BG, GREEN),
        ("선형 층 2", "128 → 10", "f32 (10)", BLUE_BG, BLUE),
        ("argmax", "가장 큰 점수의 위치", "답: 7", ORANGE_BG, ORANGE),
    ]
    x, y, w, h, gap = 200, 100, 118, 60, 26
    body.append(line(172, y + h / 2, x - 4, y + h / 2, MUTED, 2, arrow=True))
    for i, (title, sub, out_label, fill, stroke) in enumerate(steps):
        bx = x + i * (w + gap)
        body += box(bx, y, w, h, title, None, fill, stroke)
        body.append(text(bx + w / 2, y + h + 20, sub, MUTED, 11))
        body.append(text(bx + w / 2, y - 12, out_label, INK, 12))
        if i < len(steps) - 1:
            body.append(line(bx + w, y + h / 2, bx + w + gap - 4, y + h / 2, MUTED, 2, arrow=True))
    body.append(text(x + 2.5 * w + 2 * gap, 46, "상자 위의 글자: 그 단계가 내놓는 텐서", MUTED, 12))

    # 모델 파일에서 가중치가 온다.
    fx = x + 1 * (w + gap)
    body += box(fx + 40, 228, 2 * w + gap + 60, 44, "model.safetensors", None, GRAY_BG, MUTED)
    body.append(line(fx + w / 2, 228, fx + w / 2, y + h + 30, MUTED, 1.5, arrow=True, dash="5 4"))
    body.append(line(fx + 2 * (w + gap) + w / 2, 228, fx + 2 * (w + gap) + w / 2, y + h + 30, MUTED, 1.5, arrow=True, dash="5 4"))
    body.append(line(x + w / 2, 250, fx + 36, 250, MUTED, 1.5, dash="5 4"))
    body.append(line(x + w / 2, 250, x + w / 2, y + h + 30, MUTED, 1.5, arrow=True, dash="5 4"))
    body.append(text(x + w / 2, 290, "평균과 표준편차", MUTED, 11))
    body.append(text(fx + w + 80, 290, "가중치와 편향", MUTED, 11))
    svg(OUT / "pipeline.svg", 950, 305, body)


def fig_flatten():
    rows = drawn_digit()
    body = []
    body.append(text(118, 30, "사진: (28, 28)", INK, 13, weight="bold"))
    body += digit(20, 44, 7, rows)
    # 한 행을 강조
    body.append(f"<rect x='20' y='{44 + 7 * 7}' width='196' height='7' fill='none' stroke='{ORANGE}' stroke-width='2'/>")
    body.append(text(118, 262, "7번째 행", ORANGE, 12))

    body.append(line(232, 140, 290, 140, MUTED, 2, arrow=True))
    body.append(text(261, 128, "펴기", MUTED, 12))

    x0, y0, cw = 304, 122, 9.2
    body.append(text(x0 + 30 * cw, 30, "모델의 입력: (784)", INK, 13, weight="bold"))
    body.append(text(x0 + 30 * cw, 50, "행을 차례로 이어 붙인다. 4장의 reshape와 같다.", MUTED, 12))
    shade = {"#": "#f9fafb", "+": "#9ca3af", ".": "#111827"}
    shown = [6, 7, 8]
    for k, r in enumerate(shown):
        ox = x0 + k * 20 * cw
        for c in range(20):
            ch = rows[r][c + 4]
            body.append(
                f"<rect x='{ox + c * cw}' y='{y0}' width='{cw}' height='22' fill='{shade.get(ch, shade['.'])}' stroke='#374151' stroke-width='0.3'/>"
            )
        color = ORANGE if r == 7 else MUTED
        body.append(f"<rect x='{ox}' y='{y0}' width='{20 * cw}' height='22' fill='none' stroke='{color}' stroke-width='{2 if r == 7 else 1}'/>")
        body.append(text(ox + 10 * cw, y0 + 42, f"{r}번째 행", color, 12))
    body.append(text(x0 - 6, y0 + 16, "…", MUTED, 14, "end"))
    body.append(text(x0 + 60 * cw + 6, y0 + 16, "…", MUTED, 14, "start"))
    body.append(text(x0 + 30 * cw, 210, "(행, 열) 픽셀은 입력의 행 × 28 + 열 번째 자리에 놓인다.", INK, 13))
    body.append(text(x0 + 30 * cw, 232, "그림은 각 행의 가운데 20칸만 그렸다.", MUTED, 11))
    svg(OUT / "flatten.svg", 880, 275, body)


def fig_verify():
    body = []
    stages = ["전처리", "선형 층 1", "ReLU", "선형 층 2"]
    x, w, gap = 150, 130, 40

    def row(y, title, fill, stroke):
        out = [text(30, y + 30, title, INK, 14, "start", "bold")]
        for i, stage in enumerate(stages):
            bx = x + i * (w + gap)
            out += box(bx, y, w, 48, stage, None, fill, stroke)
            if i < len(stages) - 1:
                out.append(line(bx + w, y + 24, bx + w + gap - 4, y + 24, MUTED, 2, arrow=True))
        return out

    body += row(40, "PyTorch", ORANGE_BG, ORANGE)
    body += row(190, "badak", BLUE_BG, BLUE)
    names = ["input", "fc1", "relu", "logits"]
    for i, name in enumerate(names):
        cx = x + i * (w + gap) + w / 2
        body.append(line(cx, 88, cx, 124, ORANGE, 1.5, arrow=True))
        body.append(line(cx, 190, cx, 158, BLUE, 1.5, arrow=True))
        body.append(
            f"<rect x='{cx - 46}' y='126' width='92' height='30' rx='15' fill='{GREEN_BG}' stroke='{GREEN}' stroke-width='1.5'/>"
        )
        body.append(text(cx, 146, f"{name} 비교", INK, 12))
    body.append(text(430, 268, "층마다 비교하면, 틀렸을 때 어느 층부터 틀렸는지 바로 알 수 있다.", MUTED, 13))
    body.append(text(430, 288, "PyTorch의 결과는 reference.safetensors에 들어 있다.", MUTED, 12))
    svg(OUT / "verify.svg", 860, 305, body)


if __name__ == "__main__":
    for figure in (fig_pipeline, fig_flatten, fig_verify):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
