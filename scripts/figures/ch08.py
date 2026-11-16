"""8장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch08.py
출력: book/src/images/ch08/*.svg

병합 그림은 예제의 실행 결과 파일에서 그린다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch08"  # noqa: F405
OUTPUT = IMAGES.parent / "part2/output"  # noqa: F405


def fig_pipeline():
    body = []
    stages = [
        ("글", '"Hello world"', GRAY_BG, MUTED),
        ("사전 분할", '"Hello" + " world"', ORANGE_BG, ORANGE),
        ("바이트로 펼치기", "H e l l o / Ġ w o r l d", BLUE_BG, BLUE),
        ("병합", "Hello / Ġworld", BLUE_BG, BLUE),
        ("번호", "9906, 1917", GREEN_BG, GREEN),
    ]
    x, y, w, h, gap = 20, 50, 152, 62, 20
    body.append(text(x + 2.5 * w + 2 * gap, 28, "부호화: 글을 토큰 번호로", INK, 14, weight="bold"))
    for i, (title, sub, fill, stroke) in enumerate(stages):
        bx = x + i * (w + gap)
        body += box(bx, y, w, h, title, sub, fill, stroke)
        if i < len(stages) - 1:
            body.append(line(bx + w, y + h / 2, bx + w + gap - 4, y + h / 2, MUTED, 2, arrow=True))

    y2 = 190
    back = [
        ("번호", "9906, 1917", GREEN_BG, GREEN),
        ("토큰의 바이트", "48 65 6c 6c 6f / 20 77 …", BLUE_BG, BLUE),
        ("이어 붙이기", "바이트 11개", BLUE_BG, BLUE),
        ("UTF-8로 해석", '"Hello world"', GRAY_BG, MUTED),
    ]
    body.append(text(x + 2.5 * w + 2 * gap, y2 - 22, "복호화: 토큰 번호를 글로", INK, 14, weight="bold"))
    ox = x + 0.5 * (w + gap)
    for i, (title, sub, fill, stroke) in enumerate(back):
        bx = ox + i * (w + gap)
        body += box(bx, y2, w, h, title, sub, fill, stroke)
        if i < len(back) - 1:
            body.append(line(bx + w, y2 + h / 2, bx + w + gap - 4, y2 + h / 2, MUTED, 2, arrow=True))
    body.append(text(x + 2.5 * w + 2 * gap, 286, "두 방향 모두 단어 사전을 쓴다. 병합 규칙은 부호화에만 필요하다.", MUTED, 12))
    svg(OUT / "pipeline.svg", 880, 305, body)


def fig_merges():
    lines = (OUTPUT / "ch08_tokenize.txt").read_text(encoding="utf-8").splitlines()
    steps = [re.findall(r"\[([^\]]*)\]", l) for l in lines if re.match(r"^\s*\d+: \[", l)]
    body = [text(430, 26, "\" tokenization\"이 토큰 두 개가 되기까지 (GPT-2의 병합 규칙)", INK, 14, weight="bold")]
    unit, x0, y0, row = 44, 80, 46, 30
    for s, symbols in enumerate(steps):
        y = y0 + s * row
        body.append(text(x0 - 14, y + 17, str(s), MUTED, 12, "end"))
        previous = steps[s - 1] if s else symbols
        x = x0
        for symbol in symbols:
            width = unit * len(symbol)
            fresh = s > 0 and symbol not in previous
            fill, stroke = (ORANGE_BG, ORANGE) if fresh else (BLUE_BG, BLUE)
            body.append(
                f"<rect x='{x + 2}' y='{y}' width='{width - 4}' height='24' rx='5' fill='{fill}' stroke='{stroke}' stroke-width='1.3'/>"
            )
            body.append(text(x + width / 2, y + 17, symbol, INK, 13))
            x += width
    yb = y0 + len(steps) * row + 14
    body.append(text(x0 - 14, y0 - 8, "단계", MUTED, 11, "end"))
    body.append(text(430, yb, "주황색: 그 단계에서 새로 합쳐진 토큰. Ġ는 공백을 적는 표기다.", MUTED, 12))
    svg(OUT / "merges.svg", 860, yb + 16, body)


def fig_utf8():
    body = [text(400, 26, "글자의 경계와 토큰의 경계는 다를 수 있다", INK, 14, weight="bold")]
    # (글자, 바이트들)
    chars = [("추", ["ec", "b6", "94"]), ("론", ["eb", "a1", "a0"]), ("␠", ["20"]), ("🦙", ["f0", "9f", "a6", "99"])]
    tokens = [(3, "103948"), (3, "103778"), (3, "11410"), (1, "99"), (1, "247")]
    w, x0, y_char, y_byte, y_tok = 58, 90, 56, 112, 178
    x = x0
    for label, bs in chars:
        width = w * len(bs)
        body.append(f"<rect x='{x + 2}' y='{y_char}' width='{width - 4}' height='34' rx='6' fill='{GRAY_BG}' stroke='{MUTED}'/>")
        body.append(text(x + width / 2, y_char + 23, label, INK, 15))
        for i, b in enumerate(bs):
            body.append(cell(x + i * w, y_byte, w, 34, b, "#ffffff", MUTED, 13))
        x += width
    x = x0
    for i, (count, token_id) in enumerate(tokens):
        width = w * count
        splits = i >= 2
        fill, stroke = (ORANGE_BG, ORANGE) if splits else (BLUE_BG, BLUE)
        body.append(f"<rect x='{x + 2}' y='{y_tok}' width='{width - 4}' height='34' rx='6' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>")
        body.append(text(x + width / 2, y_tok + 22, token_id, INK, 12))
        x += width
    for y, label in [(y_char, "글자"), (y_byte, "UTF-8 바이트"), (y_tok, "토큰")]:
        body.append(text(x0 - 12, y + 22, label, MUTED, 12, "end"))
    body.append(text(400, 246, "주황색 토큰 셋이 모여야 공백과 🦙가 된다. 뒤의 둘은 혼자서는 글자가 아니다.", MUTED, 12))
    svg(OUT / "utf8_tokens.svg", 800, 264, body)


if __name__ == "__main__":
    for figure in (fig_pipeline, fig_merges, fig_utf8):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
