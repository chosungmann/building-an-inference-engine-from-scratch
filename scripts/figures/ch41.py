"""41장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch41.py
출력: book/src/images/ch41/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch41"  # noqa: F405
PURPLE, PURPLE_BG, RED = "#7c3aed", "#ede9fe", "#dc2626"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_mask():
    """점수를 지우고 고른다."""
    body = [text(20, 28, "고르기 전에, 지금 올 수 없는 토큰의 점수를 지운다", INK, 14, "start", "bold")]  # noqa: F405
    body.append(text(20, 62, "지금까지 쓴 글", MUTED, 12, "start"))  # noqa: F405
    body.append(rect(130, 44, 250, 28, GRAY_BG, MUTED, 4))  # noqa: F405
    body.append(text(140, 63, "{\"city\": \"Paris\", \"is_capital\":", INK, 13, "start"))  # noqa: F405
    body.append(text(400, 63, "기계가 기다리는 것: 참·거짓 (그 앞의 빈칸 하나)", GREEN, 12, "start"))  # noqa: F405

    # (토큰, 점수의 높이, 올 수 있는가)
    tokens = [(" \"", 150, False), (" true", 120, True), (" yes", 96, False), (" 1", 70, False), (" false", 52, True), ("\\n", 40, False), (" t", 24, True)]
    x0, base = 70, 250
    for panel, (title, masked) in enumerate((("모델이 낸 점수", False), ("지운 뒤의 점수", True))):
        x = x0 + panel * 440
        body.append(text(x + 170, 100, title, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(line(x - 10, base, x + 350, base, MUTED, 1))  # noqa: F405
        for index, (token, height, allowed) in enumerate(tokens):
            bx = x + index * 50
            if masked and not allowed:
                body.append(rect(bx, base - 4, 34, 4, GRAY_BG, MUTED))  # noqa: F405
                body.append(text(bx + 17, base - 12, "−∞", MUTED, 11))  # noqa: F405
            else:
                color = GREEN if allowed else (ORANGE if not masked else MUTED)  # noqa: F405
                body.append(rect(bx, base - height, 34, height, color))
            body.append(text(bx + 17, base + 16, f"'{token}'", INK, 11))  # noqa: F405
        pick = 0 if not masked else 1
        px = x + pick * 50 + 17
        body.append(text(px, base + 36, "고른다", INK, 11, "middle", "bold"))  # noqa: F405
    body.append(line(410, 180, 470, 180, INK, 1.5, True))  # noqa: F405
    body.append(text(20, 312, "모델은 따옴표를 열고 글을 쓰려 했다. 지우고 나면 올 수 있는 것 가운데 점수가 가장 높은 ' true'가 골라진다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "mask.svg", 900, 328, body)  # noqa: F405


def fig_speculate():
    """짐작하고 채점하기."""
    body = [text(20, 28, "짐작 넷을 큰 모델이 순전파 한 번으로 채점한다", INK, 14, "start", "bold")]  # noqa: F405
    w, h, x0 = 96, 30, 190
    known = ["The", " sea"]
    draft = [" is", " very", " deep", " and"]
    target = [" is", " very", " cold", " but", " we"]
    # 1. 초안.
    y = 56
    body.append(text(20, y + 20, "작은 모델의 짐작", INK, 12, "start", "bold"))  # noqa: F405
    for index, token in enumerate(known):
        body.append(rect(x0 + index * (w + 6), y, w, h, GRAY_BG, MUTED, 4))  # noqa: F405
        body.append(text(x0 + index * (w + 6) + w / 2, y + 20, token, INK, 12))  # noqa: F405
    for index, token in enumerate(draft):
        x = x0 + (index + 2) * (w + 6)
        body.append(rect(x, y, w, h, ORANGE_BG, ORANGE, 4, 1.5))  # noqa: F405
        body.append(text(x + w / 2, y + 20, token, INK, 12))  # noqa: F405
    # 2. 큰 모델의 선택: 자리마다 "이 토큰 다음에 올 것".
    y2 = 150
    body.append(text(20, y2 + 20, "큰 모델의 선택", INK, 12, "start", "bold"))  # noqa: F405
    body.append(text(20, y2 + 38, "(순전파 한 번, 다섯 줄)", MUTED, 11, "start"))  # noqa: F405
    for index, token in enumerate(target):
        x = x0 + (index + 2) * (w + 6)
        same = index < 2
        cut = index == 2
        fill, stroke = (GREEN_BG, GREEN) if same or cut else (GRAY_BG, MUTED)  # noqa: F405
        body.append(rect(x, y2, w, h, fill, stroke, 4, 1.5))
        body.append(text(x + w / 2, y2 + 20, token, INK if same or cut else MUTED, 12))  # noqa: F405
        # 위의 토큰에서 내려오는 화살표.
        body.append(line(x - 6 - w / 2 + 0, y + h + 2, x + w / 2 - 14, y2 - 4, MUTED, 1, True))  # noqa: F405
        if index < 4:
            mark = "같다" if same else ("다르다" if cut else "")
            if mark:
                body.append(text(x + w / 2, y + h + 46, mark, GREEN if same else RED, 11, "middle", "bold"))  # noqa: F405
    # 3. 결과.
    y3 = 236
    body.append(text(20, y3 + 20, "쓴 것", INK, 12, "start", "bold"))  # noqa: F405
    for index, token in enumerate(known + [" is", " very", " cold"]):
        x = x0 + index * (w + 6)
        fresh = index >= 2
        body.append(rect(x, y3, w, h, GREEN_BG if fresh else GRAY_BG, GREEN if fresh else MUTED, 4, 1.5 if fresh else 1))  # noqa: F405
        body.append(text(x + w / 2, y3 + 20, token, INK, 12))  # noqa: F405
    body.append(text(x0 + 5 * (w + 6) + 10, y3 + 20, "순전파 한 번에 토큰 셋", GREEN, 12, "start", "bold"))  # noqa: F405
    body.append(text(20, 300, "맞은 짐작 둘은 그대로 쓰고, 처음 틀린 자리에는 큰 모델의 선택을 쓴다. 그 뒤의 짐작과 점수는 버린다(틀린 글 뒤에서 계산한 것이다).", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "speculate.svg", 900, 316, body)  # noqa: F405


def fig_lora():
    """가중치 옆의 작은 행렬 둘."""
    body = [text(20, 28, "LoRA: 얼려 둔 가중치 옆에 작은 행렬 둘", INK, 14, "start", "bold")]  # noqa: F405
    # x → W → +
    body.append(rect(40, 150, 40, 40, GRAY_BG, MUTED, 4))  # noqa: F405
    body.append(text(60, 175, "x", INK, 14, "middle", "bold"))  # noqa: F405
    body.append(rect(190, 70, 140, 140, BLUE_BG, BLUE, 4, 1.5))  # noqa: F405
    body.append(text(260, 136, "W", INK, 18, "middle", "bold"))  # noqa: F405
    body.append(text(260, 158, "576 × 576", MUTED, 12))  # noqa: F405
    body.append(text(260, 176, "얼려 둔다", BLUE, 12))  # noqa: F405
    body.append(rect(190, 250, 140, 22, ORANGE_BG, ORANGE, 4, 1.5))  # noqa: F405
    body.append(text(260, 266, "A   8 × 576", INK, 12))  # noqa: F405
    body.append(rect(390, 226, 22, 70, ORANGE_BG, ORANGE, 4, 1.5))  # noqa: F405
    body.append(text(430, 266, "B   576 × 8", INK, 12, "start"))  # noqa: F405
    body.append(f"<circle cx='560' cy='170' r='16' fill='#ffffff' stroke='{INK}' stroke-width='1.5'/>")  # noqa: F405
    body.append(text(560, 176, "+", INK, 18, "middle", "bold"))  # noqa: F405
    body.append(rect(640, 150, 40, 40, GRAY_BG, MUTED, 4))  # noqa: F405
    body.append(text(660, 175, "y", INK, 14, "middle", "bold"))  # noqa: F405
    body.append(line(82, 162, 186, 142, INK, 1.5, True))  # noqa: F405
    body.append(line(82, 178, 186, 258, INK, 1.5, True))  # noqa: F405
    body.append(line(332, 140, 542, 164, INK, 1.5, True))  # noqa: F405
    body.append(line(332, 261, 386, 261, INK, 1.5, True))  # noqa: F405
    body.append(line(414, 250, 548, 184, INK, 1.5, True))  # noqa: F405
    body.append(text(486, 232, "× alpha / r", MUTED, 11, "start"))  # noqa: F405
    body.append(line(578, 170, 636, 170, INK, 1.5, True))  # noqa: F405
    body.append(text(720, 110, "값의 수", INK, 12, "start", "bold"))  # noqa: F405
    body.append(text(720, 132, "W: 331,776", BLUE, 12, "start"))  # noqa: F405
    body.append(text(720, 152, "A와 B: 9,216", ORANGE, 12, "start"))  # noqa: F405
    body.append(text(720, 184, "합치기: W + (alpha/r)·B·A를", MUTED, 11, "start"))  # noqa: F405
    body.append(text(720, 200, "한 번 계산해 W 자리에 둔다", MUTED, 11, "start"))  # noqa: F405
    body.append(text(720, 224, "따로 두기: 걸음마다 아래의", MUTED, 11, "start"))  # noqa: F405
    body.append(text(720, 240, "길을 함께 계산한다", MUTED, 11, "start"))  # noqa: F405
    body.append(text(20, 326, "학습시키는 것은 주황색뿐이다. 학습이 끝나면 그 둘이 어댑터가 된다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "lora.svg", 920, 342, body)  # noqa: F405


if __name__ == "__main__":
    fig_mask()
    fig_speculate()
    fig_lora()
