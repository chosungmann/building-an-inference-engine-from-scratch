"""37장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch37.py
출력: book/src/images/ch37/*.svg

stream.svg 는 book/src/part4/output/ch37_speak.txt 에 남긴 실행 결과를 그린다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch37"  # noqa: F405
SPEAK = IMAGES.parents[0] / "part4/output/ch37_speak.txt"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_pipeline():
    """글에서 소리까지."""
    body = [text(20, 28, "글에서 소리까지", INK, 14, "start", "bold")]  # noqa: F405
    steps = [
        ("글", "the engine reads…", GRAY_BG, MUTED),  # noqa: F405
        ("언어 모델", "2부의 Llama", BLUE_BG, BLUE),  # noqa: F405
        ("오디오 토큰", "1초에 75개", GRAY_BG, MUTED),  # noqa: F405
        ("오디오 디코더", "합성곱, 어텐션", ORANGE_BG, ORANGE),  # noqa: F405
        ("스펙트럼", "토막마다 크기와 위상", GRAY_BG, MUTED),  # noqa: F405
        ("거꾸로 푸리에 변환", "겹쳐 더하기", GREEN_BG, GREEN),  # noqa: F405
        ("소리", "24kHz", GRAY_BG, MUTED),  # noqa: F405
    ]
    w, gap, y = 138, 20, 56
    for index, (title, sub, fill, stroke) in enumerate(steps):
        x = 20 + index * (w + gap)
        body += box(x, y, w, 50, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(x - gap + 2, y + 25, x - 3, y + 25, INK, 1.5, True))  # noqa: F405
    # 언어 모델이 쓰는 것.
    y2 = 150
    body.append(text(20, y2, "언어 모델이 쓰는 글 (낱말마다 한 줄)", INK, 13, "start", "bold"))  # noqa: F405
    cells = [("the", 40, GRAY_BG, MUTED), ("&lt;|t_0.12|&gt;", 86, PURPLE_BG, PURPLE), ("&lt;|code_start|&gt;", 104, PURPLE_BG, PURPLE)]  # noqa: F405
    cells += [(code, 54, ORANGE_BG, ORANGE) for code in ("&lt;|167|&gt;", "&lt;|1715|&gt;", "&lt;|379|&gt;", "…")]  # noqa: F405
    cells += [("&lt;|code_end|&gt;", 100, PURPLE_BG, PURPLE)]
    x = 20
    for label, width, fill, stroke in cells:
        body.append(rect(x, y2 + 14, width, 28, fill, stroke, 3))
        body.append(text(x + width / 2, y2 + 33, label, INK, 11))  # noqa: F405
        x += width + 4
    body.append(text(20, y2 + 64, "낱말", MUTED, 11, "start"))  # noqa: F405
    body.append(text(64, y2 + 64, "그 낱말의 길이(초)", PURPLE, 11, "start"))  # noqa: F405
    body.append(text(258, y2 + 64, "오디오 토큰: 소리의 75분의 1초씩", ORANGE, 11, "start"))  # noqa: F405
    body.append(text(20, y2 + 92, "글자, 표시, 소리가 모두 같은 단어 사전의 토큰이다. 언어 모델은 그것이 소리인 줄 모른 채 다음 토큰을 고른다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 1140, y2 + 110, body)  # noqa: F405


def fig_stream():
    """낱말마다의 소리가 만들어지는 때와 들려야 하는 때."""
    rows = []
    for row in SPEAK.read_text(encoding="utf-8").splitlines():
        found = re.match(r"\s+(\d+)\s+(\d+)ms\s+(\d+)\s+([\d.]+)초", row)
        if found:
            rows.append((int(found[2]) / 1000, int(found[3]) * 320 / 24000))
    words = "the engine reads a model file and turns text into speech".split()
    body = [text(20, 28, "낱말마다의 소리: 만들어진 때와 들리는 때 (초)", INK, 14, "start", "bold")]  # noqa: F405
    left, top, scale, row_h = 90, 70, 190, 22
    first = rows[0][0]
    total = first + sum(duration for _, duration in rows)
    for tick in range(0, int(total * 2) + 2):
        x = left + tick / 2 * scale
        body.append(line(x, top - 8, x, top + len(rows) * row_h, "#e5e7eb", 1))  # noqa: F405
        body.append(text(x, top - 14, f"{tick / 2:g}", MUTED, 11))  # noqa: F405
    playing = first
    late = 0.0
    for index, ((ready, duration), word) in enumerate(zip(rows, words)):
        y = top + index * row_h
        body.append(text(left - 10, y + 15, word, INK, 12, "end"))  # noqa: F405
        late = max(late, ready - playing)
        body.append(rect(left + playing * scale, y + 4, max(duration * scale - 2, 2), row_h - 8, BLUE, None, 3))  # noqa: F405
        body.append(f"<circle cx='{left + ready * scale:.1f}' cy='{y + row_h / 2:.1f}' r='5' fill='{ORANGE}' stroke='#ffffff' stroke-width='1.5'/>")  # noqa: F405
        playing += duration
    note = top + len(rows) * row_h + 24
    body.append(f"<circle cx='26' cy='{note - 4}' r='5' fill='{ORANGE}'/>")  # noqa: F405
    body.append(text(38, note, "그 낱말의 소리가 만들어진 때", INK, 12, "start"))  # noqa: F405
    body.append(rect(250, note - 10, 22, 10, BLUE, None, 3))  # noqa: F405
    body.append(text(280, note, "첫 소리가 나오자마자 틀기 시작했을 때 그 낱말이 들리는 동안", INK, 12, "start"))  # noqa: F405
    body.append(text(20, note + 22, f"점이 막대의 왼쪽 끝보다 왼쪽에 있어야 소리가 끊기지 않는다. 둘째 낱말에서 {late * 1000:.0f}밀리초 늦는다. 그 뒤로는 만드는 쪽이 앞서 간다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "stream.svg", 860, note + 38, body)  # noqa: F405


if __name__ == "__main__":
    fig_pipeline()
    if SPEAK.exists():
        fig_stream()
