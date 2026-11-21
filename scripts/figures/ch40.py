"""40장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch40.py
출력: book/src/images/ch40/*.svg

scaling.svg 는 bench/ch40/ 에 남긴 측정값의 중앙값을 그린다. 그 파일이
없으면 그 그림은 건너뛴다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch40"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch40"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"
TEXTS = [(BLUE, BLUE_BG), (ORANGE, ORANGE_BG), (GREEN, GREEN_BG), (PURPLE, PURPLE_BG)]  # noqa: F405


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_batch():
    """여러 글의 새 토큰이 한 텐서의 줄들이 된다."""
    body = [text(20, 28, "여러 글의 새 토큰을 한 묶음으로", INK, 14, "start", "bold")]  # noqa: F405
    # 왼쪽: 글 셋과 그 캐시.
    texts = [("글 가", "답을 쓰는 중", 1, 9), ("글 나", "답을 쓰는 중", 1, 4), ("글 다", "글을 읽는 중", 3, 2)]
    rows = []
    y = 62
    for index, (name, state, new, past) in enumerate(texts):
        color, bg = TEXTS[index]
        body.append(text(20, y + 15, name, INK, 13, "start", "bold"))  # noqa: F405
        body.append(text(20, y + 31, state, MUTED, 11, "start"))  # noqa: F405
        for token in range(past):
            body.append(rect(100 + token * 16, y + 6, 13, 18, GRAY_BG, MUTED))  # noqa: F405
        for token in range(new):
            body.append(rect(100 + (past + token) * 16, y + 6, 13, 18, bg, color, 0, 1.5))
            rows.append((color, bg, name))
        y += 46
    body.append(text(100, y + 4, "회색: 캐시에 있는 토큰.  색: 이번 스텝의 새 토큰.", MUTED, 11, "start"))  # noqa: F405

    # 가운데: 한 텐서.
    x0, y0 = 330, 62
    body.append(text(x0 + 60, y0 - 10, "한 텐서 (다섯 줄)", INK, 12, "middle", "bold"))  # noqa: F405
    for index, (color, bg, name) in enumerate(rows):
        body.append(rect(x0, y0 + index * 24, 120, 20, bg, color, 3, 1.5))
        body.append(text(x0 + 60, y0 + index * 24 + 14, name, INK, 11))  # noqa: F405
    body.append(line(280, y0 + 60, x0 - 8, y0 + 60, INK, 1.5, True))  # noqa: F405

    # 오른쪽: 블록 안의 단계들.
    steps = [
        ("질의·키·값 만들기", "선형 층. 다섯 줄을 한 번에", GREEN_BG, GREEN),  # noqa: F405
        ("앞 토큰들과 견줘 섞기", "줄마다 제 글의 캐시에서", ORANGE_BG, ORANGE),  # noqa: F405
        ("모으기, MLP", "선형 층. 다섯 줄을 한 번에", GREEN_BG, GREEN),  # noqa: F405
        ("출력 층", "글마다 마지막 줄만. 세 줄을 한 번에", GREEN_BG, GREEN),  # noqa: F405
    ]
    x1 = 520
    for index, (title, sub, fill, stroke) in enumerate(steps):
        y = 48 + index * 60
        body += box(x1, y, 300, 46, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(x1 + 150, y - 14, x1 + 150, y - 3, INK, 1.5, True))  # noqa: F405
    body.append(line(x0 + 128, y0 + 60, x1 - 8, y0 + 60, INK, 1.5, True))  # noqa: F405
    body.append(text(20, 304, "초록 단계는 줄이 어느 글의 것인지 모른다. 가중치를 한 번 읽어 모든 줄에 쓴다. 글을 가리는 곳은 주황 단계 하나다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "batch.svg", 860, 320, body)  # noqa: F405


def fig_policies():
    """정적 배칭과 연속 배칭을 시간 위에 그린다."""
    body = [text(20, 28, "요청 다섯이 차례로 올 때 (자리 셋)", INK, 14, "start", "bold")]  # noqa: F405
    # (도착, 길이) 다섯 개. 시간의 단위는 스텝.
    jobs = [(0, 10), (1, 4), (2, 7), (4, 5), (6, 6)]
    unit = 26

    def lane(y0, title, starts):
        body.append(text(20, y0 + 12, title, INK, 13, "start", "bold"))  # noqa: F405
        waiting = 0
        for index, ((arrive, length), (slot, start)) in enumerate(zip(jobs, starts)):
            color, bg = TEXTS[index % 4]
            y = y0 + 26 + slot * 30
            x = 150 + start * unit
            body.append(rect(x, y, length * unit - 3, 24, bg, color, 4, 1.5))
            body.append(text(x + length * unit / 2, y + 17, f"요청 {index + 1}", INK, 11))  # noqa: F405
            # 기다린 시간은 자리 아래의 줄에 따로 그린다.
            if start > arrive:
                y = y0 + 26 + 3 * 30 + waiting * 16
                body.append(rect(150 + arrive * unit, y, (start - arrive) * unit - 3, 12, GRAY_BG, color))  # noqa: F405
                body.append(text(150 + arrive * unit - 6, y + 10, f"{index + 1}", MUTED, 10, "end"))  # noqa: F405
                waiting += 1
        for slot in range(3):
            body.append(text(140, y0 + 26 + slot * 30 + 17, f"자리 {slot + 1}", MUTED, 11, "end"))  # noqa: F405
        body.append(text(110, y0 + 26 + 3 * 30 + 10, "기다림", MUTED, 11, "end"))  # noqa: F405

    # 정적: 셋이 모이면 시작(스텝 2), 묶음이 끝나는 스텝 12에 다음 묶음.
    lane(44, "정적 배칭", [(0, 2), (1, 2), (2, 2), (0, 12), (1, 12)])
    for step in (2, 12):
        body.append(line(150 + step * unit, 62, 150 + step * unit, 226, MUTED, 1, False, "4,3"))  # noqa: F405
    body.append(text(150 + 9 * unit, 246, "첫 묶음이 다 끝날 때까지 자리 둘이 비어 있어도 태우지 않는다", MUTED, 11))  # noqa: F405
    # 연속: 오는 대로 빈 자리에.
    lane(266, "연속 배칭", [(0, 0), (1, 1), (2, 2), (1, 5), (2, 9)])
    body.append(text(150 + 9 * unit, 446, "스텝마다 끝난 요청은 내리고, 빈 자리가 있으면 바로 태운다", MUTED, 11))  # noqa: F405
    body.append(text(20, 472, "가로축은 스텝(순전파 한 번)이다. 회색 막대는 요청이 도착해서 탈 때까지 기다린 시간이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "policies.svg", 700, 488, body)  # noqa: F405


def fig_pages():
    """글 둘이 앞부분의 페이지를 함께 가리킨다."""
    body = [text(20, 28, "글은 페이지의 목록을 쥔다", INK, 14, "start", "bold")]  # noqa: F405
    # 풀의 페이지들.
    px, py, pw, ph = 300, 70, 78, 40
    labels = ["0", "1", "2", "3", "4", "5"]
    fills = [GREEN_BG, GREEN_BG, BLUE_BG, BLUE_BG, ORANGE_BG, GRAY_BG]  # noqa: F405
    strokes = [GREEN, GREEN, BLUE, BLUE, ORANGE, MUTED]  # noqa: F405
    notes = ["지시문", "지시문", "가의 질문", "가의 답", "나의 질문", "빈 페이지"]
    for index, label in enumerate(labels):
        x = px + index * (pw + 10)
        body.append(rect(x, py, pw, ph, fills[index], strokes[index], 4, 1.5))
        body.append(text(x + pw / 2, py + 17, f"페이지 {label}", INK, 11, "middle", "bold"))  # noqa: F405
        body.append(text(x + pw / 2, py + 33, notes[index], MUTED, 10))  # noqa: F405
    body.append(text(px, py - 12, "풀 (페이지 하나 = 토큰 16개의, 모든 블록의 키와 값)", MUTED, 11, "start"))  # noqa: F405

    def table(y, name, pages, color):
        body.append(text(20, y + 16, name, INK, 13, "start", "bold"))  # noqa: F405
        for slot, page in enumerate(pages):
            x = 110 + slot * 40
            body.append(rect(x, y, 34, 24, "#ffffff", color, 3, 1.5))
            body.append(text(x + 17, y + 17, str(page), INK, 12))  # noqa: F405
            target = px + page * (pw + 10) + pw / 2
            body.append(
                f"<path d='M{x + 17},{y} C{x + 17},{y - 30} {target},{py + ph + 40} {target},{py + ph + 3}' "
                f"fill='none' stroke='{color}' stroke-width='1.2' marker-end='url(#arrow-{color[1:]})'/>"
            )

    table(190, "대화 가", [0, 1, 2, 3], BLUE)  # noqa: F405
    table(236, "대화 나", [0, 1, 4], ORANGE)  # noqa: F405
    body.append(text(20, 292, "두 대화가 같은 지시문으로 시작한다. 그 페이지 둘(0, 1)은 한 벌만 있고 둘이 함께 가리킨다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, 310, "함께 가리키는 페이지에는 쓰지 않는다. 반쯤 찬 페이지를 이어 써야 하면 그 한 장만 베낀다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pages.svg", 840, 326, body)  # noqa: F405


def rows():
    out = []
    for path in sorted(BENCH.glob("round_*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            parts = row.split()
            if len(parts) > 5:
                out.append(parts)
    return out


def closed(data, model, threads, mode):
    """(처리량, 토큰 사이의 시간의 가운데 값)의 중앙값."""
    found = [p for p in data if p[:5] == ["badak", model, str(threads), mode, "closed"] and p[6] == mode.split(":")[-1]]
    if not found:
        return None
    pick = lambda key, offset=1: statistics.median(float(p[p.index(key) + offset]) for p in found)  # noqa: E731
    return pick("처리량"), pick("토큰사이")


def fig_scaling(data):
    """묶음을 키울 때의 처리량과 요청 하나의 속도."""
    body = [text(20, 28, "묶음을 키우면: 서버가 쓰는 토큰과 요청 하나가 받는 토큰 (스레드 12개)", INK, 14, "start", "bold")]  # noqa: F405
    body.append(rect(20, 44, 14, 10, BLUE))  # noqa: F405
    body.append(text(40, 54, "서버 전체 (토큰/초)", INK, 12, "start"))  # noqa: F405
    body.append(rect(200, 44, 14, 10, ORANGE))  # noqa: F405
    body.append(text(220, 54, "요청 하나 (토큰/초 = 1000 ÷ 토큰 사이의 밀리초)", INK, 12, "start"))  # noqa: F405
    sizes = [1, 2, 4, 8, 16]
    panels = [("smollm2_135m", "SmolLM2-135M"), ("llama32_1b", "Llama 3.2 1B")]
    for index, (model, name) in enumerate(panels):
        points = [closed(data, model, 12, f"continuous:{size}") for size in sizes]
        if any(point is None for point in points):
            continue
        x0, y0, h, bw = 70 + index * 440, 96, 180, 30
        top = max(point[0] for point in points)
        body.append(text(x0 + 180, y0 - 22, name, INK, 13, "middle", "bold"))  # noqa: F405
        body.append(line(x0 - 10, y0 + h, x0 + 370, y0 + h, MUTED, 1))  # noqa: F405
        for slot, (size, (rate, gap)) in enumerate(zip(sizes, points)):
            x = x0 + slot * 74
            each = 1000 / gap
            bar = rate / top * h
            body.append(rect(x, y0 + h - bar, bw, bar, BLUE))  # noqa: F405
            body.append(text(x + bw / 2, y0 + h - bar - 5, f"{rate:.0f}", INK, 11))  # noqa: F405
            bar = each / top * h
            body.append(rect(x + bw + 3, y0 + h - bar, bw, bar, ORANGE))  # noqa: F405
            body.append(text(x + bw + 3 + bw / 2, y0 + h - bar - 5, f"{each:.0f}", INK, 11))  # noqa: F405
            body.append(text(x + bw + 1, y0 + h + 16, f"{size}", INK, 12))  # noqa: F405
        body.append(text(x0 + 180, y0 + h + 34, "함께 도는 요청의 수", MUTED, 11))  # noqa: F405
    body.append(text(20, 338, "파란 막대는 오르고 주황 막대는 내린다. 서버는 더 많이 쓰고, 요청 하나는 더 천천히 받는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "scaling.svg", 940, 354, body)  # noqa: F405


if __name__ == "__main__":
    fig_batch()
    fig_policies()
    fig_pages()
    data = rows()
    if data:
        fig_scaling(data)
