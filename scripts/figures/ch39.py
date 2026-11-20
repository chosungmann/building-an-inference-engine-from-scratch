"""39장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch39.py
출력: book/src/images/ch39/*.svg

prefix.svg 와 queue.svg 는 bench/ch39/ 에 남긴 측정값의 중앙값을 그린다.
그 파일이 없으면 두 그림은 건너뛴다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch39"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch39"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_request():
    """요청 하나가 지나가는 길."""
    body = [text(20, 28, "요청 하나가 지나가는 길", INK, 14, "start", "bold")]  # noqa: F405
    top = [
        ("연결", "TCP, 스레드 하나", GRAY_BG, MUTED),  # noqa: F405
        ("HTTP 읽기", "머리 16KB, 몸통 4MB까지", BLUE_BG, BLUE),  # noqa: F405
        ("요청 해석", "JSON, 정해진 역할만", BLUE_BG, BLUE),  # noqa: F405
        ("토큰화", "틀과 내용을 따로", ORANGE_BG, ORANGE),  # noqa: F405
    ]
    bottom = [
        ("차례 기다리기", "한 번에 한 요청", PURPLE_BG, PURPLE),
        ("접두사 맞추기", "같은 앞부분은 캐시에", GREEN_BG, GREEN),  # noqa: F405
        ("나머지 읽기", "새 토큰만 순전파", GREEN_BG, GREEN),  # noqa: F405
        ("토큰 쓰기", "하나 쓸 때마다 조각", GREEN_BG, GREEN),  # noqa: F405
    ]
    w, gap = 190, 30
    for row, (steps, y) in enumerate(((top, 56), (bottom, 156))):
        for index, (title, sub, fill, stroke) in enumerate(steps):
            x = 150 + index * (w + gap)
            body += box(x, y, w, 50, title, sub, fill, stroke)  # noqa: F405
            if index:
                body.append(line(x - gap + 3, y + 25, x - 4, y + 25, INK, 1.5, True))  # noqa: F405
        if row == 0:
            body.append(text(70, y + 22, "클라이언트", INK, 13, "middle", "bold"))  # noqa: F405
            body.append(text(70, y + 40, "POST …/completions", MUTED, 11))  # noqa: F405
            body.append(line(118, y + 25, 146, y + 25, INK, 1.5, True))  # noqa: F405
    # 윗줄의 끝에서 아랫줄의 처음으로.
    x_end = 150 + 3 * (w + gap) + w / 2
    x_start = 150 + w / 2
    body.append(
        f"<path d='M{x_end},108 L{x_end},130 L{x_start},130 L{x_start},152' fill='none' "
        f"stroke='{INK}' stroke-width='1.5' marker-end='url(#arrow-{INK[1:]})'/>"  # noqa: F405
    )
    # 조각이 클라이언트로 돌아간다.
    body.append(
        f"<path d='M{x_end},208 L{x_end},240 L70,240 L70,102' fill='none' stroke='{GREEN}' "  # noqa: F405
        f"stroke-width='1.5' stroke-dasharray='5,4' marker-end='url(#arrow-{GREEN[1:]})'/>"  # noqa: F405
    )
    body.append(text(480, 258, "data: {…\"delta\":{\"content\":\" Paris\"}…}   토큰 하나에 조각 하나, 쓰는 대로 내보낸다", GREEN, 12))  # noqa: F405
    body.append(text(20, 288, "윗줄은 요청마다 따로 돈다(연결마다 스레드 하나). 아랫줄은 모델을 쓰므로 한 번에 한 요청만 지나간다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "request.svg", 1040, 304, body)  # noqa: F405


def rows():
    """bench/ch39 의 줄들을 (이름표, 모델, 스레드, 종류, 낱말들)로 읽는다."""
    out = []
    for path in sorted(BENCH.glob("round_*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            parts = row.split()
            if len(parts) > 4:
                out.append((parts[0], parts[1], int(parts[2]), parts[3], parts[4:]))
    return out


def number(words, key):
    """낱말들에서 `key` 바로 뒤의 숫자를 꺼낸다."""
    value = words[words.index(key) + 1]
    return None if value == "-" else float(value)


def pick(data, label, model, threads, kind, lead):
    """조건에 맞는 줄들의 낱말 목록."""
    return [
        words
        for (l, m, t, k, words) in data
        if (l, m, t, k) == (label, model, threads, kind) and words[: len(lead)] == lead
    ]


def median(found, key):
    return statistics.median(number(words, key) for words in found)


def fig_prefix(data, model, name):
    """대화가 이어질 때 차례마다 읽는 토큰과 첫 글까지의 시간."""
    body = [text(20, 28, f"대화 다섯 차례: 읽는 토큰과 첫 글까지의 시간 ({name}, 스레드 12개)", INK, 14, "start", "bold")]  # noqa: F405
    body.append(rect(20, 44, 14, 10, BLUE_BG, BLUE))  # noqa: F405
    body.append(text(40, 54, "앞 차례의 캐시에 있던 토큰", INK, 12, "start"))  # noqa: F405
    body.append(rect(230, 44, 14, 10, ORANGE))  # noqa: F405
    body.append(text(250, 54, "새로 읽은 토큰", INK, 12, "start"))  # noqa: F405
    body.append(text(700, 54, "첫 글까지 (ms)", INK, 12, "middle", "bold"))  # noqa: F405
    body.append(text(660, 72, "캐시를 쓰면", MUTED, 12))  # noqa: F405
    body.append(text(760, 72, "쓰지 않으면", MUTED, 12))  # noqa: F405
    turns = []
    for turn in range(1, 6):
        lead = ["캐시씀", "차례", str(turn)]
        used = pick(data, "badak-f32", model, 12, "turns", lead)
        fresh = pick(data, "badak-f32", model, 12, "turns", ["캐시안씀", "차례", str(turn)])
        turns.append((median(used, "읽음"), median(used, "캐시"), median(used, "첫글"), median(fresh, "첫글")))
    scale = 440 / max(read for read, *_ in turns)
    for index, (read, cached, with_cache, without) in enumerate(turns):
        y = 84 + index * 32
        body.append(text(70, y + 16, f"차례 {index + 1}", INK, 12, "end"))  # noqa: F405
        body.append(rect(80, y, max(cached * scale, 0.5), 22, BLUE_BG, BLUE))  # noqa: F405
        body.append(rect(80 + cached * scale, y, max((read - cached) * scale, 2), 22, ORANGE))  # noqa: F405
        body.append(text(80 + read * scale + 8, y + 16, f"{read:.0f}토큰 가운데 {read - cached:.0f}개", MUTED, 12, "start"))  # noqa: F405
        body.append(text(660, y + 16, f"{with_cache:.0f}", INK, 13, "middle", "bold"))  # noqa: F405
        body.append(text(760, y + 16, f"{without:.0f}", INK, 13))  # noqa: F405
    note = 84 + 5 * 32 + 20
    body.append(text(20, note, "첫 차례는 캐시가 비어 있어 둘이 같다. 그 뒤로는 새로 온 말만 읽는다. 쓰지 않으면 대화가 길어질수록 느려진다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "prefix.svg", 840, note + 16, body)  # noqa: F405


def fig_queue(data, model, name):
    """두 클라이언트가 같은 때에 요청을 보내면."""
    body = [text(20, 28, f"두 요청이 같은 때에 오면 ({name}, 스레드 12개, 64토큰씩)", INK, 14, "start", "bold")]  # noqa: F405
    legend = [("기다림", GRAY_BG, MUTED), ("첫 글까지", ORANGE, None), ("나머지 토큰", GREEN, None)]  # noqa: F405
    x = 20
    for label, fill, stroke in legend:
        body.append(rect(x, 44, 14, 10, fill, stroke))
        body.append(text(x + 20, 54, label, INK, 12, "start"))  # noqa: F405
        x += 110
    marks = []
    for label, lead in (("혼자 보낸 요청", "혼자"), ("둘 가운데 먼저", "먼저"), ("둘 가운데 나중", "나중")):
        found = pick(data, "badak-f32", model, 12, "queue", [lead])
        waited = 0.0 if lead == "혼자" else median(found, "기다림")
        marks.append((label, waited, median(found, "첫글"), median(found, "전체")))
    scale = 560 / max(total for *_, total in marks)
    for index, (label, waited, first, total) in enumerate(marks):
        y = 72 + index * 36
        body.append(text(130, y + 17, label, INK, 12, "end"))  # noqa: F405
        if waited > 0:
            body.append(rect(140, y, waited * scale, 24, GRAY_BG, MUTED))  # noqa: F405
            if waited * scale > 60:
                body.append(text(140 + waited * scale / 2, y + 17, f"{waited:.0f}", MUTED, 12))  # noqa: F405
        body.append(rect(140 + waited * scale, y, max((first - waited) * scale, 2), 24, ORANGE))  # noqa: F405
        body.append(rect(140 + first * scale, y, (total - first) * scale, 24, GREEN))  # noqa: F405
        body.append(text(140 + total * scale + 8, y + 17, f"첫 글 {first:.0f}, 끝 {total:.0f}ms", INK, 12, "start"))  # noqa: F405
    note = 72 + 3 * 36 + 16
    body.append(text(20, note, "모델은 하나이고 한 번에 한 요청만 쓴다. 나중 요청의 첫 글은 먼저 온 요청이 다 끝나야 나온다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "queue.svg", 900, note + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_request()
    data = rows()
    if data:
        models = {model for _, model, *_ in data}
        model, name = ("llama32_1b", "Llama 3.2 1B") if "llama32_1b" in models else ("smollm2_135m", "SmolLM2-135M")
        fig_prefix(data, model, name)
        fig_queue(data, model, name)
