"""34장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch34.py
출력: book/src/images/ch34/*.svg

throughput.svg 와 latency.svg 는 bench/ch34/ 에 남긴 측정값의 중앙값을
그린다. 그 파일이 없으면 두 그림은 건너뛴다.
"""

import math
import re
import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch34"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch34"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_pipeline():
    """두 가지 부하가 같은 모델을 다르게 쓴다."""
    body = [text(20, 28, "임베딩 모델을 쓰는 두 가지 일", INK, 14, "start", "bold")]  # noqa: F405
    w, h, gap = 128, 50, 24
    rows = [
        (62, "대량 처리", "글을 쌓아 둘 때", BLUE, BLUE_BG, ["글 수천~수백만 개", "토크나이저", "BERT", "평균, 길이 1", "색인"], ["", "", "배치로", "", "벡터들을 쌓는다"]),  # noqa: F405
        (182, "저지연 조회", "찾을 때", ORANGE, ORANGE_BG, ["질의 하나", "토크나이저", "BERT", "평균, 길이 1", "내적, 상위 k개"], ["", "", "하나만", "", "모든 벡터와"]),  # noqa: F405
    ]
    for y, name, when, color, fill, titles, subs in rows:
        body.append(text(20, y + 22, name, color, 13, "start", "bold"))  # noqa: F405
        body.append(text(20, y + 40, when, MUTED, 12, "start"))  # noqa: F405
        for index, (title, sub) in enumerate(zip(titles, subs)):
            x = 118 + index * (w + gap)
            body += box(x, y, w, h, title, sub or None, fill, color)  # noqa: F405
            if index:
                body.append(line(x - gap + 2, y + h / 2, x - 3, y + h / 2, color, 1.5, True))  # noqa: F405
    x_index = 118 + 4 * (w + gap) + w / 2
    body.append(line(x_index, 62 + h + 2, x_index, 182 - 4, MUTED, 1.5, True, "5,4"))  # noqa: F405
    body.append(text(x_index + 8, 150, "찾을 곳", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, 268, "대량 처리에서는 초당 몇 글인지(처리량)가, 조회에서는 질의 하나에 몇 밀리초인지(지연 시간)가 중요하다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, 286, "모델과 토크나이저는 같다. 묶는 방법과 스레드를 쓰는 곳이 달라진다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 900, 304, body)  # noqa: F405


def fig_wordpiece():
    """글 하나가 토큰 번호가 되기까지."""
    body = [text(20, 28, "WordPiece: 글이 토큰 번호가 되기까지", INK, 14, "start", "bold")]  # noqa: F405
    x0, h = 150, 34

    def row(y, label, cells, fill, stroke, note=None):
        body.append(text(20, y + 22, label, INK, 13, "start", "bold"))  # noqa: F405
        x = x0
        for label_text, width in cells:
            body.append(rect(x, y, width, h, fill, stroke, 4))
            body.append(text(x + width / 2, y + 22, label_text, INK, 13))  # noqa: F405
            x += width + 6
        if note:
            body.append(text(x + 8, y + 22, note, MUTED, 12, "start"))  # noqa: F405

    row(48, "글", [("Crème brûlée.", 150)], GRAY_BG, MUTED, "받은 그대로")  # noqa: F405
    row(100, "고치기", [("creme brulee.", 150)], GRAY_BG, MUTED, "소문자로, 악센트를 지운다")  # noqa: F405
    row(152, "단어", [("creme", 70), ("brulee", 76), (".", 30)], BLUE_BG, BLUE, "공백과 문장 부호에서 자른다")  # noqa: F405
    row(204, "조각", [("cr", 40), ("##eme", 62), ("br", 40), ("##ule", 56), ("##e", 44), (".", 30)], ORANGE_BG, ORANGE, "앞에서부터 가장 긴 조각")  # noqa: F405
    ids = [("101", 46), ("13675", 56), ("21382", 56), ("7987", 50), ("9307", 50), ("2063", 50), ("1012", 50), ("102", 46)]
    row(256, "번호", ids, GREEN_BG, GREEN)  # noqa: F405
    body.append(text(x0 + 23, 306, "[CLS]", MUTED, 11))  # noqa: F405
    body.append(text(x0 + 46 + 6 * 6 + 56 * 2 + 50 * 4 + 23, 306, "[SEP]", MUTED, 11))  # noqa: F405
    for y in (48, 100, 152, 204):
        body.append(line(x0 + 20, y + h + 2, x0 + 20, y + 50, MUTED, 1.5, True))  # noqa: F405
    body.append(text(20, 332, "단어의 첫 조각이 아닌 조각은 ##로 시작한다. 공백은 토큰이 되지 않는다. 맨 앞과 끝에 토큰 둘이 붙는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "wordpiece.svg", 760, 350, body)  # noqa: F405


def fig_batching():
    """길이가 다른 글 셋을 한 배치로 만드는 두 가지 방법."""
    body = [text(20, 28, "길이가 3, 5, 2인 글 셋을 한 배치로", INK, 14, "start", "bold")]  # noqa: F405
    lengths = [3, 5, 2]
    colors = [(BLUE, BLUE_BG), (ORANGE, ORANGE_BG), (GREEN, GREEN_BG)]  # noqa: F405
    s = 34
    # 왼쪽: 채우기.
    x, y = 40, 86
    body.append(text(x, y - 22, "채우기 (padded)", INK, 13, "start", "bold"))  # noqa: F405
    for r, length in enumerate(lengths):
        for c in range(5):
            real = c < length
            stroke, fill = colors[r] if real else (MUTED, "#ffffff")  # noqa: F405
            body.append(rect(x + c * s, y + r * (s + 6), s - 4, s - 4, fill, stroke, 3))
            if not real:
                body.append(text(x + c * s + (s - 4) / 2, y + r * (s + 6) + 19, "P", MUTED, 11))  # noqa: F405
        body.append(text(x - 10, y + r * (s + 6) + 20, f"글 {r}", MUTED, 12, "end"))  # noqa: F405
    body.append(text(x, y + 3 * (s + 6) + 16, "(3, 5, 폭)의 반듯한 텐서. 15줄.", INK, 12, "start"))  # noqa: F405
    body.append(text(x, y + 3 * (s + 6) + 34, "P는 [PAD]. 5줄이 버리는 계산이다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(x, y + 3 * (s + 6) + 52, "어텐션에서 P의 열을 가린다.", MUTED, 12, "start"))  # noqa: F405
    # 오른쪽: 이어 붙이기.
    x, y = 360, 86
    body.append(text(x, y - 22, "이어 붙이기 (ragged)", INK, 13, "start", "bold"))  # noqa: F405
    start = 0
    for r, length in enumerate(lengths):
        stroke, fill = colors[r]
        for c in range(length):
            body.append(rect(x + (start + c) * s, y, s - 4, s - 4, fill, stroke, 3))
        left, right = x + start * s, x + (start + length) * s - 4
        body.append(line(left, y + s + 6, right, y + s + 6, stroke, 1.5))  # noqa: F405
        body.append(text((left + right) / 2, y + s + 24, f"글 {r}: {start}..{start + length}", stroke, 12))  # noqa: F405
        start += length
    body.append(text(x, y + 3 * (s + 6) + 16, "(10, 폭)의 텐서와 글마다의 경계. 10줄.", INK, 12, "start"))  # noqa: F405
    body.append(text(x, y + 3 * (s + 6) + 34, "선형 층은 경계를 모른 채 10줄을 계산한다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(x, y + 3 * (s + 6) + 52, "어텐션만 글마다 따로 한다. 가릴 것이 없다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "batching.svg", 760, 276, body)  # noqa: F405


def medians():
    """bench/ch34/ 의 줄들을 (종류, 이름, 설정) → 중앙값으로 모은다."""
    values = {}
    for path in sorted(BENCH.glob("badak_*.txt")) + sorted(BENCH.glob("python_*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"latency (.+?) +스레드 +(\d+) 토큰 +(\d+) 중앙값 +([\d.]+)ms", row)
            if found:
                key = ("latency", found[1], int(found[2]), int(found[3]))
                values.setdefault(key, []).append(float(found[4]))
            found = re.match(r"bulk (.+?) +스레드 +(\d+) 배치 +(\d+) (\w+) +(\w+) +(?:(\w+) +)?줄/토큰 [\d.]+ 초당 +([\d.]+)글", row)
            if found:
                key = ("bulk", found[1], int(found[2]), int(found[3]), found[4], found[5], found[6] or "")
                values.setdefault(key, []).append(float(found[7]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def fig_throughput(values):
    """많은 글의 처리량. 엔진마다 가장 좋은 설정."""
    body = [text(20, 28, "글 4,096개를 임베딩으로 바꾸는 처리량 (초당 글의 수, 클수록 좋다)", INK, 14, "start", "bold")]  # noqa: F405
    label_w, bar_w, row_h, panel_w = 250, 300, 26, 640
    for index, threads in enumerate((1, 12)):
        where = "inside" if threads == 1 else "across"
        rows = [
            ("ONNX Runtime, 채우고 길이순", values[("bulk", "ONNX Runtime", threads, 32, "padded", "sorted", "")], MUTED),  # noqa: F405
            ("PyTorch, 채우고 길이순", values[("bulk", "PyTorch", threads, 32, "padded", "sorted", "")], MUTED),  # noqa: F405
            ("badak 기본, 채우기", values[("bulk", "기본", threads, 32, "padded", "given", where)], BLUE),  # noqa: F405
            ("badak 기본, 이어 붙이기", values[("bulk", "기본", threads, 32, "ragged", "given", where)], BLUE),  # noqa: F405
            ("badak 빠른 길, 이어 붙이기", values[("bulk", "빠른 길", threads, 32, "ragged", "sorted", where)], ORANGE),  # noqa: F405
        ]
        x, y = 20 + index * panel_w, 78
        body.append(text(x, y - 12, f"스레드 {threads}개", INK, 13, "start", "bold"))  # noqa: F405
        top = max(value for _, value, _ in rows)
        for rank, (label, value, color) in enumerate(rows):
            row = y + rank * row_h
            body.append(text(x + label_w - 8, row + 17, label, INK, 12, "end"))  # noqa: F405
            body.append(rect(x + label_w, row + 4, max(2, value / top * bar_w), row_h - 8, color, None, 3))
            body.append(text(x + label_w + value / top * bar_w + 6, row + 17, f"{value:,.0f}", INK, 12, "start"))  # noqa: F405
        body.append(line(x + label_w, y, x + label_w, y + len(rows) * row_h, MUTED, 1))  # noqa: F405
    note = 78 + 5 * row_h + 24
    body.append(text(20, note, "배치의 크기는 모두 32. 막대의 길이는 판 안에서만 견줄 수 있다. 회색은 다른 엔진, 파랑은 NEON 커널, 주황은 SME 커널이다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, note + 18, "스레드 열둘에서 badak은 배치들을 스레드에 나눴다. 토큰으로 바꾸는 시간은 어느 쪽에도 넣지 않았다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "throughput.svg", 1280, note + 34, body)  # noqa: F405


def fig_latency(values):
    """글 하나의 시간이 길이에 따라 늘어나는 모양."""
    body = [text(20, 28, "글 하나를 임베딩으로 바꾸는 시간 (스레드 1개, 두 축 모두 로그)", INK, 14, "start", "bold")]  # noqa: F405
    lengths = [8, 16, 32, 64, 128, 256]
    series = [
        ("badak 기본 (NEON)", "기본", BLUE),  # noqa: F405
        ("ONNX Runtime", "ONNX Runtime", MUTED),  # noqa: F405
        ("PyTorch", "PyTorch", GREEN),  # noqa: F405
        ("badak 빠른 길 (SME)", "빠른 길", ORANGE),  # noqa: F405
    ]
    left, top, width, height = 70, 56, 470, 280
    low, high = 0.25, 128.0

    def px(length):
        return left + (math.log2(length) - 3) / 5 * width

    def py(value):
        return top + height - (math.log2(value) - math.log2(low)) / (math.log2(high) - math.log2(low)) * height

    tick = low
    while tick <= high:
        body.append(line(left, py(tick), left + width, py(tick), "#e5e7eb", 1))  # noqa: F405
        label = f"{tick:g}"
        body.append(text(left - 8, py(tick) + 4, label, MUTED, 11, "end"))  # noqa: F405
        tick *= 2
    for length in lengths:
        body.append(text(px(length), top + height + 18, str(length), MUTED, 11))  # noqa: F405
    body.append(text(left + width / 2, top + height + 38, "토큰의 수", MUTED, 12))  # noqa: F405
    body.append(text(left - 44, top - 10, "밀리초", MUTED, 12, "start"))  # noqa: F405
    body.append(line(left, top + height, left + width, top + height, MUTED, 1))  # noqa: F405
    ends = []
    for label, name, color in series:
        points = [(px(length), py(values[("latency", name, 1, length)])) for length in lengths]
        path = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))
        body.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in points:
            body.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}' stroke='#ffffff' stroke-width='1.5'/>")
        ends.append([points[-1][1], label, color, values[("latency", name, 1, 256)]])
    # 끝점의 이름표가 겹치지 않게 위에서부터 16픽셀씩 벌린다.
    ends.sort()
    for index in range(1, len(ends)):
        ends[index][0] = max(ends[index][0], ends[index - 1][0] + 16)
    for y, label, color, value in ends:
        body.append(rect(left + width + 14, y - 5, 10, 10, color, None, 2))
        body.append(text(left + width + 30, y + 4, f"{label}  {value:.1f}ms", INK, 12, "start"))  # noqa: F405
    body.append(text(20, top + height + 62, "토큰이 두 배가 되면 선형 층의 일은 두 배, 어텐션의 일은 네 배가 된다. 긴 글에서 선이 가팔라진다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "latency.svg", 780, top + height + 78, body)  # noqa: F405


if __name__ == "__main__":
    fig_pipeline()
    fig_wordpiece()
    fig_batching()
    if list(BENCH.glob("badak_*.txt")) and list(BENCH.glob("python_*.txt")):
        values = medians()
        fig_throughput(values)
        fig_latency(values)
