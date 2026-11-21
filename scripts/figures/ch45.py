"""45장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch45.py
출력: book/src/images/ch45/*.svg

whole.svg 와 split.svg 는 bench/ch45/ 에 남긴 측정값의 중앙값을 그린다.
"""

import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch45"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch45"  # noqa: F405
PURPLE, PURPLE_BG, RED = "#7c3aed", "#ede9fe", "#dc2626"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1, dash=None):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    edge += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_partition():
    """노드 하나를 못 돌리면 그래프가 셋으로 나뉜다."""
    body = [text(20, 28, "가속기가 못 돌리는 노드 하나가 그래프를 셋으로 나눈다", INK, 14, "start", "bold")]  # noqa: F405
    names = ["합성곱", "깊이별", "합성곱", "더하기", "합성곱", "깊이별", "합성곱"]
    w, h, gap, x0 = 96, 40, 26, 40

    def row(y, label, supported, groups):
        body.append(text(20, y - 12, label, INK, 12, "start", "bold"))  # noqa: F405
        for index, name in enumerate(names):
            x = x0 + index * (w + gap)
            ok = supported[index]
            fill, stroke = (GREEN_BG, GREEN) if ok else (ORANGE_BG, ORANGE)  # noqa: F405
            body.append(rect(x, y, w, h, fill, stroke, 6, 1.5))
            body.append(text(x + w / 2, y + 25, name, INK, 13))  # noqa: F405
            if index:
                body.append(line(x - gap + 2, y + h / 2, x - 3, y + h / 2, INK, 1.5, True))  # noqa: F405
        # 잔차 연결: 0번의 출력이 3번(더하기)으로.
        x_from = x0 + w / 2
        x_to = x0 + 3 * (w + gap) + w / 2
        body.append(
            f"<path d='M{x_from},{y + h} L{x_from},{y + h + 16} L{x_to},{y + h + 16} L{x_to},{y + h + 3}' "
            f"fill='none' stroke='{MUTED}' stroke-width='1.5' marker-end='url(#arrow-{MUTED[1:]})'/>"  # noqa: F405
        )
        for first, last, title, color in groups:
            gx = x0 + first * (w + gap) - 8
            gw = (last - first) * (w + gap) + w + 16
            body.append(rect(gx, y - 6, gw, h + 30, "none", color, 8, 1.5, "5,4"))
            body.append(text(gx + gw / 2, y + h + 40, title, color, 12, "middle", "bold"))  # noqa: F405

    row(70, "모든 노드를 돌릴 수 있을 때", [True] * 7, [(0, 6, "조각 하나. 드나드는 값 둘(입력, 출력)", GREEN)])  # noqa: F405
    holes = [True, True, True, False, True, True, True]
    row(
        200,
        "더하기를 못 돌릴 때",
        holes,
        [(0, 2, "넘긴 조각", GREEN), (3, 3, "CPU", ORANGE), (4, 6, "넘긴 조각", GREEN)],  # noqa: F405
    )
    body.append(text(20, 312, "첫 조각은 값 둘을 내보낸다(더하기가 읽을 두 값). 경계를 넘는 값마다 복사와 배치 바꾸기가 든다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "partition.svg", 920, 328, body)  # noqa: F405


def rows():
    out = []
    for path in sorted(BENCH.glob("round_*.txt")):
        out += [row.split() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    return out


def number(parts, key):
    return float(parts[parts.index(key) + 1])


def median(found, key):
    return statistics.median(number(parts, key) for parts in found)


RUNNERS = [
    ("badak-cpu", "badak (CPU)", MUTED),  # noqa: F405
    ("coreml-cpu", "Core ML: CPU만", BLUE),  # noqa: F405
    ("coreml-gpu", "Core ML: CPU와 GPU", ORANGE),  # noqa: F405
    ("coreml-npu", "Core ML: CPU와 뉴럴 엔진", GREEN),  # noqa: F405
]


def fig_whole(data):
    """통째로 넘겼을 때의 시간."""
    body = [text(20, 28, "그래프를 통째로 넘겼을 때: 사진 한 장을 분류하는 시간 (M4 Max)", INK, 14, "start", "bold")]  # noqa: F405
    y = 52
    for model, title in (("mobilenetv2", "MobileNetV2"), ("resnet50", "ResNet-50")):
        marks = []
        for tag, label, color in RUNNERS:
            found = [parts for parts in data if parts[:3] == ["whole", model, tag]]
            marks.append((label, color, median(found, "가운데")))
        scale = 520 / max(value for *_, value in marks)
        body.append(text(20, y + 4, title, INK, 13, "start", "bold"))  # noqa: F405
        y += 14
        for label, color, value in marks:
            body.append(text(200, y + 15, label, INK, 12, "end"))  # noqa: F405
            body.append(rect(210, y, max(value * scale, 2), 20, color, None, 3))
            body.append(text(210 + value * scale + 8, y + 15, f"{value:.2f}ms", INK, 12, "start"))  # noqa: F405
            y += 26
        y += 16
    body.append(text(20, y + 2, "Core ML의 세 줄은 같은 모델 파일이다. 어느 장치를 써도 되는지만 다르게 일렀다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "whole.svg", 860, y + 18, body)  # noqa: F405


def fig_split(data):
    """나눠서 넘겼을 때. 뉴럴 엔진."""
    body = [text(20, 28, "나눠서 넘겼을 때: 뉴럴 엔진에 넘긴 조각들과 CPU에 남은 조각들 (M4 Max)", INK, 14, "start", "bold")]  # noqa: F405
    legend = [("넘긴 조각: Core ML을 부르는 시간", GREEN), ("넘긴 조각: 경계에서 배치 바꾸기", ORANGE), ("CPU에 남은 조각", MUTED)]  # noqa: F405
    x = 20
    for label, color in legend:
        body.append(rect(x, 42, 14, 10, color))
        body.append(text(x + 20, 52, label, INK, 12, "start"))  # noqa: F405
        x += 250
    cases = [("전부", "모두 넘김"), ("ADD빼고", "더하기를 못 넘김"), ("DW빼고", "깊이별 합성곱을 못 넘김"), ("ADD빼고_큰것하나만", "더하기를 못 넘김, 큰 조각 하나만"), ("ADD빼고_노드넷이상만", "더하기를 못 넘김, 노드 넷 이상만")]
    y = 74
    for model, title in (("mobilenetv2", "MobileNetV2"), ("resnet50", "ResNet-50")):
        alone = median([parts for parts in data if parts[:3] == ["whole", model, "badak-cpu"]], "가운데")
        marks = []
        for case, label in cases:
            found = [parts for parts in data if parts[:4] == ["split", model, "coreml-npu", case]]
            if model == "resnet50" and case == "DW빼고":
                continue
            segments = found[0][found[0].index("조각") + 1]
            handed, kept, layout = median(found, "넘긴쪽"), median(found, "CPU쪽"), median(found, "배치바꾸기")
            marks.append((f"{label} (조각 {segments})", max(handed - layout, 0), layout, kept))
        longest = max(max(sum(mark[1:]) for mark in marks), alone)
        scale = 470 / longest
        body.append(text(20, y + 4, title, INK, 13, "start", "bold"))  # noqa: F405
        y += 14
        top = y
        for label, call, layout, kept in marks:
            body.append(text(290, y + 15, label, INK, 12, "end"))  # noqa: F405
            x = 300
            for value, color in ((call, GREEN), (layout, ORANGE), (kept, MUTED)):  # noqa: F405
                if value * scale >= 0.5:
                    body.append(rect(x, y, value * scale, 20, color))
                    x += value * scale
            body.append(text(x + 8, y + 15, f"{call + layout + kept:.2f}ms", INK, 12, "start"))  # noqa: F405
            y += 26
        # 이 엔진이 CPU에서 혼자 다 돌렸을 때.
        ax = 300 + alone * scale
        body.append(line(ax, top - 4, ax, y, RED, 1.5, False, "4,3"))  # noqa: F405
        body.append(text(ax + 4, top - 6, f"badak이 CPU에서 혼자 {alone:.1f}ms", RED, 11, "start"))  # noqa: F405
        y += 22
    body.append(text(20, y, "초록 막대도 조각이 늘면 길어진다. Core ML을 부를 때마다 드는 값과, 값을 가속기로 보내고 받는 값이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "split.svg", 900, y + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_partition()
    data = rows()
    if data:
        fig_whole(data)
        fig_split(data)
