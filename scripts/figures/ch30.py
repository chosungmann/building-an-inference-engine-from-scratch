"""30장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch30.py
출력: book/src/images/ch30/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch30"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


KINDS = {
    "T": ("TRANSPOSE", ORANGE_BG, ORANGE),  # noqa: F405
    "C": ("CONV_2D", BLUE_BG, BLUE),  # noqa: F405
    "B": ("BATCH_NORM", PURPLE_BG, PURPLE),
    "R": ("CLIP", GREEN_BG, GREEN),  # noqa: F405
    "W": ("필터", GRAY_BG, MUTED),  # noqa: F405
}


def chain(x, y, nodes, w=92, h=30, gap=14):
    """노드들을 한 줄로 그린다. 노드는 (종류, 글자) 또는 종류다."""
    out = []
    for index, node in enumerate(nodes):
        kind, label = node if isinstance(node, tuple) else (node, KINDS[node][0])
        _, fill, stroke = KINDS[kind]
        left = x + index * (w + gap)
        out.append(rect(left, y, w, h, fill, stroke, 5, 1.3))
        out.append(text(left + w / 2, y + h / 2 + 4, label, INK, 11))  # noqa: F405
        if index:
            out.append(line(left - gap + 1, y + h / 2, left - 2, y + h / 2, MUTED, 1.2, arrow=True))  # noqa: F405
    return out


def fig_passes():
    """합성곱·배치 정규화·활성화 함수 한 묶음이 패스를 거치며 줄어든다."""
    body = [text(20, 28, "ONNX의 세 노드가 옮겨지고 다시 하나가 되기까지", INK, 14, "start", "bold")]  # noqa: F405
    x0, step = 150, 66
    rows = [
        ("ONNX 파일", [("C", "Conv"), ("B", "BatchNorm"), ("R", "Clip")], "축의 순서는 (장 수, 채널, 높이, 너비)"),
        ("옮긴 그대로", ["T", "C", "T", "T", "B", "T", "R"], "축을 따지는 노드마다 앞뒤에 TRANSPOSE. 필터도 TRANSPOSE를 거친다"),
        ("상수 접기", ["T", "C", "T", "T", "B", "T", "R"], "필터의 TRANSPOSE를 미리 계산했다. 줄기는 그대로다"),
        ("TRANSPOSE 없애기", ["T", "C", "B", "R", "T"], "마주 본 둘은 지우고, 남은 하나는 CLIP 뒤로 옮겼다"),
        ("배치 정규화 접기", ["T", "C", "R", "T"], "배율과 편향을 필터에 곱하고 더해 두었다"),
        ("활성화 함수 붙이기", ["T", ("C", "CONV_2D+6"), "T"], "합성곱이 결과를 적으면서 0과 6 사이로 자른다"),
    ]
    y = 56
    for index, (name, nodes, note) in enumerate(rows):
        body.append(text(x0 - 16, y + 20, name, INK, 12, "end", "bold"))  # noqa: F405
        body += chain(x0, y, nodes)
        # 필터가 노드를 거쳐 들어가는 줄.
        conv = next(i for i, node in enumerate(nodes) if (node[0] if isinstance(node, tuple) else node) == "C")
        cx = x0 + conv * 106 + 46
        if index == 1:
            body.append(rect(cx - 40, y + 40, 36, 16, GRAY_BG, MUTED, 3, 1))  # noqa: F405
            body.append(text(cx - 22, y + 52, "필터", INK, 9))  # noqa: F405
            body.append(rect(cx + 2, y + 40, 38, 16, ORANGE_BG, ORANGE, 3, 1))  # noqa: F405
            body.append(text(cx + 21, y + 52, "TRANS", INK, 8))  # noqa: F405
            body.append(line(cx + 21, y + 40, cx + 10, y + 31, MUTED, 1))  # noqa: F405
        elif index >= 2 or index == 0:
            body.append(rect(cx - 18, y + 40, 36, 16, GRAY_BG, MUTED, 3, 1))  # noqa: F405
            body.append(text(cx, y + 52, "필터", INK, 9))  # noqa: F405
            body.append(line(cx, y + 40, cx, y + 31, MUTED, 1))  # noqa: F405
        body.append(text(x0 + 760, y + 20, note, MUTED, 11, "start"))  # noqa: F405
        y += step
    body.append(text(20, y + 8, "앞과 뒤에 남은 TRANSPOSE는 이웃한 묶음의 것과 다시 마주 보고 지워진다. 그래프 전체에서는 맨 앞의 하나만 남는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "passes.svg", 1330, y + 28, body)  # noqa: F405


def fig_sink():
    """TRANSPOSE를 덧셈 뒤로 옮기고 마주 본 것을 지운다."""
    body = [text(20, 28, "TRANSPOSE를 뒤로 옮겨 마주 보게 한다", INK, 14, "start", "bold")]  # noqa: F405
    w, h = 96, 30

    def node(x, y, label, kind):
        _, fill, stroke = KINDS[kind]
        return [rect(x, y, w, h, fill, stroke, 5, 1.3), text(x + w / 2, y + h / 2 + 4, label, INK, 11)]  # noqa: F405

    def arrow(x1, y1, x2, y2):
        return line(x1, y1, x2, y2, MUTED, 1.2, arrow=True)  # noqa: F405

    panels = [
        (40, "1. 옮긴 그대로", "두 갈래가 각각 NCHW로 되돌려진 뒤 더해지고,|다음 합성곱 앞에서 다시 NHWC가 된다."),
        (470, "2. 뒤로 옮기기", "덧셈은 원소가 놓인 순서를 따지지 않는다.|두 입력이 같은 순서로 바뀌어 있으면|바꾸기 전의 값을 더하고 나서 바꿔도 같다."),
        (900, "3. 상쇄", "→NCHW 다음의 →NHWC는 제자리다.|둘 다 지운다."),
    ]
    for x, title, note in panels:
        body.append(text(x, 62, title, INK, 13, "start", "bold"))  # noqa: F405
        for index, part in enumerate(note.split("|")):
            body.append(text(x, 346 + index * 17, part, MUTED, 11.5, "start"))  # noqa: F405
    # 1
    x = 40
    body += node(x, 84, "CONV_2D", "C") + node(x + 130, 84, "CONV_2D", "C")
    body += node(x, 136, "→NCHW", "T") + node(x + 130, 136, "→NCHW", "T")
    body += [arrow(x + 48, 114, x + 48, 134), arrow(x + 178, 114, x + 178, 134)]
    body += node(x + 65, 190, "ADD", "R")
    body += [arrow(x + 48, 166, x + 100, 188), arrow(x + 178, 166, x + 126, 188)]
    body += node(x + 65, 240, "→NHWC", "T") + node(x + 65, 288, "CONV_2D", "C")
    body += [arrow(x + 113, 220, x + 113, 238), arrow(x + 113, 270, x + 113, 286)]
    # 2
    x = 470
    body += node(x, 84, "CONV_2D", "C") + node(x + 130, 84, "CONV_2D", "C")
    body += node(x + 65, 136, "ADD", "R")
    body += [arrow(x + 48, 114, x + 100, 134), arrow(x + 178, 114, x + 126, 134)]
    body += node(x + 65, 190, "→NCHW", "T") + node(x + 65, 240, "→NHWC", "T") + node(x + 65, 288, "CONV_2D", "C")
    body += [arrow(x + 113, 166, x + 113, 188), arrow(x + 113, 220, x + 113, 238), arrow(x + 113, 270, x + 113, 286)]
    body.append(rect(x + 55, 182, 116, 96, "none", ORANGE, 8, 1.5))  # noqa: F405
    # 3
    x = 900
    body += node(x, 84, "CONV_2D", "C") + node(x + 130, 84, "CONV_2D", "C")
    body += node(x + 65, 136, "ADD", "R")
    body += [arrow(x + 48, 114, x + 100, 134), arrow(x + 178, 114, x + 126, 134)]
    body += node(x + 65, 190, "CONV_2D", "C")
    body.append(arrow(x + 113, 166, x + 113, 188))
    svg(OUT / "sink.svg", 1180, 404, body)  # noqa: F405


def bench():
    """bench/ch30/badak_*.txt 에서 단계마다의 중앙값(세 차례의 중앙값)을 읽는다."""
    import re
    import statistics

    pattern = re.compile(r"(\w+) (\w+) 스레드\s+(\d+) 단계 (\d) .*중앙값\s+([\d.]+)ms")
    samples = {}
    for path in sorted((IMAGES.parents[2] / "bench/ch30").glob("badak_*.txt")):  # noqa: F405
        for row in path.read_text(encoding="utf-8").splitlines():
            found = pattern.match(row)
            key = (found[1], found[2], int(found[3]), int(found[4]))
            samples.setdefault(key, []).append(float(found[5]))
    return {key: statistics.median(values) for key, values in samples.items()}


def fig_stages():
    """패스를 하나씩 더해 갈 때 사진 한 장의 시간."""
    times = bench()
    stages = ["옮긴 그대로", "+ 상수 접기", "+ TRANSPOSE 없애기", "+ 배치 정규화 접기", "+ 활성화 함수 붙이기", "+ 제자리 계산"]
    body = [text(20, 28, "패스를 하나씩 더해 갈 때 사진 한 장의 시간 (ONNX에서 옮긴 그래프, 밀리초)", INK, 14, "start", "bold")]  # noqa: F405
    panels = [("resnet50", "ResNet-50", 12), ("mobilenetv2", "MobileNetV2", 12), ("resnet50", "ResNet-50", 1), ("mobilenetv2", "MobileNetV2", 1)]
    label_w, bar_w, row_h = 170, 330, 24
    for index, (model, title, threads) in enumerate(panels):
        x = 20 + (index % 2) * 580
        y = 58 + (index // 2) * 210
        body.append(text(x, y, f"{title}, 스레드 {threads}개", INK, 13, "start", "bold"))  # noqa: F405
        values = [times[("onnx", model, threads, stage)] for stage in range(6)]
        top = max(values)
        for stage, value in enumerate(values):
            row = y + 14 + stage * row_h
            body.append(text(x + label_w - 8, row + 15, stages[stage], INK, 11.5, "end"))  # noqa: F405
            # 막대의 색은 한 가지다. 길이가 값이다. 막대 사이에 2픽셀을 띄운다.
            body.append(rect(x + label_w, row + 3, max(2, value / top * bar_w), row_h - 6, BLUE, None, 2))  # noqa: F405
            digits = 1 if value >= 20 else 2
            body.append(text(x + label_w + value / top * bar_w + 8, row + 15, f"{value:.{digits}f}", INK, 11.5, "start"))  # noqa: F405
        body.append(line(x + label_w, y + 12, x + label_w, y + 16 + 6 * row_h, MUTED, 1))  # noqa: F405
    note = 58 + 2 * 210 + 6
    body.append(text(20, note, "세 차례(MobileNetV2의 스레드 12개는 일곱 차례) 잰 중앙값. 스레드 12개에서는 TRANSPOSE를 없애는 단계가 대부분을 줄인다. 스레드 1개에서는 합성곱이 시간을 정해서 차이가 작다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "stages.svg", 1180, note + 18, body)  # noqa: F405


if __name__ == "__main__":
    fig_passes()
    fig_sink()
    fig_stages()
