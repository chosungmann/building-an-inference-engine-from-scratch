"""26장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch26.py
출력: book/src/images/ch26/*.svg

수명과 자리의 그림은 book/src/part4/output/ch26_plan_residual.txt 를 읽어
그린다. 그 파일은 `ch26_plan residual 1`의 출력이다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch26"  # noqa: F405
PLAN = IMAGES.parent / "part4/output/ch26_plan_residual.txt"  # noqa: F405
GRID = "#e5e7eb"
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=4, width=1.5):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def path(d, stroke=INK, width=1.5, arrow=False, dash=None):  # noqa: F405
    extra = f" marker-end='url(#arrow-{stroke[1:]})'" if arrow else ""
    extra += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}'{extra}/>"


def fig_graph():
    """잔차 모델의 그래프. 노드 열두 개와 그 사이를 흐르는 값들."""
    nodes = [("FC", "relu")]
    for _ in range(3):
        nodes += [("FC", "relu"), ("FC", ""), ("ADD", "relu")]
    nodes += [("FC", ""), ("SOFTMAX", "")]
    sizes = [784, 256, 512, 256, 256, 512, 256, 256, 512, 256, 256, 10, 10]
    w, h, gap, x0, y0 = 46, 44, 28, 50, 96
    body = [text(20, 28, "residual.tflite의 그래프: 노드(연산) 열두 개와 값 열세 개", INK, 14, "start", "bold")]  # noqa: F405
    xs = [x0 + index * (w + gap) for index in range(len(nodes))]
    for index, (name, fused) in enumerate(nodes):
        x = xs[index]
        fill, stroke = (ORANGE_BG, ORANGE) if name == "ADD" else ((GREEN_BG, GREEN) if name == "SOFTMAX" else (BLUE_BG, BLUE))  # noqa: F405
        body.append(rect(x, y0, w, h, fill, stroke, 6))
        label = "SOFT" if name == "SOFTMAX" else name
        body.append(text(x + w / 2, y0 + (20 if fused or name == "SOFTMAX" else 27), label, INK, 12, "middle", "bold"))  # noqa: F405
        if name == "SOFTMAX":
            body.append(text(x + w / 2, y0 + 35, "MAX", INK, 12, "middle", "bold"))  # noqa: F405
        if fused:
            body.append(text(x + w / 2, y0 + 36, f"+{fused}", MUTED, 11))  # noqa: F405
        body.append(text(x + w / 2, y0 + h + 16, str(index), MUTED, 10))  # noqa: F405
    # 값: 노드 사이의 화살표와 그 위의 크기.
    for index, size in enumerate(sizes):
        left = xs[index - 1] + w if index > 0 else x0 - 28
        right = xs[index] if index < len(nodes) else xs[-1] + w + 28
        y = y0 + h / 2
        body.append(line(left, y, right - 1, y, INK, 1.5, arrow=True))  # noqa: F405
        cx = (left + right) / 2
        body.append(text(cx, y + 15, f"t{index}", MUTED, 10))  # noqa: F405
        body.append(text(cx, y + 27, str(size), MUTED, 10))  # noqa: F405
    # 건너뛰는 길: ADD는 블록의 입력도 받는다.
    for block in range(3):
        source = xs[3 * block] + w + gap / 2
        target = xs[3 * block + 3] + w / 2
        top = y0 - 30
        body.append(path(f"M{source:.1f},{y0 + h / 2 - 2} L{source:.1f},{top} L{target:.1f},{top} L{target:.1f},{y0 - 2}", ORANGE, 1.5, arrow=True))  # noqa: F405
    body.append(text((xs[0] + w + xs[3]) / 2 + 20, y0 - 38, "건너뛰는 길: t1은 3번 노드까지 살아 있어야 한다", ORANGE, 11))  # noqa: F405
    body.append(text(20, y0 + h + 44, "화살표 하나가 값(텐서) 하나다. 이름 아래 숫자는 사진 한 장일 때의 원소 수, 노드 아래 숫자는 실행 순서.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y0 + h + 62, "가중치와 편향도 값이지만 파일에 들어 있어 계산할 필요가 없다. 그림에서는 뺐다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "graph.svg", 960, y0 + h + 78, body)  # noqa: F405


def fig_flatbuffers():
    """FlatBuffers의 표 하나가 바이트로 놓인 모양."""
    body = [text(20, 28, "FlatBuffers의 표: 필드를 찾으려면 vtable을 거친다", INK, 14, "start", "bold")]  # noqa: F405
    y, h = 134, 40
    # (너비, 글, 색, 아래 설명)
    cells = [
        (70, "20", GRAY_BG, "뿌리 표까지"),  # noqa: F405
        (70, "TFL3", GRAY_BG, "식별자"),  # noqa: F405
        (24, "", "#ffffff", ""),
        (56, "10", PURPLE_BG, "vtable 길이"),
        (56, "12", PURPLE_BG, "표 길이"),
        (56, "4", PURPLE_BG, "필드 0"),
        (56, "0", PURPLE_BG, "필드 1"),
        (56, "8", PURPLE_BG, "필드 2"),
        (30, "", "#ffffff", ""),
        (70, "12", BLUE_BG, "vtable까지"),  # noqa: F405
        (70, "3", BLUE_BG, "필드 0의 값"),  # noqa: F405
        (70, "16", BLUE_BG, "필드 2: 오프셋"),  # noqa: F405
        (40, "…", "#ffffff", ""),
        (56, "2", GREEN_BG, "원소 수"),  # noqa: F405
        (56, "784", GREEN_BG, "원소 0"),  # noqa: F405
        (56, "256", GREEN_BG, "원소 1"),  # noqa: F405
    ]
    x = 20
    starts = []
    for width, label, fill, note in cells:
        starts.append(x)
        if label not in ("…", ""):
            body.append(rect(x, y, width, h, fill, MUTED, 0, 1))  # noqa: F405
            body.append(text(x + width / 2, y + 25, label, INK, 13))  # noqa: F405
            body.append(text(x + width / 2, y + h + 15, note, MUTED, 10))  # noqa: F405
        elif label == "…":
            body.append(text(x + width / 2, y + 25, "…", MUTED, 14))  # noqa: F405
        x += width
    # 묶음의 이름.
    body.append(text(starts[3] + 140, y - 10, "vtable (16비트 값들)", PURPLE, 12, "middle", "bold"))
    body.append(text(starts[9] + 105, y - 10, "표 (필드의 값들)", BLUE, 12, "middle", "bold"))  # noqa: F405
    body.append(text(starts[13] + 84, y - 10, "벡터", GREEN, 12, "middle", "bold"))  # noqa: F405
    # 화살표: 뿌리 → 표, 표 → vtable(뒤로), vtable 필드 → 표 안의 자리, 표 필드 → 벡터.
    high, higher = y - 34, y - 62
    body.append(path(f"M{starts[0] + 35},{y - 2} L{starts[0] + 35},{higher} L{starts[9] + 14},{higher} L{starts[9] + 14},{y - 2}", INK, 1.5, arrow=True))  # noqa: F405
    body.append(text(starts[0] + 45, higher - 8, "① 파일의 첫 값을 따라가면 뿌리 표", INK, 11, "start"))  # noqa: F405
    body.append(path(f"M{starts[9] + 40},{y - 2} L{starts[9] + 40},{high} L{starts[3] + 10},{high} L{starts[3] + 10},{y - 2}", PURPLE, 1.5, arrow=True))
    body.append(text(starts[3] + 20, high - 8, "② 표의 첫 값을 표의 위치에서 빼면 vtable", PURPLE, 11, "start"))
    low = y + h + 26
    body.append(path(f"M{starts[7] + 28},{low} L{starts[7] + 28},{low + 16} L{starts[11] + 20},{low + 16} L{starts[11] + 20},{low + 2}", PURPLE, 1.5, arrow=True))
    body.append(text(starts[7] + 28, low + 34, "③ vtable의 칸: 그 필드가 표의 몇 번째 바이트에 있나 (0이면 없는 필드)", PURPLE, 11, "start"))
    body.append(path(f"M{starts[11] + 50},{low} L{starts[11] + 50},{low + 16} L{starts[13] + 10},{low + 16} L{starts[13] + 10},{low + 2}", GREEN, 1.5, arrow=True))  # noqa: F405
    body.append(text(starts[13] + 20, low + 34, "④ 오프셋을 더하면 벡터", GREEN, 11, "start"))  # noqa: F405
    svg(OUT / "flatbuffers.svg", 930, low + 50, body)  # noqa: F405


def fig_pipeline():
    """파일에서 실행까지. 한 번만 하는 일과 추론마다 하는 일."""
    body = [text(20, 28, "파일에서 실행까지: 계획은 한 번, 실행은 여러 번", INK, 14, "start", "bold")]  # noqa: F405
    y, h = 60, 62
    stages = [
        (20, 120, ".tflite 파일", "메모리에 비춘다", GRAY_BG, MUTED),  # noqa: F405
        (172, 150, "tflite::Model", "FlatBuffers를 따라간다", PURPLE_BG, PURPLE),
        (354, 130, "Graph", "값과 노드", BLUE_BG, BLUE),  # noqa: F405
    ]
    for x, w, title, sub, fill, stroke in stages:
        body += box(x, y, w, h, title, sub, fill, stroke)  # noqa: F405
    body.append(line(140, y + h / 2, 170, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    body.append(line(322, y + h / 2, 352, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    body.append(line(484, y + h / 2, 514, y + h / 2, INK, 1.5, arrow=True))  # noqa: F405
    # 해석기를 만드는 다섯 단계.
    x0, w0 = 516, 394
    body.append(rect(x0, y - 16, w0, 196, "#ffffff", ORANGE, 8))  # noqa: F405
    body.append(text(x0 + 14, y + 4, "Interpreter::with (계획)", ORANGE, 12, "start", "bold"))  # noqa: F405
    steps = [
        ("1", "순서를 정하고 커널을 찾는다", "Graph::order, Registry::resolve"),
        ("2", "값마다의 형상을 정한다", "Kernel::shapes"),
        ("3", "값마다의 수명을 읽는다", "memory::lifetimes"),
        ("4", "자리를 나누고 확인한다", "memory::plan, memory::check"),
        ("5", "단계마다 읽고 쓰는 자리를 확인한다", "check_steps"),
    ]
    for index, (number, what, where) in enumerate(steps):
        yy = y + 16 + index * 32
        body.append(rect(x0 + 12, yy, 22, 22, ORANGE_BG, ORANGE, 11, 1))  # noqa: F405
        body.append(text(x0 + 23, yy + 16, number, INK, 12))  # noqa: F405
        body.append(text(x0 + 44, yy + 16, what, INK, 12, "start"))  # noqa: F405
        body.append(text(x0 + w0 - 12, yy + 16, where, MUTED, 10, "end"))  # noqa: F405
    # 실행.
    ry = y + 206
    body += box(x0, ry, w0, 54, "Interpreter::run (실행)", "단계마다 커널을 부른다. 메모리를 얻지 않는다", GREEN_BG, GREEN)  # noqa: F405
    body.append(line(x0 + w0 / 2, y + 180, x0 + w0 / 2, ry - 2, INK, 1.5, arrow=True))  # noqa: F405
    body.append(text(20, ry + 20, "왼쪽부터 계획까지는 모델을 열 때 한 번 한다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, ry + 38, "실행은 사진이 들어올 때마다 한다. 정할 것이 남아 있지 않다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 930, ry + 70, body)  # noqa: F405


def read_plan():
    """`ch26_plan`의 출력에서 값마다의 (크기, 처음, 끝, 자리)와 합계들을 읽는다."""
    values, totals = [], {}
    for row in PLAN.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s+t(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$", row)
        if match:
            values.append(tuple(int(group) for group in match.groups()[1:]))
        match = re.match(r"\s+(값마다 따로|순서대로 놓기|큰 것부터 놓기|이보다 작을 수 없는 크기)\s+(\d+)\s+\d+$", row)
        if match:
            totals[match.group(1)] = int(match.group(2))
    return values, totals


def place(values, order):
    """crates/badak_graph/src/memory.rs의 `place`와 같은 일을 한다."""
    offsets, placed = [0] * len(values), []
    for index in order:
        size, first, last, _ = values[index]
        best, end = None, 0
        for other in placed:
            other_size, other_first, other_last, _ = values[other]
            if not (first <= other_last and other_first <= last):
                continue
            start = offsets[other]
            if end + size <= start and (best is None or start - end < best[1]):
                best = (end, start - end)
            end = max(end, start + other_size)
        offsets[index] = best[0] if best else end
        placed.append(index)
        placed.sort(key=lambda other: offsets[other])
    return offsets


def shade(size):
    return {784: (PURPLE_BG, PURPLE), 512: (BLUE_BG, BLUE), 256: (ORANGE_BG, ORANGE)}.get(size, (GREEN_BG, GREEN))  # noqa: F405


def fig_lifetimes(values):
    """값마다의 수명. 가로가 단계다."""
    steps = max(last for _, _, last, _ in values) + 1
    x0, y0, cw, rh = 150, 88, 54, 22
    body = [text(20, 28, "값마다의 수명 (residual.tflite, 사진 한 장)", INK, 14, "start", "bold")]  # noqa: F405
    names = ["FC", "FC", "FC", "ADD", "FC", "FC", "ADD", "FC", "FC", "ADD", "FC", "SOFTMAX", "끝난 뒤"]
    for step in range(steps):
        x = x0 + step * cw
        body.append(rect(x, y0 - 4, cw, len(values) * rh + 8, "#ffffff" if step % 2 else "#f9fafb", None, 0))
        body.append(text(x + cw / 2, y0 - 26, str(step) if step < steps - 1 else "", MUTED, 11))  # noqa: F405
        body.append(text(x + cw / 2, y0 - 12, names[step], MUTED, 9))  # noqa: F405
    for index, (size, first, last, _) in enumerate(values):
        y = y0 + index * rh
        fill, stroke = shade(size)
        body.append(text(x0 - 62, y + 15, f"t{index}", INK, 12, "end"))  # noqa: F405
        body.append(text(x0 - 12, y + 15, f"{size}", MUTED, 11, "end"))  # noqa: F405
        body.append(rect(x0 + first * cw + 3, y + 3, (last - first + 1) * cw - 6, rh - 6, fill, stroke, 4, 1.2))
    body.append(text(x0 - 62, y0 - 12, "값", MUTED, 10, "end"))  # noqa: F405
    body.append(text(x0 - 12, y0 - 12, "크기", MUTED, 10, "end"))  # noqa: F405
    y = y0 + len(values) * rh + 26
    body.append(text(20, y, "막대는 값이 만들어지는 단계에서 마지막으로 읽히는 단계까지다. 입력(t0)은 시작할 때 이미 있고,", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 18, "출력(t12)은 실행이 끝난 뒤에도 읽을 수 있어야 한다. 한 열에 걸친 막대들이 그 단계에 함께 살아 있다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "lifetimes.svg", x0 + steps * cw + 30, y + 34, body)  # noqa: F405


def fig_layout(values, totals):
    """세 가지 방법이 나눈 자리. 가로가 단계, 세로가 자리다."""
    steps = max(last for _, _, last, _ in values) + 1
    by_size = [offset for _, _, _, offset in values]
    in_order = place(values, sorted(range(len(values)), key=lambda index: values[index][1]))
    assert max(o + v[0] for o, v in zip(by_size, values)) == totals["큰 것부터 놓기"]
    assert max(o + v[0] for o, v in zip(in_order, values)) == totals["순서대로 놓기"]
    assert sum(v[0] for v in values) == totals["값마다 따로"]
    bound = totals["이보다 작을 수 없는 크기"]

    body = [text(20, 28, "같은 값들을 세 가지 방법으로 놓은 결과 (세로가 자리, 눈금은 f32의 수)", INK, 14, "start", "bold")]  # noqa: F405
    y0, scale = 70, 0.2
    height = 1400 * scale

    # 왼쪽: 값마다 따로. 한 줄로 쌓는다. 스케일이 다르다.
    column_scale = height / totals["값마다 따로"]
    body.append(text(64, y0 - 12, "값마다 따로", INK, 12, "middle", "bold"))  # noqa: F405
    offset = 0
    for size, *_ in values:
        fill, stroke = shade(size)
        body.append(rect(40, y0 + offset * column_scale, 48, max(size * column_scale - 1, 1), fill, stroke, 0, 0.8))
        offset += size
    body.append(text(64, y0 + height + 18, f"{totals['값마다 따로']:,}", INK, 12))  # noqa: F405
    body.append(text(64, y0 + height + 34, "(스케일이 다름)", MUTED, 10))  # noqa: F405

    for panel, (title, offsets, total) in enumerate(
        [("순서대로 놓기", in_order, totals["순서대로 놓기"]), ("큰 것부터 놓기 (LiteRT의 방법)", by_size, totals["큰 것부터 놓기"])]
    ):
        x0, cw = 200 + panel * 370, 24
        body.append(text(x0, y0 - 12, title, INK, 12, "start", "bold"))  # noqa: F405
        body.append(rect(x0, y0, steps * cw, height, "#f9fafb", GRID, 0, 1))
        for tick in (0, 512, 1024):
            y = y0 + tick * scale
            body.append(line(x0 - 4, y, x0, y, MUTED, 1))  # noqa: F405
            body.append(text(x0 - 8, y + 4, f"{tick:,}", MUTED, 10, "end"))  # noqa: F405
        for index, (size, first, last, _) in enumerate(values):
            fill, stroke = shade(size)
            x, y = x0 + first * cw, y0 + offsets[index] * scale
            w, h = (last - first + 1) * cw, size * scale
            body.append(rect(x + 0.5, y + 0.5, w - 1, max(h - 1, 1.5), fill, stroke, 0, 1))
            if h >= 14:
                body.append(text(x + w / 2, y + h / 2 + 4, f"t{index}", INK, 10))  # noqa: F405
        y = y0 + bound * scale
        body.append(line(x0, y, x0 + steps * cw, y, GREEN, 1.5, dash="5 4"))  # noqa: F405
        y = y0 + total * scale
        body.append(line(x0, y, x0 + steps * cw, y, INK, 1.5))  # noqa: F405
        body.append(text(x0 + steps * cw - 4, y + 16, f"전체 {total:,}", INK, 12, "end", "bold"))  # noqa: F405
        for step in range(0, steps, 2):
            body.append(text(x0 + step * cw + cw / 2, y0 + height + 14, str(step), MUTED, 10))  # noqa: F405
        body.append(text(x0 + steps * cw / 2, y0 + height + 32, "단계", MUTED, 10))  # noqa: F405
    y = y0 + height + 58
    body.append(line(20, y - 4, 48, y - 4, GREEN, 1.5, dash="5 4"))  # noqa: F405
    body.append(text(54, y, f"이보다 작을 수 없는 크기 ({bound:,}). 한 단계에 함께 살아 있는 값들을 더한 것 가운데 가장 큰 값이다.", MUTED, 12, "start"))  # noqa: F405
    body.append(text(20, y + 20, "네모 하나가 값 하나다. 가로로 겹치는(함께 살아 있는) 두 네모는 세로로 겹칠 수 없다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "layout.svg", 930, y + 36, body)  # noqa: F405


if __name__ == "__main__":
    fig_graph()
    fig_flatbuffers()
    fig_pipeline()
    plan_values, plan_totals = read_plan()
    assert len(plan_values) == 13 and len(plan_totals) == 4, (plan_values, plan_totals)
    fig_lifetimes(plan_values)
    fig_layout(plan_values, plan_totals)
    print("그림을", OUT, "에 썼습니다")
