"""22장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch22.py
출력: book/src/images/ch22/*.svg

측정값은 book/src/part3/output/ch20_memory_llama32.txt와 ch22_*.txt에서
읽는다. 다시 측정하면 그림도 다시 만들어야 한다.
"""

import re

from common import *  # noqa: F403

OUT = IMAGES / "ch22"  # noqa: F405
OUTPUT = IMAGES.parent / "part3/output"  # noqa: F405
GRID = "#e5e7eb"
LIGHT_BLUE = "#93c5fd"


def read(name):
    return (OUTPUT / name).read_text(encoding="utf-8")


def fig_two_numbers():
    body = [text(20, 30, "프로세스의 메모리를 세는 두 가지 방법", INK, 14, "start", "bold")]
    body.append(f"<rect x='40' y='56' width='560' height='150' rx='8' fill='#ffffff' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='6 4'/>")
    body.append(text(56, 78, "지금 실제 메모리에 올라와 있는 페이지 (올라온 메모리)", MUTED, 12, "start"))
    body += box(60, 92, 250, 96, "스스로 만든 값", "힙에 계산해 적은 것", ORANGE_BG, ORANGE)
    body += box(330, 92, 250, 96, "파일을 비춘 페이지", "파일에 같은 내용이 있다", BLUE_BG, BLUE)
    body.append(text(185, 226, "버리면 되찾을 길이 없다", ORANGE, 12))
    body.append(text(185, 244, "→ 운영체제가 지켜야 한다", ORANGE, 12))
    body.append(text(455, 226, "버려도 파일에서 다시 읽으면 된다", BLUE, 12))
    body.append(text(455, 244, "→ 모자라면 언제든 내준다", BLUE, 12))
    body.append(line(620, 140, 664, 140, INK, 1.5, arrow=True))
    body += box(670, 92, 190, 96, "지켜야 하는 메모리", "주황색 상자만 센다", ORANGE_BG, ORANGE)
    svg(OUT / "two_numbers.svg", 880, 262, body)


def read_memory(name):
    rows = {}
    for row in read(name).splitlines():
        kind = row.split()[0]
        numbers = [float(v) for v in re.findall(r"([\d.]+)GB", row)]
        seconds = float(re.search(r"읽는 시간\s+([\d.]+)초", row)[1])
        rows[kind] = (seconds, numbers)
    return rows


def fig_memory():
    before = read_memory("ch20_memory_llama32.txt")
    after = read_memory("ch22_memory_llama32.txt")
    labels = [("F32", "f32로 펼침"), ("Stored", "BF16 그대로"), ("Q8_0", "Q8_0"), ("Q4_0", "Q4_0")]
    x0, w, top = 210, 420, 8.0
    body = [text(20, 28, "가중치를 두는 방식과 메모리 (Llama 3.2 1B, GB)", INK, 14, "start", "bold")]
    for index, (label, color) in enumerate([("올라온 메모리", LIGHT_BLUE), ("지켜야 하는 메모리", ORANGE), ("지켜야 하는 메모리 (20장)", MUTED)]):
        body.append(f"<rect x='{x0 + index * 170}' y='42' width='14' height='11' rx='2' fill='{color}'/>")
        body.append(text(x0 + 20 + index * 170, 52, label, MUTED, 11, "start"))
    y = 72
    for key, label in labels:
        seconds, numbers = after[key]
        resident, footprint = numbers[1], numbers[2]
        old = before[key][1][1]
        body.append(text(x0 - 12, y + 26, label, INK, 12, "end"))
        for value, color in ((resident, LIGHT_BLUE), (footprint, ORANGE), (old, MUTED)):
            width = value / top * w
            body.append(f"<rect x='{x0}' y='{y}' width='{max(width, 1.5):.1f}' height='13' rx='3' fill='{color}'/>")
            body.append(text(x0 + max(width, 1.5) + 8, y + 11, f"{value:.2f}", MUTED, 11, "start"))
            y += 16
        y += 12
    body.append(text(20, y + 8, "회색은 읽을 때마다 양자화하던 20장의 값이다. 가중치가 파일을 비춘 메모리에 있으면 지켜야 하는 메모리는 거의 0이다.", MUTED, 12, "start"))
    svg(OUT / "memory.svg", 880, y + 24, body)


def read_allocations(name):
    out = {}
    text_ = read(name)
    for block in text_.split("모델 ")[1:]:
        kind = re.search(r"가중치 (\w+)", block)[1]
        prompt = int(re.search(r"프롬프트 256토큰\s+(\d+)", block)[1])
        token = int(re.search(r"새 토큰 하나 \(32개의 평균\)\s+(\d+)", block)[1])
        # 캐시의 자리를 미리 잡은 줄이 있으면 그 값을 쓴다(22.7절부터의 기본).
        reserved = re.search(r"캐시의 자리를 미리 잡으면\s+(\d+)", block)
        if reserved:
            token = int(reserved[1])
        out[kind] = (prompt, token)
    return out


def fig_allocations():
    import math

    stages = [
        ("21장까지", read_allocations("ch22_allocations_before.txt")["Stored"], MUTED),
        ("눈에 띄는 것을 고친 뒤 (22.8절)", read_allocations("ch22_allocations_after.txt")["Stored"], LIGHT_BLUE),
        ("형상을 텐서 안에, 자리를 돌려 써서 (22.10절)", read_allocations("ch22_allocations_no_arena.txt")["Stored"], BLUE),
        ("아레나 (22.11절)", read_allocations("ch22_allocations_arena.txt")["Stored"], GREEN),
    ]
    x0, w, top = 330, 400, 6.0
    body = [text(20, 28, "메모리를 할당하는 횟수 (스케일 하나가 열 배)", INK, 14, "start", "bold")]
    y = 52
    for index, title in enumerate(("프롬프트 256토큰", "새 토큰 하나")):
        body.append(text(20, y + 12, title, INK, 13, "start", "bold"))
        y += 22
        for power in range(0, 7):
            x = x0 + power / top * w
            body.append(line(x, y - 4, x, y + 4 * 24, GRID, 1))
        for label, values, color in stages:
            value = values[index]
            width = math.log10(value) / top * w
            body.append(text(x0 - 12, y + 14, label, INK, 12, "end"))
            body.append(f"<rect x='{x0}' y='{y + 2}' width='{max(width, 2):.1f}' height='16' rx='3' fill='{color}'/>")
            body.append(text(x0 + max(width, 2) + 8, y + 15, f"{value:,}번", MUTED, 12, "start"))
            y += 24
        y += 6
        if index == 1:
            for power in range(0, 7):
                x = x0 + power / top * w
                label = "1" if power == 0 else f"{10 ** power:,}"
                body.append(text(x, y + 8, label, MUTED, 10))
        y += 16
    body.append(text(20, y + 8, "막대의 길이는 횟수의 로그다. 길이가 조금 줄어도 횟수는 몇 배씩 준다.", MUTED, 12, "start"))
    svg(OUT / "allocations.svg", 880, y + 24, body)


def fig_arena():
    body = [text(20, 28, "아레나: 덩어리 하나에서 차례로 떼어 주고, 한꺼번에 돌려준다", INK, 14, "start", "bold")]
    x0, w, h = 150, 680, 40
    parts = [("정규화", 1), ("질의", 1), ("키", 0.5), ("값", 0.5), ("…", 0.8), ("MLP의 중간 결과", 2.6)]
    unit = 86

    def bar(y, filled, title, note):
        out = [text(x0 - 14, y + 18, title, INK, 12, "end", "bold"), text(x0 - 14, y + 34, note, MUTED, 11, "end")]
        out.append(f"<rect x='{x0}' y='{y}' width='{w}' height='{h}' rx='4' fill='{GRAY_BG}' stroke='{MUTED}'/>")
        x = x0
        for label, size in parts[:filled]:
            width = size * unit
            out.append(f"<rect x='{x}' y='{y}' width='{width}' height='{h}' fill='{BLUE_BG}' stroke='{BLUE}'/>")
            out.append(text(x + width / 2, y + 25, label, INK, 12))
            x += width
        out.append(line(x, y + h + 22, x, y + h + 3, ORANGE, 2, arrow=True))
        anchor = "start" if filled == 0 else "middle"
        out.append(text(x + (8 if filled == 0 else 0), y + h + (20 if filled == 0 else 36), "다음에 내줄 자리", ORANGE, 11, anchor))
        return out

    body += bar(56, 2, "블록을 계산하는 중", "텐서 둘을 만들었다")
    body.append(text(x0 + 3 * unit, 56 + 25, "아직 내주지 않은 자리", MUTED, 12, "start"))
    body += bar(156, 6, "블록의 끝", "텐서 열여섯을 만들었다")
    body += bar(256, 0, "다음 블록의 시작", "수 하나를 0으로 되돌렸다")
    body.append(text(x0 + w / 2, 256 + 25, "같은 메모리를 처음부터 다시 쓴다", MUTED, 12))
    body.append(text(20, 352, "얻는 일은 \"다음에 내줄 자리\"에 크기를 더하는 것이고, 돌려주는 일은 블록이 끝날 때 한 번뿐이다.", MUTED, 12, "start"))
    body.append(text(20, 370, "블록이 끝날 때까지는 아무것도 돌려주지 않으므로, 블록이 만드는 텐서를 모두 더한 만큼을 쥔다.", MUTED, 12, "start"))
    svg(OUT / "arena.svg", 880, 386, body)


if __name__ == "__main__":
    for figure in (fig_two_numbers, fig_memory, fig_allocations, fig_arena):
        figure()
    print(f"wrote {len(list(OUT.glob('*.svg')))} figures to {OUT}")
