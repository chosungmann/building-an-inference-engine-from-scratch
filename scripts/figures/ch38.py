"""38장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch38.py
출력: book/src/images/ch38/*.svg

latency.svg 는 bench/ch38/ 에 남긴 측정값의 중앙값을 그린다. 그 파일이
없으면 그 그림은 건너뛴다.
"""

import re
import statistics

from common import *  # noqa: F403

OUT = IMAGES / "ch38"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch38"  # noqa: F405
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_pipeline():
    """모델 셋을 잇는다."""
    body = [text(20, 28, "말을 듣고 말로 답하기: 모델 셋", INK, 14, "start", "bold")]  # noqa: F405
    steps = [
        ("사용자의 말", "소리, 16kHz", GRAY_BG, MUTED),  # noqa: F405
        ("받아쓰기", "Whisper (33장)", BLUE_BG, BLUE),  # noqa: F405
        ("글", "And so my fellow…", GRAY_BG, MUTED),  # noqa: F405
        ("답 쓰기", "언어 모델 (2부)", ORANGE_BG, ORANGE),  # noqa: F405
        ("문장", "끝나는 대로 하나씩", GRAY_BG, MUTED),  # noqa: F405
        ("소리로", "음성 합성 (37장)", GREEN_BG, GREEN),  # noqa: F405
        ("답의 소리", "24kHz, 낱말마다", GRAY_BG, MUTED),  # noqa: F405
    ]
    w, gap, y = 138, 20, 56
    for index, (title, sub, fill, stroke) in enumerate(steps):
        x = 20 + index * (w + gap)
        body += box(x, y, w, 50, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(x - gap + 2, y + 25, x - 3, y + 25, INK, 1.5, True))  # noqa: F405
    x_from = 20 + 3 * (w + gap) + w / 2
    body.append(f"<path d='M{x_from + 2 * (w + gap)},{y + 52} L{x_from + 2 * (w + gap)},{y + 84} L{x_from},{y + 84} L{x_from},{y + 54}' fill='none' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5,4' marker-end='url(#arrow-{MUTED[1:]})'/>")  # noqa: F405
    body.append(text(x_from + (w + gap), y + 100, "문장 하나를 다 말하면 다음 문장을 쓰러 돌아간다", MUTED, 12))  # noqa: F405
    body.append(text(20, y + 130, "모델 사이를 오가는 것은 글이다. 앞 모델이 다 끝나기를 기다리지 않고, 쓸 수 있는 만큼이 나오면 다음 모델로 넘긴다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "pipeline.svg", 1140, y + 148, body)  # noqa: F405


def medians():
    values = {}
    for path in sorted(BENCH.glob("badak_*.txt")):
        for row in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"talk (\S+) (\S+) 스레드 +(\d+) (.*)", row)
            if not found:
                continue
            fields = found[4].split()
            pairs = dict(zip(fields[0:16:2], fields[1:16:2]))
            for name, value in pairs.items():
                values.setdefault((found[1], found[2], int(found[3]), name), []).append(float(value))
    return {key: statistics.median(samples) for key, samples in values.items()}


def fig_latency(values):
    """말이 끝난 때부터 첫 소리까지, 구간마다."""
    body = [text(20, 28, "말이 끝난 때부터 첫 소리까지 (밀리초, 스레드 12개)", INK, 14, "start", "bold")]  # noqa: F405
    parts = [("듣기", BLUE), ("답의 첫 토큰", ORANGE), ("첫 문장(또는 답 전체)", PURPLE), ("첫 낱말의 소리", GREEN)]  # noqa: F405
    x_legend = 20
    for label, color in parts:
        body.append(rect(x_legend, 44, 14, 10, color, None, 2))
        body.append(text(x_legend + 20, 54, label, INK, 12, "start"))  # noqa: F405
        x_legend += 60 + 12 * len(label)
    rows = []
    for chat, name in (("smollm2_135m", "SmolLM2-135M"), ("llama32_1b", "Llama 3.2 1B")):
        for mode, label in (("sentence", "문장마다"), ("whole", "다 쓴 뒤에")):
            key = (chat, mode, 12)
            if key + ("첫소리",) not in values:
                continue
            get = lambda field: values[key + (field,)]  # noqa: E731
            marks = [get("들음"), get("첫토큰"), get("첫문장") if mode == "sentence" else get("답끝"), get("첫소리")]
            rows.append((f"{name}, {label}", marks))
    top = max(marks[-1] for _, marks in rows)
    scale = 620 / top
    for index, (label, marks) in enumerate(rows):
        y = 78 + index * 36
        body.append(text(210, y + 17, label, INK, 12, "end"))  # noqa: F405
        previous = 0.0
        for mark, (_, color) in zip(marks, parts):
            w = (mark - previous) * scale
            body.append(rect(220 + previous * scale, y, max(w - 2, 1), 24, color, None, 3))
            if w > 40:
                body.append(text(220 + previous * scale + w / 2 - 1, y + 17, f"{mark - previous:.0f}", "#ffffff", 12))  # noqa: F405
            previous = mark
        body.append(text(220 + marks[-1] * scale + 8, y + 17, f"{marks[-1]:.0f}", INK, 12, "start"))  # noqa: F405
    note = 78 + len(rows) * 36 + 16
    body.append(text(20, note, "막대의 길이가 사용자가 기다리는 시간이다. 답이 한 문장이면(Llama 3.2 1B) 두 방법이 같다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "latency.svg", 960, note + 16, body)  # noqa: F405


if __name__ == "__main__":
    fig_pipeline()
    if list(BENCH.glob("badak_*.txt")):
        fig_latency(medians())
