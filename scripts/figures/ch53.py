"""53장의 표와 그림을 만든다.

실행: python3 scripts/figures/ch53.py
출력: book/src/images/ch53/scorecard.svg, bench/ch53/scorecard.md

둘 다 bench/ch53/scorecard.tsv 에서 나온다. 그 파일의 숫자는 앞 장들이 잰
것을 옮겨 적은 것이다.
"""

import math

from common import *  # noqa: F403

OUT = IMAGES / "ch53"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch53"  # noqa: F405
CHAPTERS = {
    25: "../part3/ch25_rematch.md",
    31: "../part4/ch31_mobile_engines.md",
    32: "../part4/ch32_vit_detection.md",
    33: "../part4/ch33_speech_recognition.md",
    34: "../part4/ch34_embedding_search.md",
    35: "../part4/ch35_image_generation.md",
    36: "../part4/ch36_multimodal.md",
    37: "../part4/ch37_speech_synthesis.md",
    45: "../part6/ch45_gpu_npu.md",
    47: "../part6/ch47_on_device_llm.md",
    50: "../part6/ch50_web_ecosystem.md",
    51: "ch51_why_not_pytorch.md",
}


def rows():
    out = []
    for line in (BENCH / "scorecard.tsv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        chapter, field, model, task, condition, ours, rival, theirs, unit, better = line.split("\t")
        ours, theirs = float(ours), float(theirs)
        # 우리가 상대의 몇 배 빠른가. 1보다 크면 우리가 빠르다.
        ratio = ours / theirs if better == "high" else theirs / ours
        out.append((int(chapter), field, model, task, condition, ours, rival, theirs, unit, ratio))
    return out


def number(value):
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:g}"


def table(data):
    lines = [
        "| 장 | 모델 | 일 | 조건 | badak | 상대 | 상대의 값 | 우리가 몇 배 빠른가 |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for chapter, _, model, task, condition, ours, rival, theirs, unit, ratio in data:
        shown = f"**{ratio:.2f}**" if ratio >= 1.05 else f"{ratio:.2f}"
        lines.append(
            f"| [{chapter}장]({CHAPTERS[chapter]}) | {model} | {task} | {condition} | {number(ours)}{unit} "
            f"| {rival} | {number(theirs)}{unit} | {shown} |"
        )
    (BENCH / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(data):
    left, right, top, step = 420, 900, 74, 17
    low, high = math.log10(0.02), math.log10(40)

    def x(ratio):
        return left + (math.log10(ratio) - low) / (high - low) * (right - left)

    height = top + len(data) * step + 46
    body = [
        text(20, 28, "우리 엔진이 상대의 몇 배 빠른가 (같은 기계, 같은 모델. 눈금은 로그)", INK, 14, "start", "bold"),  # noqa: F405
        text(x(1) - 8, 50, "← 상대가 빠르다", MUTED, 11, "end"),  # noqa: F405
        text(x(1) + 8, 50, "우리가 빠르다 →", MUTED, 11, "start"),  # noqa: F405
    ]
    for tick in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25):
        strong = tick == 1
        body.append(line(x(tick), top - 8, x(tick), height - 34, INK if strong else "#e5e7eb", 1.5 if strong else 1))  # noqa: F405
        body.append(text(x(tick), height - 18, f"{tick:g}배", MUTED, 10))  # noqa: F405
    previous = None
    for index, (chapter, field, model, task, condition, _, rival, _, _, ratio) in enumerate(data):
        y = top + index * step
        if field != previous and index:
            body.append(line(20, y - 5, right, y - 5, "#e5e7eb", 1))  # noqa: F405
        if field != previous:
            body.append(text(20, y + 8, f"{field} ({chapter}장)", INK, 11, "start", "bold"))  # noqa: F405
        previous = field
        body.append(text(left - 12, y + 8, f"{model} · {task} · {condition} · {rival}", MUTED, 9.5, "end"))  # noqa: F405
        color = BLUE if ratio >= 1.05 else (MUTED if ratio > 0.95 else ORANGE)  # noqa: F405
        body.append(line(x(1), y + 4, x(ratio), y + 4, color, 2))  # noqa: F405
        body.append(f"<circle cx='{x(ratio):.1f}' cy='{y + 4}' r='4' fill='{color}'/>")
        anchor, shift = ("start", 8) if ratio >= 1 else ("end", -8)
        body.append(text(x(ratio) + shift, y + 8, f"{ratio:.2f}", INK, 10, anchor))  # noqa: F405
    svg(OUT / "scorecard.svg", 930, height, body)  # noqa: F405


if __name__ == "__main__":
    data = rows()
    table(data)
    figure(data)
    wins = sum(ratio >= 1.05 for *_, ratio in data)
    ties = sum(0.95 < ratio < 1.05 for *_, ratio in data)
    print(f"줄 {len(data)} 우리가 빠른 것 {wins} 비슷한 것 {ties} 상대가 빠른 것 {len(data) - wins - ties}")
