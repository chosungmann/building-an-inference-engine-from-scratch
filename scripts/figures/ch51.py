"""51장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch51.py
출력: book/src/images/ch51/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch51"  # noqa: F405
BENCH = IMAGES.parents[2] / "bench/ch51"  # noqa: F405
PURPLE = "#7c3aed"


def rect(x, y, w, h, fill, stroke=None, rx=6, width=1.5):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_whole():
    """글 한 줄을 쓰고 끝나기까지: 시간과 메모리."""
    rows = {}
    for line in (BENCH / "whole.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        rows[parts[1]] = (float(parts[3]), float(parts[5]), float(parts[7]))
    order = [
        ("python-만", "Python만 띄운다", MUTED),  # noqa: F405
        ("python-import-torch", "import torch", ORANGE),  # noqa: F405
        ("pytorch-글-한-줄", "PyTorch로 글 한 줄", ORANGE),  # noqa: F405
        ("badak-글-한-줄", "badak으로 글 한 줄", BLUE),  # noqa: F405
    ]
    body = [text(20, 28, "프로그램을 띄워 SmolLM2-135M으로 토큰 40개를 쓰고 끝내기까지", INK, 14, "start", "bold")]  # noqa: F405
    panels = [(0, "걸린 시간", "초", 250), (2, "가장 많이 쥔 메모리 (운영체제가 돌려받을 수 없는 것)", "MB", 590)]
    for index, title, unit, left in panels:
        body.append(text(left, 56, title, INK, 12, "start", "bold"))  # noqa: F405
        longest = max(rows[key][index] for key, _, _ in order)
        y = 70
        for key, label, color in order:
            value = rows[key][index]
            if index == 0:
                body.append(text(left - 10, y + 12, label, INK, 12, "end"))  # noqa: F405
            wide = max(1.0, value / longest * 200)
            body.append(rect(left, y, wide, 16, color, None, 2))
            shown = f"{value:.2f}{unit}" if unit == "초" else f"{value:.0f}{unit}"
            body.append(text(left + wide + 6, y + 12, shown, INK, 11, "start"))  # noqa: F405
            y += 24
    svg(OUT / "whole.svg", 900, 174, body)  # noqa: F405


def fig_paths():
    """PyTorch 모델이 Python 밖으로 나가는 길들."""
    body = [text(20, 28, "PyTorch 모델이 Python을 떠나는 길", INK, 14, "start", "bold")]  # noqa: F405

    def node(x, y, w, title, sub, fill, stroke):
        body.append(rect(x, y, w, 46, fill, stroke))
        body.append(text(x + w / 2, y + 20, title, INK, 12, "middle", "bold"))  # noqa: F405
        body.append(text(x + w / 2, y + 37, sub, MUTED, 11))  # noqa: F405

    node(20, 120, 150, "PyTorch 모델", "Python 코드와 가중치", ORANGE_BG, ORANGE)  # noqa: F405
    node(230, 50, 170, "TorchScript", "trace, script. 낡은 길", GRAY_BG, MUTED)  # noqa: F405
    node(230, 120, 170, "torch.export", "그래프 (ExportedProgram)", BLUE_BG, BLUE)  # noqa: F405
    node(230, 190, 170, "torch.compile", "Python 안에서 컴파일", GRAY_BG, MUTED)  # noqa: F405
    node(470, 85, 170, "ONNX 파일", "torch.onnx.export", BLUE_BG, BLUE)  # noqa: F405
    node(470, 155, 170, "ExecuTorch의 .pte", "낮추기, 맡기기, 메모리 계획", BLUE_BG, BLUE)  # noqa: F405
    node(710, 50, 170, "ONNX Runtime", "30장, 50장", GREEN_BG, GREEN)  # noqa: F405
    node(710, 108, 170, "badak", "이 장에서 읽어 본다", GREEN_BG, GREEN)  # noqa: F405
    node(710, 166, 170, "ExecuTorch 실행기", "기기에서 돈다 (47장)", GREEN_BG, GREEN)  # noqa: F405
    for x1, y1, x2, y2 in [(170, 135, 226, 78), (170, 143, 226, 143), (170, 151, 226, 208), (400, 138, 466, 112), (400, 148, 466, 174), (640, 100, 706, 76), (640, 112, 706, 130), (640, 180, 706, 188)]:
        body.append(line(x1, y1, x2, y2, INK, 1.5, True))  # noqa: F405
    svg(OUT / "paths.svg", 900, 250, body)  # noqa: F405


if __name__ == "__main__":
    fig_whole()
    fig_paths()
