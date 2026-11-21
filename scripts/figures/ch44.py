"""44장의 그림을 SVG로 생성한다.

실행: python3 scripts/figures/ch44.py
출력: book/src/images/ch44/*.svg
"""

from common import *  # noqa: F403

OUT = IMAGES / "ch44"  # noqa: F405
PURPLE, PURPLE_BG, RED = "#7c3aed", "#ede9fe", "#dc2626"


def rect(x, y, w, h, fill, stroke=None, rx=0, width=1, dash=None):
    edge = f" stroke='{stroke}' stroke-width='{width}'" if stroke else ""
    edge += f" stroke-dasharray='{dash}'" if dash else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}'{edge}/>"


def fig_boundary():
    """앱에서 엔진까지의 층."""
    body = [text(20, 28, "앱에서 엔진까지: 두 운영체제가 같은 문으로 들어온다", INK, 14, "start", "bold")]  # noqa: F405
    columns = [(60, "iOS", "Swift"), (500, "Android", "Java, Kotlin")]
    w = 380
    for x, system, language in columns:
        body.append(text(x + w / 2, 58, system, INK, 13, "middle", "bold"))  # noqa: F405
        body += box(x, 68, w, 46, "앱의 코드", language, BLUE_BG, BLUE)  # noqa: F405
        body.append(line(x + w / 2, 116, x + w / 2, 138, INK, 1.5, True))  # noqa: F405
    # 감싼 층.
    body += box(60, 142, w, 46, "Classifier (Swift 클래스)", "오류 숫자 → throw, deinit에서 닫기", BLUE_BG, BLUE)  # noqa: F405
    body += box(500, 142, w, 46, "Badak.Classifier (Java 클래스)", "주소를 long으로, close()에서 닫기", BLUE_BG, BLUE)  # noqa: F405
    body.append(line(60 + w / 2, 190, 60 + w / 2, 300, INK, 1.5, True))  # noqa: F405
    body.append(text(60 + w / 2 + 10, 222, "badak.h를 그대로 읽는다", MUTED, 12, "start"))  # noqa: F405
    body.append(text(60 + w / 2 + 10, 240, "(정적 라이브러리 .a)", MUTED, 12, "start"))  # noqa: F405
    body.append(line(500 + w / 2, 190, 500 + w / 2, 212, INK, 1.5, True))  # noqa: F405
    body += box(500, 216, w, 46, "JNI 함수 (Rust)", "Java_book_badak_Badak_…  배열과 글을 복사해 온다", ORANGE_BG, ORANGE)  # noqa: F405
    body.append(line(500 + w / 2, 264, 500 + w / 2, 300, INK, 1.5, True))  # noqa: F405
    body.append(text(500 + w / 2 + 10, 286, "(동적 라이브러리 .so)", MUTED, 12, "start"))  # noqa: F405
    # 문.
    body.append(rect(40, 304, 860, 62, ORANGE_BG, ORANGE, 8, 1.5))  # noqa: F405
    body.append(text(470, 328, "문: C 함수 (badak_ffi)", INK, 14, "middle", "bold"))  # noqa: F405
    body.append(text(470, 350, "패닉을 잡는다 · 오류를 숫자로 바꾼다 · 빈 포인터를 걸러 낸다 · 메모리는 만든 쪽이 지운다", MUTED, 12))  # noqa: F405
    body.append(line(470, 368, 470, 390, INK, 1.5, True))  # noqa: F405
    body.append(rect(40, 394, 860, 50, GREEN_BG, GREEN, 8, 1.5))  # noqa: F405
    body.append(text(470, 416, "엔진: badak_graph, badak_formats, badak_core, badak_image", INK, 14, "middle", "bold"))  # noqa: F405
    body.append(text(470, 434, "맥에서 돌던 것과 같은 Rust 코드. 고친 줄이 없다", MUTED, 12))  # noqa: F405
    body.append(text(20, 470, "문 위쪽은 운영체제마다 따로 적는다. 문과 그 아래는 한 번만 적는다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "boundary.svg", 940, 486, body)  # noqa: F405


def fig_install():
    """받은 파일을 놓는 차례와, 어디에서 멈춰도 남는 것."""
    body = [text(20, 28, "받은 모델을 놓는 차례: 어디에서 멈춰도 쓰던 모델이 남는다", INK, 14, "start", "bold")]  # noqa: F405
    steps = [
        ("크기를 본다", "끊긴 내려받기", BLUE_BG, BLUE),
        ("해시를 본다", "바뀐 비트", BLUE_BG, BLUE),
        ("옆 폴더에 차린다", "버전.partial/", GRAY_BG, MUTED),  # noqa: F405
        ("폴더의 이름을 바꾼다", "버전.partial → 버전", ORANGE_BG, ORANGE),
        ("current를 바꾼다", "한 번의 rename", GREEN_BG, GREEN),  # noqa: F405
        ("돌려 본다", "안 되면 되돌린다", PURPLE_BG, PURPLE),
    ]
    w, gap, x0, y = 150, 18, 20, 56
    for index, (title, sub, fill, stroke) in enumerate(steps):
        x = x0 + index * (w + gap)
        body += box(x, y, w, 50, title, sub, fill, stroke)  # noqa: F405
        if index:
            body.append(line(x - gap + 2, y + 25, x - 3, y + 25, INK, 1.5, True))  # noqa: F405
    # 아래: 그 단계에서 실패하거나 죽으면.
    y2 = 150
    notes = [
        ("거절. 다시 받는다", RED),
        ("거절. 다시 받는다", RED),
        ("하다 만 폴더가 남는다", MUTED),  # noqa: F405
        ("새 버전이 놓였지만", MUTED),  # noqa: F405
        ("이 순간부터 새 버전", GREEN),  # noqa: F405
        ("previous로 돌아간다", PURPLE),
    ]
    second = ["쓰던 버전 그대로", "쓰던 버전 그대로", "다음에 치운다. 쓰던 버전 그대로", "아직 쓰던 버전을 쓴다", "옛 버전은 previous에", "옛 버전은 지우지 않았다"]
    for index, ((note, color), more) in enumerate(zip(notes, second)):
        x = x0 + index * (w + gap)
        body.append(rect(x, y2, w, 46, "#ffffff", color, 6, 1, "4,3"))
        body.append(text(x + w / 2, y2 + 19, note, color, 12))  # noqa: F405
        body.append(text(x + w / 2, y2 + 36, more, MUTED, 11))  # noqa: F405
        body.append(line(x + w / 2, y + 52, x + w / 2, y2 - 2, MUTED, 1, False, "3,3"))  # noqa: F405
    body.append(rect(14, y2 - 28, 216, 20, "#ffffff"))
    body.append(text(20, y2 - 14, "여기에서 실패하거나 앱이 죽으면", INK, 12, "start", "bold"))  # noqa: F405
    body.append(text(20, 226, "쓰던 버전을 건드리는 단계가 없다. 바뀌는 것은 current라는 글 파일 하나이고, 그것도 이름 바꾸기 한 번이다.", MUTED, 12, "start"))  # noqa: F405
    svg(OUT / "install.svg", 1040, 242, body)  # noqa: F405


if __name__ == "__main__":
    fig_boundary()
    fig_install()
