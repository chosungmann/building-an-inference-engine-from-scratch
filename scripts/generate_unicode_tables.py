"""토크나이저가 쓰는 유니코드 글자 분류 표를 Rust 소스로 생성한다.

실행: python3 scripts/generate_unicode_tables.py
출력: crates/badak_models/src/tokenizer/unicode_tables.rs

정규식의 \\p{L}(글자)과 \\p{N}(숫자)에 해당하는 코드 포인트의 구간을
만든다. 분류는 Python 표준 라이브러리의 unicodedata에서 가져온다.

34장의 WordPiece 토크나이저가 쓰는 표도 여기서 만든다. 결합 기호(Mn),
문장 부호(P), 그 밖(C)의 구간과, 글자를 바탕 글자와 결합 기호로 푸는
표(정준 분해, NFD)다. 한글 음절은 식으로 풀 수 있어서 표에 넣지 않는다.
"""

import sys
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "crates/badak_models/src/tokenizer/unicode_tables.rs"


def ranges(prefix):
    """일반 범주가 prefix로 시작하는 코드 포인트를 이어진 구간으로 묶는다."""
    result, start = [], None
    for code in range(sys.maxunicode + 2):
        inside = code <= sys.maxunicode and unicodedata.category(chr(code)).startswith(prefix)
        if inside and start is None:
            start = code
        elif not inside and start is not None:
            result.append((start, code - 1))
            start = None
    return result


def table(name, doc, items):
    lines = [f"/// {doc}", f"pub(super) const {name}: &[(u32, u32)] = &["]
    for first, last in items:
        lines.append(f"    ({first:#06x}, {last:#06x}),")
    lines.append("];")
    return "\n".join(lines)


def decompositions():
    """정준 분해가 자신과 다른 글자와 그 분해. 한글 음절(가-힣)은 뺀다."""
    result = []
    for code in range(sys.maxunicode + 1):
        if 0xAC00 <= code <= 0xD7A3 or 0xD800 <= code <= 0xDFFF:
            continue
        parts = unicodedata.normalize("NFD", chr(code))
        if parts != chr(code):
            result.append((code, [ord(part) for part in parts]))
    return result


def decomposition_table(items):
    lines = [
        "/// 정준 분해(NFD). (글자, 끝까지 푼 글자들)이며 글자의 오름차순이다.",
        "/// 한글 음절은 식으로 풀므로 여기에 없다.",
        "pub(super) const DECOMPOSITION: &[(u32, &[u32])] = &[",
    ]
    for code, parts in items:
        inner = ", ".join(f"{part:#06x}" for part in parts)
        lines.append(f"    ({code:#06x}, &[{inner}]),")
    lines.append("];")
    return "\n".join(lines)


def main():
    version = unicodedata.unidata_version
    source = "\n".join(
        [
            "//! 유니코드 글자 분류 표. `scripts/generate_unicode_tables.py`가 생성한다.",
            "//! 손으로 고치지 않는다.",
            "//!",
            f"//! 유니코드 {version} 기준이다. 각 항목은 (첫 코드 포인트, 마지막 코드 포인트)이며",
            "//! 오름차순으로 정렬되어 있다.",
            "",
            table("LETTER", "일반 범주가 L(글자)인 구간. 정규식의 `\\p{L}`에 해당한다.", ranges("L")),
            "",
            table("NUMBER", "일반 범주가 N(숫자)인 구간. 정규식의 `\\p{N}`에 해당한다.", ranges("N")),
            "",
            table("NONSPACING_MARK", "일반 범주가 Mn(자리를 차지하지 않는 결합 기호)인 구간.", ranges("Mn")),
            "",
            table("PUNCTUATION", "일반 범주가 P(문장 부호)인 구간.", ranges("P")),
            "",
            table("OTHER", "일반 범주가 C(제어 문자, 서식 문자, 배정되지 않은 자리 등)인 구간.", ranges("C")),
            "",
            decomposition_table(decompositions()),
            "",
        ]
    )
    OUT.write_text(source, encoding="utf-8")
    print(f"유니코드 {version}: {OUT.name} 생성")


if __name__ == "__main__":
    main()
