"""토크나이저가 쓰는 유니코드 글자 분류 표를 Rust 소스로 생성한다.

실행: python3 scripts/generate_unicode_tables.py
출력: crates/badak_models/src/tokenizer/unicode_tables.rs

정규식의 \\p{L}(글자)과 \\p{N}(숫자)에 해당하는 코드 포인트의 구간을
만든다. 분류는 Python 표준 라이브러리의 unicodedata에서 가져온다.
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
        ]
    )
    OUT.write_text(source, encoding="utf-8")
    print(f"유니코드 {version}: {OUT.name} 생성")


if __name__ == "__main__":
    main()
