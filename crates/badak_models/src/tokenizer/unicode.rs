//! 토크나이저에 필요한 유니코드 처리: 글자 분류와 바이트 표기.

use super::unicode_tables::{DECOMPOSITION, LETTER, NONSPACING_MARK, NUMBER, OTHER, PUNCTUATION};

/// 정렬된 구간 표에서 코드 포인트를 이진 탐색한다.
fn contains(table: &[(u32, u32)], character: char) -> bool {
    let code = character as u32;
    table
        .binary_search_by(|&(first, last)| {
            if code < first {
                std::cmp::Ordering::Greater
            } else if code > last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

// ANCHOR: classes
/// 글자인가. 정규식의 `\p{L}`.
pub fn is_letter(character: char) -> bool {
    contains(LETTER, character)
}

/// 숫자인가. 정규식의 `\p{N}`.
pub fn is_number(character: char) -> bool {
    contains(NUMBER, character)
}

/// 공백인가. 정규식의 `\s`.
pub fn is_space(character: char) -> bool {
    character.is_whitespace()
}
// ANCHOR_END: classes

// ANCHOR: bert_classes
/// 자리를 차지하지 않는 결합 기호인가. 일반 범주 Mn. `é`를 풀었을 때
/// 나오는 `´`(U+0301) 같은 것이다.
pub fn is_nonspacing_mark(character: char) -> bool {
    contains(NONSPACING_MARK, character)
}

/// 문장 부호인가. 일반 범주 P.
pub fn is_punctuation(character: char) -> bool {
    contains(PUNCTUATION, character)
}

/// 제어 문자, 서식 문자, 배정되지 않은 자리인가. 일반 범주 C.
pub fn is_other(character: char) -> bool {
    contains(OTHER, character)
}
// ANCHOR_END: bert_classes

// ANCHOR: decompose
/// 글자 하나를 바탕 글자와 결합 기호로 풀어 `out`에 붙인다. 유니코드의
/// 정준 분해(NFD)다.
///
/// 한글 음절은 표 없이 식으로 푼다. `가`(U+AC00)부터 초성 19 × 중성 21 ×
/// 종성 28(없음 포함)의 순서로 놓여 있기 때문이다. 나머지는 표에서 찾는다.
///
/// 결합 기호가 여럿일 때 정해진 순서로 다시 놓는 일은 하지 않는다.
/// WordPiece는 푼 다음에 결합 기호를 지우므로 그 순서가 결과에 남지 않는다.
pub fn decompose(character: char, out: &mut Vec<char>) {
    const FIRST: u32 = 0xac00;
    const LAST: u32 = 0xd7a3;
    let code = character as u32;
    if (FIRST..=LAST).contains(&code) {
        let index = code - FIRST;
        let (lead, vowel, tail) = (index / (21 * 28), index / 28 % 21, index % 28);
        // 초성은 U+1100, 중성은 U+1161, 종성은 U+11A8부터다.
        out.extend(char::from_u32(0x1100 + lead));
        out.extend(char::from_u32(0x1161 + vowel));
        if tail > 0 {
            out.extend(char::from_u32(0x11a7 + tail));
        }
        return;
    }
    match DECOMPOSITION.binary_search_by_key(&code, |&(from, _)| from) {
        Ok(found) => out.extend(
            DECOMPOSITION[found]
                .1
                .iter()
                .filter_map(|&part| char::from_u32(part)),
        ),
        Err(_) => out.push(character),
    }
}
// ANCHOR_END: decompose

// ANCHOR: byte_char
/// 바이트 하나를 눈에 보이는 글자 하나로 적는다.
///
/// 바이트 단위 BPE의 단어 사전은 토큰을 바이트열이 아니라 문자열로
/// 적는다. 공백이나 제어 문자처럼 그대로 적기 곤란한 바이트는 U+0100부터의
/// 글자로 옮겨 적는다. 예를 들어 공백(0x20)은 `Ġ`(U+0120)가 된다.
pub fn byte_to_char(byte: u8) -> char {
    BYTE_CHARS[usize::from(byte)]
}

/// [`byte_to_char`]의 반대. 그런 글자가 아니면 `None`.
pub fn char_to_byte(character: char) -> Option<u8> {
    BYTE_CHARS
        .iter()
        .position(|&mapped| mapped == character)
        .map(|byte| byte as u8)
}

/// 바이트 256개 각각에 대응하는 글자.
static BYTE_CHARS: [char; 256] = {
    let mut table = ['\0'; 256];
    let mut shifted = 0;
    let mut byte = 0;
    while byte < 256 {
        // 그대로 적어도 되는 바이트: 인쇄 가능한 ASCII와 Latin-1의 대부분.
        let printable = matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
        table[byte] = if printable {
            char::from_u32(byte as u32).unwrap()
        } else {
            // 나머지는 U+0100부터 차례로 배정한다.
            shifted += 1;
            char::from_u32(0x100 + shifted - 1).unwrap()
        };
        byte += 1;
    }
    table
};
// ANCHOR_END: byte_char

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_letters_numbers_and_marks() {
        for letter in ['a', 'Z', 'é', '한', '我', 'ក', 'я'] {
            assert!(is_letter(letter), "{letter}");
        }
        for number in ['0', '9', '½', '٣', 'Ⅳ'] {
            assert!(is_number(number) && !is_letter(number), "{number}");
        }
        // 결합 기호는 글자가 아니다. 표준 라이브러리의 is_alphabetic은
        // 이들을 글자로 보므로 직접 만든 표가 필요하다.
        let khmer_vowel_sign = '\u{17b6}';
        assert!(!is_letter(khmer_vowel_sign) && khmer_vowel_sign.is_alphabetic());
        for other in [' ', '!', '🦙', '\n', '_'] {
            assert!(!is_letter(other) && !is_number(other), "{other:?}");
        }
    }

    #[test]
    fn decomposes_accents_and_hangul() {
        let decomposed = |text: &str| {
            let mut out = Vec::new();
            text.chars()
                .for_each(|character| decompose(character, &mut out));
            out
        };
        assert_eq!(decomposed("é"), ['e', '\u{301}']);
        // 두 번 풀어야 하는 글자도 끝까지 푼 것이 표에 들어 있다.
        assert_eq!(decomposed("ǘ"), ['u', '\u{308}', '\u{301}']);
        assert_eq!(decomposed("한"), ['\u{1112}', '\u{1161}', '\u{11ab}']);
        assert_eq!(decomposed("가"), ['\u{1100}', '\u{1161}']);
        assert_eq!(decomposed("힣"), ['\u{1112}', '\u{1175}', '\u{11c2}']);
        assert_eq!(decomposed("a1!"), ['a', '1', '!']);
        assert!(is_nonspacing_mark('\u{301}') && !is_nonspacing_mark('e'));
        assert!(is_punctuation('!') && is_punctuation('。') && !is_punctuation('$'));
        assert!(is_other('\u{0}') && is_other('\u{200d}') && !is_other('a'));
    }

    #[test]
    fn byte_mapping_matches_known_entries_and_round_trips() {
        assert_eq!(byte_to_char(b'A'), 'A');
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
        assert_eq!(byte_to_char(0xe9), 'é');
        assert_eq!(byte_to_char(0xad), '\u{143}');
        for byte in 0..=255_u8 {
            assert_eq!(char_to_byte(byte_to_char(byte)), Some(byte));
        }
        assert_eq!(char_to_byte('한'), None);
    }
}
