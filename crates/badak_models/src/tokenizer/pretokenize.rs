//! 사전 분할: 문장을 단어 크기의 조각으로 자른다.
//!
//! BPE는 이 조각 안에서만 글자를 합친다. 조각의 경계를 넘는 토큰은
//! 만들어지지 않는다. 자르는 규칙은 모델마다 정규식으로 정해져 있는데,
//! 정규식 엔진 없이 같은 규칙을 직접 구현한다.

use super::unicode::{is_letter, is_number, is_space};

// ANCHOR: kind
/// 모델이 쓰는 사전 분할 규칙.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// GPT-2의 규칙.
    Gpt2,
    /// Llama 3의 규칙. 숫자를 세 자리씩 끊고 줄바꿈을 따로 다룬다.
    Llama3,
}
// ANCHOR_END: kind

impl PreTokenizer {
    /// GGUF의 `tokenizer.ggml.pre` 값에서 규칙을 고른다.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "gpt-2" => Some(Self::Gpt2),
            "llama-bpe" | "llama3" | "llama-v3" => Some(Self::Llama3),
            _ => None,
        }
    }

    // ANCHOR: split
    /// 문장을 조각으로 자른다. 조각을 이어 붙이면 원래 문장이 된다.
    pub fn split(self, text: &str) -> Vec<&str> {
        // 글자 단위로 다루되, 조각은 원래 문자열의 바이트 구간으로 돌려준다.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let offset = |index: usize| chars.get(index).map_or(text.len(), |&(byte, _)| byte);

        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let len = match self {
                Self::Gpt2 => match_gpt2(&chars[start..]),
                Self::Llama3 => match_llama3(&chars[start..]),
            };
            // 어느 규칙에도 맞지 않는 글자는 혼자 조각이 된다.
            let end = start + len.max(1);
            pieces.push(&text[offset(start)..offset(end)]);
            start = end;
        }
        pieces
    }
    // ANCHOR_END: split
}

/// `chars`의 앞에서 조건을 만족하는 글자가 몇 개 이어지는지 센다.
fn run(chars: &[(usize, char)], predicate: impl Fn(char) -> bool) -> usize {
    chars.iter().take_while(|&&(_, c)| predicate(c)).count()
}

fn at(chars: &[(usize, char)], index: usize) -> Option<char> {
    chars.get(index).map(|&(_, c)| c)
}

fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// 영어의 줄임 표현(`'s`, `'re` 등)의 길이. `any_case`이면 대문자도 받는다.
fn contraction(chars: &[(usize, char)], any_case: bool) -> usize {
    if at(chars, 0) != Some('\'') {
        return 0;
    }
    let fold = |c: char| if any_case { c.to_ascii_lowercase() } else { c };
    let first = at(chars, 1).map(fold);
    let second = at(chars, 2).map(fold);
    match (first, second) {
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
        (Some('s' | 't' | 'm' | 'd'), _) => 2,
        _ => 0,
    }
}

/// 공백이 이어진 구간에서 `\s+(?!\S)`가 차지하는 길이.
///
/// 구간 뒤에 공백이 아닌 글자가 오면 마지막 공백 하나를 남긴다. 그 공백은
/// 뒤에 오는 단어에 붙는다.
fn trailing_space_run(chars: &[(usize, char)]) -> usize {
    let spaces = run(chars, is_space);
    if spaces == chars.len() {
        spaces
    } else {
        spaces.saturating_sub(1)
    }
}

// ANCHOR: gpt2
/// GPT-2의 규칙으로 맨 앞 조각의 길이를 구한다.
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
fn match_gpt2(chars: &[(usize, char)]) -> usize {
    let found = contraction(chars, false);
    if found > 0 {
        return found;
    }
    // 공백 하나가 앞에 붙을 수 있는 세 가지: 글자들, 숫자들, 그 밖의 것들.
    let lead = usize::from(at(chars, 0) == Some(' '));
    for class in [is_letter as fn(char) -> bool, is_number, is_other] {
        let body = run(&chars[lead..], class);
        if body > 0 {
            return lead + body;
        }
    }
    match trailing_space_run(chars) {
        0 => run(chars, is_space),
        spaces => spaces,
    }
}
// ANCHOR_END: gpt2

// ANCHOR: llama3
/// Llama 3의 규칙으로 맨 앞 조각의 길이를 구한다.
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
fn match_llama3(chars: &[(usize, char)]) -> usize {
    // 1. 줄임 표현. 대소문자를 가리지 않는다.
    let found = contraction(chars, true);
    if found > 0 {
        return found;
    }
    let first = chars[0].1;

    // 2. 글자들. 앞에 글자도 숫자도 줄바꿈도 아닌 글자 하나가 붙을 수 있다.
    if is_letter(first) {
        return run(chars, is_letter);
    }
    if !is_newline(first) && !is_number(first) {
        let letters = run(&chars[1..], is_letter);
        if letters > 0 {
            return 1 + letters;
        }
    }

    // 3. 숫자는 세 자리까지만 한 조각이 된다.
    if is_number(first) {
        return run(chars, is_number).min(3);
    }

    // 4. 그 밖의 것들. 앞에 공백 하나, 뒤에 줄바꿈들이 붙을 수 있다.
    let lead = usize::from(first == ' ');
    let body = run(&chars[lead..], is_other);
    if body > 0 {
        let end = lead + body;
        return end + run(&chars[end..], is_newline);
    }

    // 5. 공백들 가운데 마지막 줄바꿈까지.
    let spaces = run(chars, is_space);
    if let Some(last_newline) = chars[..spaces].iter().rposition(|&(_, c)| is_newline(c)) {
        return last_newline + 1;
    }

    // 6과 7. 줄바꿈이 없는 공백들.
    match trailing_space_run(chars) {
        0 => spaces,
        kept => kept,
    }
}
// ANCHOR_END: llama3

#[cfg(test)]
mod tests {
    use super::PreTokenizer::{Gpt2, Llama3};

    #[test]
    fn pieces_always_join_back_into_the_input() {
        let text = "Hello,  y'all!\n\n 12345 한글과 English 🦙.cpp\t\n  끝 ";
        for rule in [Gpt2, Llama3] {
            assert_eq!(rule.split(text).concat(), text);
        }
        assert!(Llama3.split("").is_empty());
    }

    #[test]
    fn a_leading_space_sticks_to_the_word() {
        assert_eq!(Gpt2.split("Hello world"), ["Hello", " world"]);
        assert_eq!(Llama3.split("Hello world"), ["Hello", " world"]);
        // 공백이 둘이면 하나만 단어에 붙는다.
        assert_eq!(Llama3.split("a  b"), ["a", " ", " b"]);
        assert_eq!(Llama3.split("   Hello"), ["  ", " Hello"]);
    }

    #[test]
    fn llama3_splits_digits_in_threes() {
        assert_eq!(Llama3.split("1234567"), ["123", "456", "7"]);
        assert_eq!(Gpt2.split("1234567"), ["1234567"]);
        // GPT-2는 숫자 앞의 공백을 붙이고, Llama 3는 붙이지 않는다.
        assert_eq!(Gpt2.split("a 42"), ["a", " 42"]);
        assert_eq!(Llama3.split("a 42"), ["a", " ", "42"]);
    }

    #[test]
    fn contractions_and_punctuation() {
        assert_eq!(Llama3.split("y'all don't"), ["y", "'all", " don", "'t"]);
        assert_eq!(Llama3.split("I'LL"), ["I", "'LL"]);
        assert_eq!(Gpt2.split("I'LL"), ["I", "'", "LL"]);
        assert_eq!(Llama3.split("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(Llama3.split("a !!! b"), ["a", " !!!", " b"]);
    }

    #[test]
    fn llama3_newline_rules() {
        // 기호 뒤의 줄바꿈은 기호에 붙는다.
        assert_eq!(Llama3.split("end.\n\nNext"), ["end", ".\n\n", "Next"]);
        // 공백과 줄바꿈이 섞이면 마지막 줄바꿈까지가 한 조각이다.
        assert_eq!(Llama3.split("a \n \n  b"), ["a", " \n \n", " ", " b"]);
        assert_eq!(Llama3.split("\n ="), ["\n", " ="]);
    }

    #[test]
    fn non_latin_scripts() {
        assert_eq!(Llama3.split("한글 처리"), ["한글", " 처리"]);
        // 결합 기호는 글자가 아니므로 조각을 끊는다.
        assert_eq!(Llama3.split("កា"), ["ក", "ា"]);
    }
}
