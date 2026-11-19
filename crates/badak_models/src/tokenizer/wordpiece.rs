//! WordPiece 토크나이저 (34장).
//!
//! BERT와 그 뒤를 이은 인코더 모델들이 쓴다. 8장의 BPE와 견주면 다음이
//! 다르다.
//!
//! - 글을 먼저 고친다. 소문자로 바꾸고 악센트를 지운다(모델에 따라 다르다).
//! - 단어를 공백과 문장 부호에서 자른다. 공백은 토큰에 남지 않는다.
//! - 단어 하나를 앞에서부터 **가장 긴 조각**으로 잘라 나간다. 병합 규칙의
//!   순서를 따라가지 않는다. 단어 사전만 있으면 된다.
//! - 단어의 첫 조각이 아닌 조각은 `##`로 시작한다.

use std::collections::HashMap;
use std::path::Path;

use badak_formats::Result;

use super::unicode::{decompose, is_nonspacing_mark, is_other, is_punctuation};
use super::{TokenId, invalid};

/// 글자가 이보다 많은 단어는 자르지 않고 모르는 단어로 친다.
const LONGEST_WORD: usize = 100;

// ANCHOR: wordpiece
/// WordPiece 토크나이저.
pub struct WordPiece {
    /// 번호에서 조각으로.
    tokens: Vec<String>,
    /// 조각에서 번호로.
    ids: HashMap<String, TokenId>,
    /// 모르는 단어. `[UNK]`.
    unknown: TokenId,
    /// 글의 맨 앞에 붙이는 토큰. `[CLS]`.
    class: TokenId,
    /// 글의 끝에 붙이는 토큰. `[SEP]`.
    separator: TokenId,
    /// 길이를 맞추려고 채우는 토큰. `[PAD]`.
    padding: TokenId,
}
// ANCHOR_END: wordpiece

impl WordPiece {
    // ANCHOR: load
    /// `vocab.txt`를 읽는다. 한 줄에 조각 하나이고 줄 번호가 토큰 번호다.
    pub fn from_vocab(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let tokens: Vec<String> = text.lines().map(str::to_owned).collect();
        let ids: HashMap<String, TokenId> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as TokenId))
            .collect();
        let special = |name: &str| {
            ids.get(name)
                .copied()
                .ok_or_else(|| invalid(format!("단어 사전에 {name}가 없습니다")))
        };
        Ok(Self {
            unknown: special("[UNK]")?,
            class: special("[CLS]")?,
            separator: special("[SEP]")?,
            padding: special("[PAD]")?,
            tokens,
            ids,
        })
    }
    // ANCHOR_END: load

    /// 단어 사전의 크기.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// 길이를 맞추려고 채우는 토큰의 번호.
    pub fn padding(&self) -> TokenId {
        self.padding
    }

    /// 토큰 번호의 조각.
    pub fn token_text(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    // ANCHOR: normalize
    /// 글을 자르기 전에 고친다. 소문자만 쓰는(uncased) BERT의 방식이다.
    ///
    /// 1. 제어 문자를 지우고 온갖 공백을 빈칸 하나로 바꾼다.
    /// 2. 한자는 글자마다 양옆에 빈칸을 넣는다. 글자 하나가 단어 하나가 된다.
    /// 3. 글자를 바탕 글자와 결합 기호로 풀고 결합 기호를 지운다. `é`가
    ///    `e`가 된다.
    /// 4. 소문자로 바꾼다.
    pub fn normalize(text: &str) -> String {
        let mut cleaned = Vec::with_capacity(text.len());
        for character in text.chars() {
            let space = matches!(character, '\t' | '\n' | '\r') || character.is_whitespace();
            if space {
                cleaned.push(' ');
            } else if character == '\u{fffd}' || is_other(character) {
                // 지운다. 탭과 줄 바꿈은 제어 문자지만 위에서 공백으로 쳤다.
            } else if is_han(character) {
                cleaned.extend([' ', character, ' ']);
            } else {
                cleaned.push(character);
            }
        }
        let mut decomposed = Vec::with_capacity(cleaned.len());
        for character in cleaned {
            decompose(character, &mut decomposed);
        }
        decomposed
            .into_iter()
            .filter(|&character| !is_nonspacing_mark(character))
            .flat_map(char::to_lowercase)
            .collect()
    }
    // ANCHOR_END: normalize

    // ANCHOR: words
    /// 고친 글을 단어로 자른다. 공백에서 자르고 공백은 버린다. 문장 부호는
    /// 하나하나가 단어다.
    pub fn words(normalized: &str) -> Vec<&str> {
        let mut words = Vec::new();
        let mut start = None;
        for (offset, character) in normalized.char_indices() {
            let punctuation = character.is_ascii_punctuation() || is_punctuation(character);
            if character.is_whitespace() || punctuation {
                if let Some(start) = start.take() {
                    words.push(&normalized[start..offset]);
                }
                if punctuation {
                    words.push(&normalized[offset..offset + character.len_utf8()]);
                }
            } else if start.is_none() {
                start = Some(offset);
            }
        }
        if let Some(start) = start {
            words.push(&normalized[start..]);
        }
        words
    }
    // ANCHOR_END: words

    // ANCHOR: pieces
    /// 단어 하나를 조각들로 자른다.
    ///
    /// 앞에서부터, 단어 사전에 있는 가장 긴 조각을 떼어 낸다. 첫 조각이
    /// 아니면 `##`를 붙여서 찾는다. 어느 자리에서든 맞는 조각이 없으면
    /// 단어 전체가 `[UNK]` 하나가 된다.
    pub fn pieces(&self, word: &str, out: &mut Vec<TokenId>) {
        // 글자의 경계(바이트 위치). 조각은 글자 단위로 줄여 가며 찾는다.
        let bounds: Vec<usize> = word
            .char_indices()
            .map(|(offset, _)| offset)
            .chain([word.len()])
            .collect();
        let count = bounds.len() - 1;
        if count > LONGEST_WORD {
            out.push(self.unknown);
            return;
        }
        let first = out.len();
        let mut candidate = String::with_capacity(word.len() + 2);
        let mut start = 0;
        while start < count {
            let found = (start + 1..=count).rev().find_map(|end| {
                candidate.clear();
                if start > 0 {
                    candidate.push_str("##");
                }
                candidate.push_str(&word[bounds[start]..bounds[end]]);
                self.ids.get(candidate.as_str()).map(|&id| (end, id))
            });
            let Some((end, id)) = found else {
                out.truncate(first);
                out.push(self.unknown);
                return;
            };
            out.push(id);
            start = end;
        }
    }
    // ANCHOR_END: pieces

    // ANCHOR: encode
    /// 글 하나를 토큰 번호로 바꾼다. 맨 앞에 `[CLS]`, 끝에 `[SEP]`를
    /// 붙인다. 둘을 합쳐 `limit`개를 넘으면 뒤를 자른다.
    pub fn encode(&self, text: &str, limit: usize) -> Vec<TokenId> {
        let mut ids = vec![self.class];
        let normalized = Self::normalize(text);
        for word in Self::words(&normalized) {
            self.pieces(word, &mut ids);
        }
        ids.truncate(limit.max(2) - 1);
        ids.push(self.separator);
        ids
    }
    // ANCHOR_END: encode
}

/// 한자인가. BERT가 "글자 하나가 단어 하나"로 다루는 구간이다. 한글과
/// 가나는 들어 있지 않다.
fn is_han(character: char) -> bool {
    matches!(
        character as u32,
        0x4e00..=0x9fff
            | 0x3400..=0x4dbf
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b73f
            | 0x2b740..=0x2b81f
            | 0x2b920..=0x2ceaf
            | 0xf900..=0xfaff
            | 0x2f800..=0x2fa1f
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> WordPiece {
        let tokens: Vec<String> = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "want", "##ed", ",", "a",
            "##b", "cafe", "中",
        ]
        .iter()
        .map(|&token| token.to_owned())
        .collect();
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as TokenId))
            .collect();
        WordPiece {
            tokens,
            ids,
            unknown: 1,
            class: 2,
            separator: 3,
            padding: 0,
        }
    }

    #[test]
    fn normalizes_case_accents_spaces_and_han() {
        assert_eq!(WordPiece::normalize("Café\tAU\u{0}Lait"), "cafe aulait");
        assert_eq!(WordPiece::normalize("a中b"), "a 中 b");
        // 한글은 자모로 풀린 채 남는다. 결합 기호가 아니기 때문이다.
        assert_eq!(WordPiece::normalize("가"), "\u{1100}\u{1161}");
    }

    #[test]
    fn splits_on_spaces_and_punctuation() {
        assert_eq!(
            WordPiece::words("hello, world!  it's"),
            ["hello", ",", "world", "!", "it", "'", "s"]
        );
        assert!(WordPiece::words("   ").is_empty());
    }

    #[test]
    fn takes_the_longest_piece_first() {
        let tokenizer = tiny();
        let mut out = Vec::new();
        tokenizer.pieces("unaffable", &mut out);
        assert_eq!(out, [4, 5, 6]);
        // 중간에 막히면 단어 전체가 [UNK] 하나다. 앞에서 찾은 조각도 버린다.
        out.clear();
        tokenizer.pieces("unknown", &mut out);
        assert_eq!(out, [1]);
        out.clear();
        tokenizer.pieces(&"a".repeat(101), &mut out);
        assert_eq!(out, [1]);
    }

    #[test]
    fn wraps_and_truncates() {
        let tokenizer = tiny();
        assert_eq!(
            tokenizer.encode("Wanted, Café 中", 16),
            [2, 7, 8, 9, 12, 13, 3]
        );
        assert_eq!(tokenizer.encode("wanted, cafe", 4), [2, 7, 8, 3]);
        assert_eq!(tokenizer.encode("", 16), [2, 3]);
    }
}
