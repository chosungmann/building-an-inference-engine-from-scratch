//! CLIP의 토크나이저 (35장).
//!
//! 8장의 바이트 단위 BPE와 같은 계열이고 세 가지가 다르다.
//!
//! - 글을 먼저 소문자로 바꾸고 공백을 정리한다.
//! - 공백을 토큰에 남기지 않는다. 대신 단어의 **마지막** 조각에 `</w>`를
//!   붙여 단어가 끝났음을 적는다.
//! - 결과를 늘 같은 길이(77)로 맞춘다. 모자라면 끝 토큰으로 채운다.

use std::collections::HashMap;
use std::path::Path;

use badak_formats::Result;
use badak_formats::json::{self, Value};

use super::unicode::{byte_to_char, is_letter, is_number};
use super::{TokenId, invalid};

/// 단어의 끝을 적는 표시.
const END_OF_WORD: &str = "</w>";

// ANCHOR: clip
/// CLIP의 토크나이저.
pub struct ClipTokenizer {
    /// 조각에서 번호로.
    ids: HashMap<String, TokenId>,
    /// 병합 규칙. 이웃한 두 조각에서 우선순위로. 작을수록 먼저 합친다.
    ranks: HashMap<(String, String), usize>,
    /// 글의 시작 `<|startoftext|>`.
    start: TokenId,
    /// 글의 끝 `<|endoftext|>`. 모르는 조각과 채우는 자리에도 쓴다.
    end: TokenId,
}
// ANCHOR_END: clip

impl ClipTokenizer {
    /// 디렉터리에서 `vocab.json`과 `merges.txt`를 읽는다.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let vocab = json::parse(&std::fs::read_to_string(directory.join("vocab.json"))?)?;
        let ids: HashMap<String, TokenId> = vocab
            .as_object()
            .ok_or_else(|| invalid("vocab.json이 객체가 아닙니다"))?
            .iter()
            .filter_map(|(token, id)| Some((token.clone(), Value::as_usize(id)? as TokenId)))
            .collect();
        // merges.txt: 첫 줄은 설명이고, 그 뒤로 한 줄에 "왼쪽 오른쪽"이다.
        // 위에 있을수록 먼저 합친다.
        let merges = std::fs::read_to_string(directory.join("merges.txt"))?;
        let ranks = merges
            .lines()
            .skip(1)
            .filter_map(|line| line.split_once(' '))
            .enumerate()
            .map(|(rank, (left, right))| ((left.to_owned(), right.to_owned()), rank))
            .collect();
        let special = |name: &str| {
            ids.get(name)
                .copied()
                .ok_or_else(|| invalid(format!("단어 사전에 {name}가 없습니다")))
        };
        Ok(Self {
            start: special("<|startoftext|>")?,
            end: special("<|endoftext|>")?,
            ids,
            ranks,
        })
    }

    /// 단어 사전의 크기.
    pub fn vocab_size(&self) -> usize {
        self.ids.len()
    }

    /// 글의 끝 토큰의 번호.
    pub fn end(&self) -> TokenId {
        self.end
    }

    // ANCHOR: words
    /// 글을 소문자로 바꾸고 단어로 자른다.
    ///
    /// 글자가 이어진 것이 한 단어, 숫자는 하나하나가 단어, 그 밖의 기호는
    /// 이어진 것이 한 단어다. `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`는
    /// 따로 뗀다. 공백은 버린다.
    pub fn words(text: &str) -> Vec<String> {
        let characters: Vec<char> = text.chars().flat_map(char::to_lowercase).collect();
        let mut words = Vec::new();
        let mut at = 0;
        while at < characters.len() {
            let character = characters[at];
            let rest = &characters[at..];
            let length = if character.is_whitespace() {
                at += 1;
                continue;
            } else if let Some(suffix) = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"]
                .iter()
                .find(|suffix| rest.starts_with(&suffix.chars().collect::<Vec<_>>()))
            {
                suffix.len()
            } else if is_letter(character) {
                rest.iter().take_while(|&&c| is_letter(c)).count()
            } else if is_number(character) {
                1
            } else {
                rest.iter()
                    .take_while(|&&c| !c.is_whitespace() && !is_letter(c) && !is_number(c))
                    .count()
            };
            words.push(rest[..length].iter().collect());
            at += length;
        }
        words
    }
    // ANCHOR_END: words

    // ANCHOR: bpe
    /// 단어 하나를 조각들로 합친다. 8장의 BPE와 같은 반복이고, 출발점의
    /// 마지막 글자에 `</w>`가 붙어 있다는 것만 다르다.
    pub fn bpe(&self, word: &str, out: &mut Vec<TokenId>) {
        let mut symbols: Vec<String> = word
            .bytes()
            .map(|byte| byte_to_char(byte).to_string())
            .collect();
        let Some(last) = symbols.last_mut() else {
            return;
        };
        last.push_str(END_OF_WORD);
        loop {
            // 이웃한 쌍 가운데 우선순위가 가장 높은 것.
            let best = symbols
                .windows(2)
                .filter_map(|pair| {
                    let key = (pair[0].clone(), pair[1].clone());
                    self.ranks.get(&key).map(|&rank| (rank, key))
                })
                .min();
            let Some((_, (left, right))) = best else {
                break;
            };
            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(std::mem::take(&mut symbols[index]));
                    index += 1;
                }
            }
            symbols = merged;
        }
        out.extend(
            symbols
                .iter()
                .map(|symbol| self.ids.get(symbol).copied().unwrap_or(self.end)),
        );
    }
    // ANCHOR_END: bpe

    // ANCHOR: encode
    /// 글을 정확히 `length`개의 토큰 번호로 바꾼다. 시작 토큰, 글, 끝
    /// 토큰의 순서이고, 넘치면 글을 자르고 모자라면 끝 토큰으로 채운다.
    pub fn encode(&self, text: &str, length: usize) -> Vec<TokenId> {
        let mut ids = vec![self.start];
        for word in Self::words(text) {
            self.bpe(&word, &mut ids);
        }
        ids.truncate(length.max(2) - 1);
        ids.resize(length.max(2), self.end);
        ids
    }
    // ANCHOR_END: encode
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_the_reference_pattern() {
        assert_eq!(
            ClipTokenizer::words("A photo of 2 cats, isn't it?!"),
            [
                "a", "photo", "of", "2", "cats", ",", "isn", "'t", "it", "?!"
            ]
        );
        assert_eq!(ClipTokenizer::words("42  km"), ["4", "2", "km"]);
        assert!(ClipTokenizer::words(" \n ").is_empty());
    }

    #[test]
    fn merges_by_rank_and_marks_the_end_of_a_word() {
        let ids = [
            "<|startoftext|>",
            "<|endoftext|>",
            "c",
            "a",
            "t</w>",
            "ca",
            "cat</w>",
            "s</w>",
        ]
        .iter()
        .enumerate()
        .map(|(id, token)| ((*token).to_owned(), id as TokenId))
        .collect();
        let ranks = [("c", "a"), ("ca", "t</w>")]
            .iter()
            .enumerate()
            .map(|(rank, (left, right))| (((*left).to_owned(), (*right).to_owned()), rank))
            .collect();
        let tokenizer = ClipTokenizer {
            ids,
            ranks,
            start: 0,
            end: 1,
        };
        let mut out = Vec::new();
        tokenizer.bpe("cat", &mut out);
        assert_eq!(out, [6]);
        // 단어의 끝이 아닌 t는 사전에 없다. 모르는 조각은 끝 토큰이 된다.
        out.clear();
        tokenizer.bpe("cats", &mut out);
        assert_eq!(out, [5, 1, 7]);
        assert_eq!(tokenizer.encode("cat cat", 6), [0, 6, 6, 1, 1, 1]);
        assert_eq!(tokenizer.encode("cat cat cat", 3), [0, 6, 1]);
    }
}
