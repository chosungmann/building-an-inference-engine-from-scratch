//! 대화를 모델이 학습한 형식의 글로 바꾼다.

use std::path::Path;

use badak_formats::json;

use super::{TokenId, Tokenizer};

// ANCHOR: message
/// 대화의 한 노드.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// 누가 한 말인지. 보통 `system`, `user`, `assistant` 가운데 하나다.
    pub role: String,
    /// 말의 내용.
    pub content: String,
}

impl Message {
    /// 역할과 내용으로 한 노드를 만든다.
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// 대화를 글로 적는 형식. 모델마다 학습할 때 쓴 형식이 정해져 있다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// Llama 3 계열의 형식.
    Llama3,
    /// ChatML. SmolLM2, Qwen 등 여러 모델이 쓴다.
    ChatMl,
}
// ANCHOR_END: message

impl ChatFormat {
    /// 토크나이저가 아는 특수 토큰으로 형식을 알아본다. `<|im_start|>`가
    /// 토큰 하나이면 ChatML이고, 아니면 Llama 3의 형식이다.
    pub fn detect(tokenizer: &Tokenizer) -> Self {
        if tokenizer.token_id("<|im_start|>").is_some() {
            Self::ChatMl
        } else {
            Self::Llama3
        }
    }

    // ANCHOR: render
    /// 대화를 글로 적는다.
    ///
    /// `add_generation_prompt`가 참이면 끝에 "이제 assistant가 말할
    /// 차례"라는 표시를 붙인다. 모델에게 답을 시킬 때는 참으로 둔다.
    ///
    /// 결과에는 특수 토큰의 표기가 들어 있으므로, 토큰으로 바꿀 때
    /// `parse_special`을 참으로 주어야 한다.
    pub fn render(self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut text = String::new();
        match self {
            Self::Llama3 => {
                for message in messages {
                    text.push_str("<|start_header_id|>");
                    text.push_str(&message.role);
                    text.push_str("<|end_header_id|>\n\n");
                    text.push_str(message.content.trim());
                    text.push_str("<|eot_id|>");
                }
                if add_generation_prompt {
                    text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            }
            Self::ChatMl => {
                for message in messages {
                    text.push_str("<|im_start|>");
                    text.push_str(&message.role);
                    text.push('\n');
                    text.push_str(&message.content);
                    text.push_str("<|im_end|>\n");
                }
                if add_generation_prompt {
                    text.push_str("<|im_start|>assistant\n");
                }
            }
        }
        text
    }
    // ANCHOR_END: render

    // ANCHOR: encode
    /// 대화를 토큰으로 바꾼다. [`render`](Self::render)로 글을 만들어
    /// 통째로 토큰화하는 것과 달리, **틀의 뼈대와 말의 내용을 따로**
    /// 토큰화한다 (39장).
    ///
    /// 뼈대(`<|im_start|>` 같은 것)는 특수 토큰으로 읽고, 내용은 보통의
    /// 글로만 읽는다. 사용자가 내용에 `<|im_end|>`라고 적어도 그것은 글자
    /// 열 개일 뿐 대화의 경계가 되지 않는다. 8.7절에서 본 문제를 막는
    /// 방법이다.
    ///
    /// 역할의 이름은 뼈대의 일부다. 부르는 쪽이 `system`, `user`,
    /// `assistant`처럼 정해진 것만 넘겨야 한다.
    pub fn encode(
        self,
        tokenizer: &Tokenizer,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Vec<TokenId> {
        let mut ids = tokenizer.prefix().to_vec();
        let frame = |ids: &mut Vec<TokenId>, text: &str| ids.extend(tokenizer.encode(text, true));
        let (open, middle, close, trim) = match self {
            Self::Llama3 => (
                "<|start_header_id|>",
                "<|end_header_id|>\n\n",
                "<|eot_id|>",
                true,
            ),
            Self::ChatMl => ("<|im_start|>", "\n", "<|im_end|>\n", false),
        };
        for message in messages {
            frame(&mut ids, &format!("{open}{}{middle}", message.role));
            let content = if trim {
                message.content.trim()
            } else {
                &message.content
            };
            ids.extend(tokenizer.encode(content, false));
            frame(&mut ids, close);
        }
        if add_generation_prompt {
            frame(&mut ids, &format!("{open}assistant{middle}"));
        }
        ids
    }
    // ANCHOR_END: encode
}

// ANCHOR: template
/// 모델 하나의 대화 틀: 형식과, 시스템의 말이 없을 때 넣는 기본 지시문.
///
/// [`ChatFormat`]은 틀의 뼈대만 안다. 모델에 따라서는 대화에 시스템의 말이
/// 없을 때 **정해진 한 줄을 대신 넣으라**고 틀에 적혀 있고, 모델은 그 줄이
/// 늘 있는 대화로 학습되었다(47.3절). 그 줄까지 아는 것이 이 타입이다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplate {
    /// 틀의 뼈대.
    pub format: ChatFormat,
    /// 대화가 시스템의 말로 시작하지 않을 때 맨 앞에 넣는 말.
    pub default_system: Option<String>,
}

impl ChatTemplate {
    /// 모델의 디렉터리에서 틀을 읽는다.
    ///
    /// 형식은 토크나이저의 특수 토큰으로 알아보고, 기본 지시문은
    /// `tokenizer_config.json`의 `chat_template`에서 찾는다. 그 파일이
    /// 없거나 틀에 기본 지시문이 없으면 뼈대만 쓴다.
    pub fn load(directory: impl AsRef<Path>, tokenizer: &Tokenizer) -> Self {
        let format = ChatFormat::detect(tokenizer);
        let default_system =
            std::fs::read_to_string(directory.as_ref().join("tokenizer_config.json"))
                .ok()
                .and_then(|text| json::parse(&text).ok())
                .and_then(|config| {
                    let template = config.get("chat_template")?.as_str()?;
                    default_system(format, template)
                });
        Self {
            format,
            default_system,
        }
    }

    /// 기본 지시문이 없는 틀. [`ChatFormat`]을 그대로 쓰는 것과 같다.
    pub fn bare(format: ChatFormat) -> Self {
        Self {
            format,
            default_system: None,
        }
    }

    /// 대화가 시스템의 말로 시작하지 않으면 기본 지시문을 앞에 놓는다.
    fn complete(&self, messages: &[Message]) -> Vec<Message> {
        let starts_with_system = messages.first().is_some_and(|first| first.role == "system");
        let mut all = Vec::with_capacity(messages.len() + 1);
        if let (Some(system), false) = (&self.default_system, starts_with_system) {
            all.push(Message::new("system", system));
        }
        all.extend_from_slice(messages);
        all
    }

    /// 대화를 글로 적는다. [`ChatFormat::render`]에 기본 지시문을 더한 것.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        self.format
            .render(&self.complete(messages), add_generation_prompt)
    }

    /// 대화를 토큰으로 바꾼다. [`ChatFormat::encode`]에 기본 지시문을 더한 것.
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Vec<TokenId> {
        self.format
            .encode(tokenizer, &self.complete(messages), add_generation_prompt)
    }
}

/// 틀(Jinja로 적힌 글)에서 기본 지시문을 찾는다.
///
/// Jinja를 해석하지 않는다. **한 가지 꼴만 알아본다.** "첫 말이 시스템의
/// 것이 아니면"이라는 조건(`messages[0]['role'] != 'system'`) 바로 뒤에
/// 시스템의 말 하나가 통째로 글자 그대로 적혀 있는 꼴이다. SmolLM2의 틀이
/// 이렇게 생겼다. 다른 꼴이면 `None`을 돌려준다. 틀린 지시문을 지어내는
/// 것보다 넣지 않는 편이 낫다.
fn default_system(format: ChatFormat, template: &str) -> Option<String> {
    let (open, close) = match format {
        ChatFormat::ChatMl => ("<|im_start|>system", "<|im_end|>"),
        ChatFormat::Llama3 => return None,
    };
    let condition = template.find("messages[0]['role'] != 'system'")?;
    let after = &template[condition..];
    // 조건이 닫히기 전에 나오는 것만 본다.
    let branch = &after[..after.find("{% endif %}")?];
    let start = branch.find(open)? + open.len();
    let end = start + branch[start..].find(close)?;
    // 역할 뒤의 줄바꿈은 틀에 따라 진짜 줄바꿈이기도, `\n`이라는 두 글자이기도 하다.
    let text = &branch[start..end];
    let text = text
        .strip_prefix('\n')
        .or_else(|| text.strip_prefix("\\n"))?;
    // 변수를 끼워 넣는 틀이면(`' + date + '`) 글자 그대로가 아니다.
    if text.is_empty() || text.contains(['\'', '{', '}']) {
        return None;
    }
    Some(text.replace("\\n", "\n"))
}
// ANCHOR_END: template

#[cfg(test)]
mod tests {
    use super::*;

    /// SmolLM2-135M-Instruct(버전 `12fd25f77366`)의 `tokenizer_config.json`에 든 틀.
    const SMOLLM2: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn the_default_system_message_is_read_from_the_template() {
        assert_eq!(
            default_system(ChatFormat::ChatMl, SMOLLM2).as_deref(),
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
        );
        // 조건이 없는 틀, 조건 뒤에 글자 그대로의 말이 없는 틀에서는 찾지 않는다.
        let plain = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% endfor %}";
        assert_eq!(default_system(ChatFormat::ChatMl, plain), None);
        let variable = "{% if messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nToday is ' + date + '<|im_end|>' }}{% endif %}";
        assert_eq!(default_system(ChatFormat::ChatMl, variable), None);
        assert_eq!(default_system(ChatFormat::Llama3, SMOLLM2), None);
        // 줄바꿈을 두 글자(`\n`)로 적은 틀도 같은 말을 낸다.
        assert_eq!(
            default_system(ChatFormat::ChatMl, &SMOLLM2.replace('\n', "\\n")),
            default_system(ChatFormat::ChatMl, SMOLLM2)
        );
    }

    #[test]
    fn the_default_is_used_only_when_no_system_message_is_given() {
        let template = ChatTemplate {
            format: ChatFormat::ChatMl,
            default_system: default_system(ChatFormat::ChatMl, SMOLLM2),
        };
        // transformers 5.23.0의 `apply_chat_template`이 같은 대화에 내는 글이다
        // (`reference/ch47_chat_template.py`).
        assert_eq!(
            template.render(&[Message::new("user", "Hi")], true),
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert_eq!(
            template.render(&conversation(), true),
            ChatFormat::ChatMl.render(&conversation(), true)
        );
        assert_eq!(
            ChatTemplate::bare(ChatFormat::ChatMl).render(&[Message::new("user", "Hi")], true),
            ChatFormat::ChatMl.render(&[Message::new("user", "Hi")], true)
        );
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message::new("system", "You are helpful."),
            Message::new("user", " 안녕? "),
        ]
    }

    #[test]
    fn llama3_format() {
        assert_eq!(
            ChatFormat::Llama3.render(&conversation(), true),
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n안녕?<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn chatml_format() {
        assert_eq!(
            ChatFormat::ChatMl.render(&conversation(), true),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\n 안녕? <|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert!(
            ChatFormat::ChatMl
                .render(&conversation(), false)
                .ends_with("<|im_end|>\n")
        );
    }
}
