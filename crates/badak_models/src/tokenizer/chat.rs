//! 대화를 모델이 학습한 형식의 글로 바꾼다.

// ANCHOR: message
/// 대화의 한 마디.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// 누가 한 말인지. 보통 `system`, `user`, `assistant` 가운데 하나다.
    pub role: String,
    /// 말의 내용.
    pub content: String,
}

impl Message {
    /// 역할과 내용으로 한 마디를 만든다.
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
