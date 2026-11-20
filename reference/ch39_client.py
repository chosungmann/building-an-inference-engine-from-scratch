"""39장: OpenAI의 공식 클라이언트가 badak의 서버를 그대로 쓸 수 있는지 본다.

먼저 서버를 띄운다.

    cargo run --release -p badak_cli -- serve --port 8039

그다음 이 파일을 돌린다.

    python ch39_client.py [http://127.0.0.1:8039/v1]

클라이언트는 주소만 바꿨다. 열쇠는 아무 글이나 준다(이 서버는 보지
않는다). 응답을 클라이언트가 제 형식(pydantic 모델)으로 해석하므로,
꼴이 어긋나면 여기서 오류가 난다.
"""

import sys

import openai

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8039/v1"
client = openai.OpenAI(base_url=BASE, api_key="none")

print("openai", openai.__version__)
model = client.models.list().data[0].id
print("모델:", model)

messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "What is the capital of France?"},
]

# 1. 한 번에 받기.
whole = client.chat.completions.create(model=model, messages=messages, max_tokens=32, temperature=0)
answer = whole.choices[0].message.content
print("한 번에:", repr(answer))
print("  끝난 까닭:", whole.choices[0].finish_reason)
print("  토큰: 읽음", whole.usage.prompt_tokens, "씀", whole.usage.completion_tokens,
      "캐시", whole.usage.prompt_tokens_details.cached_tokens)

# 2. 조각으로 받기. 둘째 차례를 이어서 묻는다.
messages += [
    {"role": "assistant", "content": answer},
    {"role": "user", "content": "And what river runs through it?"},
]
pieces, usage, reason = [], None, None
stream = client.chat.completions.create(
    model=model, messages=messages, max_tokens=32, temperature=0, stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        pieces.append(chunk.choices[0].delta.content)
    if chunk.choices and chunk.choices[0].finish_reason:
        reason = chunk.choices[0].finish_reason
    if chunk.usage:
        usage = chunk.usage
print("조각으로:", repr("".join(pieces)), f"({len(pieces)}조각)")
print("  끝난 까닭:", reason)
print("  토큰: 읽음", usage.prompt_tokens, "씀", usage.completion_tokens,
      "캐시", usage.prompt_tokens_details.cached_tokens)

# 3. 틀린 요청은 클라이언트의 예외가 된다.
try:
    client.chat.completions.create(model=model, messages=[], max_tokens=4)
except openai.BadRequestError as error:
    print("빈 대화:", type(error).__name__, error.status_code, error.body["message"])
try:
    client.chat.completions.create(model=model, messages=messages, max_tokens=4, temperature=9)
except openai.BadRequestError as error:
    print("온도 9:", type(error).__name__, error.status_code, error.body["message"])
