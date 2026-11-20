"""41장: 문법에 묶어 쓴 답이 스키마에 맞는지를 Python의 jsonschema로 다시 본다.

두 가지로 쓴다.

    python reference/ch41_validate.py file bench/ch41/answers_smollm2_135m.jsonl

`ch41_json` 예제가 적어 둔 답들을 검사한다. badak의 기계가 "맞다"고 한
것을 다른 구현이 다시 판단한다.

    python reference/ch41_validate.py server http://127.0.0.1:8041/v1

OpenAI의 공식 클라이언트로 `response_format`에 스키마를 실어 서버를
부르고, 돌아온 답을 검사한다. badak의 서버와 llama.cpp의 서버에 같은
요청을 보낼 수 있다.
"""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jsonschema

SCHEMA = json.loads((Path(__file__).parent / "ch41_schema.json").read_text())
CITIES = [
    "Paris", "Tokyo", "Nairobi", "Lima", "Sydney", "Toronto", "Cairo", "Seoul", "Berlin", "Mumbai",
    "Lagos", "Buenos Aires", "Rome", "Bangkok", "Mexico City", "Oslo", "Istanbul", "Auckland",
    "Chicago", "Cape Town",
]


def valid(text):
    """글이 JSON이고 스키마에 맞는가."""
    try:
        jsonschema.validate(json.loads(text), SCHEMA, cls=jsonschema.Draft202012Validator)
        return True
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return False


def from_file(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    free = sum(valid(row["free"]) for row in rows)
    bound = sum(valid(row["bound"]) for row in rows)
    print(f"jsonschema {version('jsonschema')}: 답 {len(rows)}개 가운데 스키마에 맞는 것")
    print(f"  그냥 쓴 답 {free}개, 문법에 묶어 쓴 답 {bound}개")
    return bound == len(rows)


def from_server(base):
    import openai

    client = openai.OpenAI(base_url=base, api_key="none")
    model = client.models.list().data[0].id
    passed, tokens, seconds = 0, 0, 0.0
    for index, city in enumerate(CITIES):
        question = (
            f"Describe the city of {city} as a JSON object with the fields city, country, "
            "continent, is_capital, population_millions and languages. Reply with JSON only."
        )
        start = time.perf_counter()
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=96,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA}},
        )
        seconds += time.perf_counter() - start
        text = reply.choices[0].message.content
        tokens += reply.usage.completion_tokens
        passed += valid(text)
        if index < 2:
            print(f"  {city}: {text}")
    print(f"openai {version('openai')}, 모델 {model}: 답 {len(CITIES)}개 가운데 스키마에 맞는 것 {passed}개")
    print(f"  토큰 {tokens}개, 요청 하나에 {seconds / len(CITIES) * 1e3:.0f}ms")
    return passed == len(CITIES)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("file", "server"):
        sys.exit(__doc__)
    ok = from_file(sys.argv[2]) if sys.argv[1] == "file" else from_server(sys.argv[2])
    sys.exit(0 if ok else 1)
