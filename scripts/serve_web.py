"""48장과 49장: 브라우저 데모와 측정을 위한 작은 서버.

실행:
    sh scripts/build_web.sh target/web_site
    python3 scripts/serve_web.py [--isolate] [--port 8048] [--reports 폴더]

target/web_site/ 를 내주고, /models/ 아래로는 models/ 폴더를 내준다.

--isolate 를 주면 응답마다 헤더 둘을 붙여 페이지를 "교차 출처 격리"한다.
브라우저가 스레드(공유 메모리)를 허락하는 조건이다. 주지 않으면 GitHub
Pages처럼 헤더 없이 내준다. 그때는 페이지의 서비스 워커가 헤더를 붙인다.

POST /report?name=이름 으로 온 글은 --reports 폴더의 이름.txt 에 적는다.
측정 페이지가 결과를 이리로 보낸다.

자기 컴퓨터 안에서만 쓰는 서버다. 127.0.0.1 에서만 듣는다.
"""

import argparse
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]


def handler(site, models, reports, isolate):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {
            **SimpleHTTPRequestHandler.extensions_map,
            ".wasm": "application/wasm",
            ".js": "text/javascript",
            ".mjs": "text/javascript",
        }

        def translate_path(self, path):
            path = urlparse(path).path
            base, rest = (models, path[len("/models/") :]) if path.startswith("/models/") else (site, path.lstrip("/"))
            target = (base / rest).resolve()
            # 내주기로 한 폴더 밖은 내주지 않는다.
            return str(target if target.is_relative_to(base.resolve()) else base / "없음")

        def end_headers(self):
            if isolate:
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_POST(self):
            url = urlparse(self.path)
            name = parse_qs(url.query).get("name", [""])[0]
            if url.path != "/report" or not re.fullmatch(r"[a-z0-9_]+", name):
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            reports.mkdir(parents=True, exist_ok=True)
            (reports / f"{name}.txt").write_bytes(body)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *arguments):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--site", type=Path, default=ROOT / "target/web_site")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--reports", type=Path, default=ROOT / "target/web_reports")
    options = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", options.port), handler(options.site, options.models, options.reports, options.isolate)
    )
    print(f"http://127.0.0.1:{options.port}/ ({'격리' if options.isolate else '헤더 없음'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
