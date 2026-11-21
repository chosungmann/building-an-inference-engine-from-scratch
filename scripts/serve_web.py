"""48장과 49장: 브라우저 데모와 측정을 위한 작은 서버.

실행:
    sh scripts/build_web.sh target/web_site
    python3 scripts/serve_web.py [--isolate] [--port 8048] [--reports 폴더]

target/web_site/ 를 내주고, /models/ 아래로는 models/ 폴더를,
/third_party/ 아래로는 third_party/ 폴더를 내준다(50장에서 견줄 엔진들).

POST /served 는 지난번에 물은 뒤로 내준 파일의 길과 바이트 수(그대로의
크기와 gzip으로 누른 크기)를 JSON으로 돌려준다. 엔진 하나가 무엇을 얼마나 받아 갔는지 재는 데 쓴다.

--isolate 를 주면 응답마다 헤더 둘을 붙여 페이지를 "교차 출처 격리"한다.
브라우저가 스레드(공유 메모리)를 허락하는 조건이다. 주지 않으면 GitHub
Pages처럼 헤더 없이 내준다. 그때는 페이지의 서비스 워커가 헤더를 붙인다.

POST /report?name=이름 으로 온 글은 --reports 폴더의 이름.txt 에 적는다.
측정 페이지가 결과를 이리로 보낸다.

자기 컴퓨터 안에서만 쓰는 서버다. 127.0.0.1 에서만 듣는다.
"""

import argparse
import gzip
import json
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]


def handler(site, models, reports, isolate, third_party):
    served = {}
    squeezed = {}
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {
            **SimpleHTTPRequestHandler.extensions_map,
            ".wasm": "application/wasm",
            ".js": "text/javascript",
            ".mjs": "text/javascript",
        }

        def translate_path(self, path):
            path = urlparse(path).path
            if path.startswith("/models/"):
                base, rest = models, path[len("/models/") :]
            elif path.startswith("/third_party/"):
                base, rest = third_party, path[len("/third_party/") :]
            else:
                base, rest = site, path.lstrip("/")
            target = (base / rest).resolve()
            # 내주기로 한 폴더 밖은 내주지 않는다.
            return str(target if target.is_relative_to(base.resolve()) else base / "없음")

        def end_headers(self):
            if isolate:
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_head(self):
            file = super().send_head()
            # 내주기로 한 파일의 크기를 적어 둔다.
            if file is not None and hasattr(file, "fileno"):
                size = os.fstat(file.fileno()).st_size
                key = urlparse(self.path).path
                # 코드(.js, .mjs, .wasm)는 눌렀을 때의 크기도 적는다. 실제
                # 서버는 이런 파일을 눌러서 보낸다. 모델은 눌러도 거의 줄지
                # 않으므로 세지 않는다.
                packed = size
                if key.endswith((".js", ".mjs", ".wasm")):
                    path = self.translate_path(self.path)
                    stamp = (path, os.path.getmtime(path))
                    if stamp not in squeezed:
                        squeezed[stamp] = len(gzip.compress(Path(path).read_bytes(), 6))
                    packed = squeezed[stamp]
                with lock:
                    before = served.get(key, [0, 0])
                    served[key] = [before[0] + size, before[1] + packed]
            return file

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/served":
                with lock:
                    body = json.dumps(served).encode()
                    served.clear()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
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
    parser.add_argument("--third-party", type=Path, default=ROOT / "third_party")
    options = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", options.port), handler(options.site, options.models, options.reports, options.isolate, options.third_party)
    )
    print(f"http://127.0.0.1:{options.port}/ ({'격리' if options.isolate else '헤더 없음'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
