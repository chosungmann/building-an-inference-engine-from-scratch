// 엔진을 품은 워커 (48장).
//
// 엔진은 페이지의 주 스레드에서 돌리지 않는다. 까닭이 둘이다. 엔진이 계산하는
// 동안 화면이 멈춘다. 그리고 주 스레드는 "기다릴" 수 없다. 스레드 풀이 일을
// 기다리며 잠드는 것을 브라우저가 주 스레드에서는 허락하지 않는다.
//
// 페이지는 이 워커에 글로 된 요청을 보내고(client.js), 워커는 답을 보낸다.

import { load } from './badak.js';

const files = new Map();
let engine = null;
let graph = null;
let llm = null;

// ANCHOR: cache
// 파일을 받는다. 받은 것은 브라우저의 캐시 저장소에 넣어 두고, 다음에는
// 거기에서 꺼낸다. 브라우저의 보통 캐시(HTTP 캐시)에 맡기지 않는 까닭은,
// 그것은 큰 파일을 말없이 버릴 수 있고 무엇이 들어 있는지 물을 수도 없기
// 때문이다.
async function fetchCached(url, sha256, onProgress) {
  const store = await caches.open('badak-models-v1');
  const hit = await store.match(url);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    // 넣어 둔 것도 믿지 않고 다시 확인한다. 틀리면 버리고 새로 받는다.
    if ((await digest(bytes)) === sha256) return { bytes, cached: true };
    await store.delete(url);
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  // 받는 대로 세어 진행 상황을 알린다.
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  // 받은 것이 받으려던 그 파일인가 (44장의 모델 저장소와 같은 확인).
  const actual = await digest(bytes);
  if (actual !== sha256) throw new Error(`${url}: 해시가 다릅니다 (${actual})`);
  // 저장소가 찼으면 넣지 못한다. 그래도 이번에는 쓸 수 있다.
  try {
    await store.put(url, new Response(bytes, { headers: { 'content-length': String(received) } }));
  } catch (error) {
    console.warn('캐시에 넣지 못했습니다:', error);
  }
  return { bytes, cached: false };
}
// ANCHOR_END: cache

// SHA-256. 브라우저에 들어 있는 것을 쓴다.
async function digest(bytes) {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(hash, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

const handlers = {
  // 엔진을 올린다.
  async start({ wasm, threads, env }) {
    const started = performance.now();
    const bytes = await (await fetch(wasm)).arrayBuffer();
    engine = await load({ wasm: bytes, files, threads, env, print: (line) => console.log(line) });
    return { backend: engine.backend(), bytes: bytes.byteLength, ms: performance.now() - started };
  },
  // 파일 하나를 받아 엔진에게 보여 준다.
  async file({ path, url, sha256 }, post) {
    const started = performance.now();
    const { bytes, cached } = await fetchCached(url, sha256, (received, total) => post({ kind: 'progress', path, received, total }));
    files.set(path, bytes);
    return { bytes: bytes.length, cached, ms: performance.now() - started };
  },
  // 엔진이 읽어 간 파일을 JavaScript 쪽에서 내려놓는다.
  forget({ path }) {
    files.delete(path);
    return {};
  },
  linear({ rows, cols, count, repeats }) {
    return engine.benchLinear(rows, cols, count, repeats);
  },
  mnist({ pixels, repeats = 1 }) {
    graph ??= engine.openGraph('/mnist/mlp.tflite');
    // 브라우저의 시계는 일부러 무디게 만들어져 있다(48장). 한 번은 너무 짧아
    // 잴 수 없으니 여러 번을 통째로 재서 나눈다.
    const started = performance.now();
    for (let index = 0; index < repeats; index++) {
      graph.input().set(pixels);
      graph.run();
    }
    return { scores: Array.from(graph.output()), ms: (performance.now() - started) / repeats };
  },
  open({ weights = 0 }) {
    const started = performance.now();
    llm ??= engine.openLlm('/smollm2_135m', weights);
    return { ms: performance.now() - started, memory: engine.memoryBytes() };
  },
  chat({ system, question, maxTokens }, post) {
    const started = performance.now();
    const times = [];
    const written = llm.chat(system, question, maxTokens, (text) => {
      times.push(performance.now() - started);
      post({ kind: 'piece', text });
    });
    return { written, first: times[0] ?? 0, total: performance.now() - started, pieces: times.length, memory: engine.memoryBytes() };
  },
  memory() {
    return { bytes: engine.memoryBytes() };
  },
  // 엔진의 메모리를 chunk 바이트씩 얻어 본다. 얻지 못하면 엔진이 죽으므로
  // 맨 마지막에만 부른다.
  exhaust({ chunk }, post) {
    let got = 0;
    try {
      for (;;) {
        if (!engine.exports.badak_web_alloc(chunk)) break;
        got += chunk;
        post({ kind: 'grown', bytes: engine.memoryBytes() });
      }
    } catch (error) {
      return { got, bytes: engine.memoryBytes(), error: String(error) };
    }
    return { got, bytes: engine.memoryBytes() };
  },
};

self.onmessage = async ({ data }) => {
  const post = (message) => self.postMessage({ ...message, id: data.id });
  try {
    post({ kind: 'done', result: await handlers[data.kind](data, post) });
  } catch (error) {
    post({ kind: 'error', message: String(error?.message ?? error) });
  }
};
