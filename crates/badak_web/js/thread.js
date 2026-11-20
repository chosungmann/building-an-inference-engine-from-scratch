// 엔진의 스레드 하나가 되는 워커 (48장).
//
// 브라우저에는 스레드를 만드는 함수가 없다. 워커가 있을 뿐이다. 워커는 저마다
// 자기 메모리를 갖는 따로 떨어진 세계인데, WebAssembly의 메모리를 "공유"로
// 만들면 여러 워커가 같은 메모리를 볼 수 있다. 그러면 워커가 스레드 노릇을
// 한다.
//
// 이 워커는 같은 모듈을, 같은 메모리로 한 번 더 올린다. 그리고 엔진이
// 스레드를 달라고 할 때까지 기다렸다가, 엔진이 정해 준 함수를 실행한다.

import { Wasi } from './wasi.js';

const inNode = typeof process !== 'undefined' && Boolean(process.versions?.node);
const port = inNode ? (await import('node:worker_threads')).parentPort : self;
const listen = (handler) =>
  inNode ? port.on('message', handler) : (port.onmessage = (event) => handler(event.data));

let exports = null;

listen(async (message) => {
  if (message.kind === 'init') {
    const wasi = new Wasi({ env: message.env });
    wasi.bind(message.memory);
    const instance = await WebAssembly.instantiate(message.module, {
      env: { memory: message.memory },
      wasi_snapshot_preview1: wasi.imports(),
      // 엔진의 스레드는 글을 쓰지도, 스레드를 더 만들지도 않는다.
      badak: { piece: () => 0 },
      wasi: { 'thread-spawn': () => -1 },
    });
    exports = instance.exports;
    port.postMessage({ kind: 'ready' });
  } else if (message.kind === 'start') {
    // 여기에서 돌아오지 않는다. 엔진의 스레드는 일을 기다리며 계속 산다.
    exports.wasi_thread_start(message.thread, message.argument);
  }
});
