// badak을 브라우저(와 Node)에서 올리고 부른다 (48장).
//
// 하는 일은 셋이다. 엔진의 .wasm 파일을 올리고, 엔진이 들여오는 함수들을
// 건네고(wasi.js), 엔진이 내놓는 함수들을 JavaScript에서 쓰기 편한 꼴로
// 감싼다. 스레드를 쓰는 빌드라면 스레드가 될 워커들도 여기에서 만든다.

import { noGpu } from './gpu.js';
import { Wasi, importedMemory } from './wasi.js';

const inNode = typeof process !== 'undefined' && Boolean(process.versions?.node);

// 워커 하나. 브라우저와 Node의 워커는 쓰는 법이 조금 달라서 한 겹 감싼다.
async function startWorker(url, onMessage, onError) {
  if (inNode) {
    const { Worker } = await import('node:worker_threads');
    const worker = new Worker(url);
    worker.on('message', onMessage);
    worker.on('error', onError);
    return worker;
  }
  const worker = new Worker(url, { type: 'module' });
  worker.onmessage = (event) => onMessage(event.data);
  worker.onerror = (event) => onError(new Error(event.message ?? '워커의 스크립트를 읽지 못했습니다'));
  return worker;
}

// 엔진을 올린다.
//
//   wasm     .wasm 파일의 바이트 (ArrayBuffer)
//   files    엔진에게 보여 줄 파일들. 길 → Uint8Array
//   env      엔진에게 줄 환경 변수
//   threads  엔진이 쓸 스레드의 수 (부르는 스레드를 포함). 스레드를 쓰는
//            빌드에서만 뜻이 있다
//   print    엔진이 표준 출력에 적은 줄을 받는 함수
//   gpu      gpu.js의 openGpu가 준 것. 없으면 엔진은 GPU를 쓰지 않는다
export async function load({ wasm, files = new Map(), env = {}, threads = 1, print = console.log, gpu = null } = {}) {
  const module = await WebAssembly.compile(wasm);
  const wasi = new Wasi({ files, env, print });
  const state = { onPiece: null };
  const imports = {
    wasi_snapshot_preview1: wasi.imports(),
    badak: {
      // 엔진이 글 조각마다 부른다. 0을 돌려주면 계속 쓴다.
      piece(pointer, length) {
        return state.onPiece && state.onPiece(wasi.text(pointer >>> 0, length)) === false ? 1 : 0;
      },
    },
  };

  // ANCHOR: jspi
  // GPU의 일 가운데 셋(기다리기, 읽어 오기, 시계 읽기)은 JavaScript에서
  // "나중에 끝나는" 함수다. 엔진은 그것을 보통 함수처럼 부르고 답을 기다린다.
  // 그 사이를 브라우저가 메운다(JSPI). 들여오는 쪽에는 Suspending으로
  // 표시하고, 그런 함수를 부를 수 있는 내놓은 함수는 promising으로 감싼다.
  const jspi = typeof WebAssembly.Suspending === 'function';
  if (gpu && !jspi) gpu = null;
  imports.gpu = gpu ? gpu.imports((call) => new WebAssembly.Suspending(call)) : noGpu();
  // ANCHOR_END: jspi

  // ANCHOR: threads
  // 스레드를 쓰는 빌드는 메모리를 스스로 만들지 않고 들여온다. 여러 워커가
  // 같은 메모리를 봐야 하기 때문이다.
  const shape = importedMemory(wasm);
  let memory = null;
  if (shape) {
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new Error('이 페이지에서는 공유 메모리를 쓸 수 없습니다. 교차 출처 격리가 필요합니다.');
    }
    memory = new WebAssembly.Memory({ initial: shape.initial, maximum: shape.maximum, shared: true });
    imports.env = { memory };
    // 스레드가 될 워커들을 미리 만들어 둔다. 엔진이 스레드를 달라고 하는
    // 순간에는 기다릴 수 없다. 그때 엔진은 계산의 한가운데에 있다.
    const idle = [];
    const url = new URL('./thread.js', import.meta.url);
    for (let index = 1; index < threads; index++) {
      await new Promise((ready, failed) => {
        const died = (error) => failed(new Error(`스레드가 될 워커를 만들지 못했습니다: ${error.message}`));
        startWorker(url, (message) => message.kind === 'ready' && ready(), died).then((worker) => {
          idle.push(worker);
          worker.postMessage({ kind: 'init', module, memory, env });
        });
      });
    }
    let nextThread = 1;
    imports.wasi = {
      // 엔진이 스레드를 하나 만들어 달라고 한다. 쉬고 있는 워커에게 "이
      // 번호로, 이 인자로 시작하라"고 알리고 번호를 돌려준다.
      'thread-spawn'(argument) {
        const worker = idle.pop();
        if (!worker) return -1;
        const thread = nextThread++;
        worker.postMessage({ kind: 'start', thread, argument });
        return thread;
      },
    };
  }
  // ANCHOR_END: threads

  const instance = await WebAssembly.instantiate(module, imports);
  memory ??= instance.exports.memory;
  wasi.bind(memory);
  gpu?.bind(memory);
  instance.exports._initialize?.();
  instance.exports.badak_web_threads(shape ? threads : 1);
  return new Engine(instance.exports, memory, wasi, state, gpu);
}

// 엔진이 내놓은 함수들을 감싼 것.
export class Engine {
  constructor(exports, memory, wasi, state, gpu = null) {
    this.exports = exports;
    this.memory = memory;
    this.wasi = wasi;
    this.state = state;
    this.gpu = gpu;
    // GPU를 기다릴 수 있는 함수들. 이렇게 감싼 함수는 답 대신 약속을
    // 돌려준다. GPU가 없으면 감싸지 않고, 답이 바로 온다.
    this.waiting = {};
    for (const name of ['badak_web_llm_chat', 'badak_web_llm_speed', 'badak_web_gpu_linear']) {
      this.waiting[name] = gpu ? WebAssembly.promising(exports[name]) : exports[name];
    }
  }

  // f64 `count`개를 받아 올 자리를 얻어 `call`에게 주고, 적힌 값을 돌려준다.
  // `call`이 약속을 돌려주면 그것을 기다린다.
  numbers(count, call) {
    const slot = this.exports.badak_web_alloc(count * 8) >>> 0;
    const finish = (status) => {
      const values = Array.from(new Float64Array(this.memory.buffer, slot, count));
      this.exports.badak_web_free(slot, count * 8);
      if (status < 0) throw new Error(this.error());
      return values;
    };
    const status = call(slot);
    return status instanceof Promise ? status.then(finish) : finish(status);
  }

  // ANCHOR: gpu_calls
  // GPU를 연다. 쓸 수 있으면 그 이름을, 없으면 null을 돌려준다.
  openGpu() {
    if (!this.exports.badak_web_gpu_open()) return null;
    const pointer = this.exports.badak_web_alloc(256) >>> 0;
    const length = Math.min(this.exports.badak_web_gpu_name(pointer, 256), 256);
    const name = this.wasi.text(pointer, length);
    this.exports.badak_web_free(pointer, 256);
    return name;
  }

  // 선형 층 하나를 GPU에서 계산해 시간(초)과 지문을 돌려준다. 재기 전에
  // warm초 동안 같은 커널을 돌려 GPU를 데운다.
  async gpuLinear(variant, count, rows, cols, repeats, warm = 0.5) {
    const digestSlot = this.exports.badak_web_alloc(8) >>> 0;
    const [wall, inside, read, error] = await this.numbers(4, (out) =>
      this.waiting.badak_web_gpu_linear(variant, count, rows, cols, repeats, warm, out, digestSlot));
    const digest = new DataView(this.memory.buffer).getBigUint64(digestSlot, true).toString(16).padStart(16, '0');
    this.exports.badak_web_free(digestSlot, 8);
    return { wall, inside, read, error, digest };
  }
  // ANCHOR_END: gpu_calls

  // ANCHOR: strings
  // 글을 엔진의 메모리에 써 넣는다. 엔진에게 자리를 얻고, 거기에 바이트를
  // 복사한다. 쓰고 나면 release로 돌려준다.
  write(text) {
    const bytes = new TextEncoder().encode(text);
    // 주소는 부호 없는 수로 바꿔 쓴다. 2GB 위의 주소가 음수로 오기 때문이다.
    const pointer = this.exports.badak_web_alloc(bytes.length) >>> 0;
    new Uint8Array(this.memory.buffer, pointer, bytes.length).set(bytes);
    return { pointer, length: bytes.length };
  }

  release({ pointer, length }) {
    this.exports.badak_web_free(pointer, length);
  }
  // ANCHOR_END: strings

  error() {
    const capacity = 1024;
    const pointer = this.exports.badak_web_alloc(capacity) >>> 0;
    const length = Math.min(this.exports.badak_web_error(pointer, capacity), capacity);
    const text = this.wasi.text(pointer, length);
    this.exports.badak_web_free(pointer, capacity);
    return text;
  }

  // 어떤 커널로 빌드된 모듈인지.
  backend() {
    return ['기준 구현', '128비트 벡터', '128비트 벡터 + relaxed-simd'][this.exports.badak_web_backend()];
  }

  // 모듈의 메모리가 지금 얼마나 큰지(바이트). 줄어들지는 않는다.
  memoryBytes() {
    return this.memory.buffer.byteLength;
  }

  // 선형 층 하나를 여러 번 계산해, 가장 빨랐던 시간(초)과 결과의 지문을 돌려준다.
  benchLinear(rows, cols, count, repeats) {
    const slot = this.exports.badak_web_alloc(8) >>> 0;
    const seconds = this.exports.badak_web_bench_linear(rows, cols, count, repeats, slot);
    const digest = new DataView(this.memory.buffer).getBigUint64(slot, true).toString(16).padStart(16, '0');
    this.exports.badak_web_free(slot, 8);
    return { seconds, digest };
  }

  // ANCHOR: graph
  // .tflite 모델을 연다. 입력과 출력은 엔진의 메모리를 그대로 들여다보는
  // 배열이다. 복사가 없다.
  openGraph(path) {
    const name = this.write(path);
    const handle = this.exports.badak_web_graph_open(name.pointer, name.length);
    this.release(name);
    if (!handle) throw new Error(this.error());
    const { exports, memory } = this;
    return {
      // 메모리가 자라면 예전 배열은 못 쓰게 된다. 쓸 때마다 새로 만든다.
      input: () =>
        new Float32Array(memory.buffer, exports.badak_web_graph_input(handle) >>> 0, exports.badak_web_graph_input_len(handle)),
      output: () =>
        new Float32Array(memory.buffer, exports.badak_web_graph_output(handle) >>> 0, exports.badak_web_graph_output_len(handle)),
      run: () => {
        if (exports.badak_web_graph_run(handle) !== 0) throw new Error(this.error());
      },
    };
  }
  // ANCHOR_END: graph

  // ANCHOR: whisper
  // Whisper를 연다 (50장). transcribe는 WAV 파일의 길을 받아 글을 돌려준다.
  openWhisper(directory) {
    const name = this.write(directory);
    const handle = this.exports.badak_web_whisper_open(name.pointer, name.length);
    this.release(name);
    if (!handle) throw new Error(this.error());
    return {
      transcribe: (path, beams = 1) => {
        const [file, capacity] = [this.write(path), 4096];
        const out = this.exports.badak_web_alloc(capacity) >>> 0;
        const length = this.exports.badak_web_whisper_transcribe(handle, file.pointer, file.length, beams, out, capacity);
        this.release(file);
        const text = length < 0 ? null : this.wasi.text(out, Math.min(length, capacity));
        this.exports.badak_web_free(out, capacity);
        if (text === null) throw new Error(this.error());
        return text;
      },
      close: () => this.exports.badak_web_whisper_close(handle),
    };
  }
  // ANCHOR_END: whisper

  // 언어 모델을 연다. weights는 0(파일 그대로), 1(Q8_0), 2(Q4_0).
  openLlm(directory, weights = 0) {
    const name = this.write(directory);
    const handle = this.exports.badak_web_llm_open(name.pointer, name.length, weights);
    this.release(name);
    if (!handle) throw new Error(this.error());
    return {
      // 질문 하나에 답한다. 글 조각마다 onPiece를 부르고, 그것이 false를
      // 돌려주면 그만 쓴다. 쓴 토큰의 수를 돌려준다.
      chat: (system, question, maxTokens, onPiece) => {
        const [first, second] = [this.write(system ?? ''), this.write(question)];
        this.state.onPiece = onPiece;
        const finish = (written) => {
          this.state.onPiece = null;
          this.release(first);
          this.release(second);
          if (written < 0) throw new Error(this.error());
          return written;
        };
        // GPU가 있으면 약속이 온다. 없으면 답이 바로 온다.
        const written = this.waiting.badak_web_llm_chat(
          handle, first.pointer, first.length, second.pointer, second.length, maxTokens);
        return written instanceof Promise ? written.then(finish) : finish(written);
      },
      // 선형 층을 GPU에 맡긴다. mode 1은 층마다, 2는 MLP를 통째로.
      // 입력 벡터가 minCount개보다 적은 일은 CPU에 남는다. 올린 양(MB)을 돌려준다.
      offload: (mode, minCount = 1) => {
        const megabytes = this.exports.badak_web_llm_offload(handle, mode, minCount);
        if (megabytes < 0) throw new Error(this.error());
        return megabytes;
      },
      // 프롬프트 처리와 토큰 생성의 속도(토큰/초)를 잰다.
      speed: async (promptTokens, newTokens, repeats) => {
        const [prompt, generation] = await this.numbers(2, (out) =>
          this.waiting.badak_web_llm_speed(handle, promptTokens, newTokens, repeats, out));
        return { prompt, generation };
      },
      // GPU의 일에 든 시간. 물을 때마다 0으로 되돌아간다.
      statistics: () => {
        const [upload, compute, download, calls, onCpu] = this.numbers(5, (out) => {
          this.exports.badak_web_gpu_statistics(out);
          return 0;
        });
        return { upload, compute, download, calls, onCpu };
      },
      close: () => this.exports.badak_web_llm_close(handle),
    };
  }
}
