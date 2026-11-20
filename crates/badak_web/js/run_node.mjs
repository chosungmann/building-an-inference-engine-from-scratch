// Node에서 엔진의 .wasm을 돌려 본다 (48장).
//
// 쓰는 법: node crates/badak_web/js/run_node.mjs <.wasm> <스레드 수> [llm]
//
// 브라우저에서 쓰는 것과 같은 badak.js와 wasi.js로 올린다. 브라우저 없이
// 답이 맞는지 확인하고, CI에서도 돌린다. 적는 줄은
// crates/badak_web/examples/ch48_native.rs 가 맥에서 적는 줄과 같은 꼴이다.

import { readFile } from 'node:fs/promises';
import { load } from './badak.js';

const [path, threadText = '1', what = ''] = process.argv.slice(2);
const root = new URL('../../../', import.meta.url);
const models = process.env.BADAK_MODELS ? new URL(`file://${process.env.BADAK_MODELS}/`) : new URL('models/', root);
const file = async (name) => new Uint8Array(await readFile(new URL(name, models)));

const files = new Map([['mnist/mlp.tflite', await file('mnist_tflite/mlp.tflite')]]);
if (what === 'llm') {
  for (const name of ['config.json', 'tokenizer.json', 'model.safetensors']) {
    files.set(`smollm2_135m/${name}`, await file(`smollm2_135m/${name}`));
  }
}
const wasm = await readFile(path);
const engine = await load({
  wasm: wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength),
  files,
  threads: Number(threadText),
  // 스레드가 일을 기다리는 방법을 바꿔 볼 수 있다 (sleep, spin).
  env: process.env.BADAK_WAITING ? { BADAK_WAITING: process.env.BADAK_WAITING } : {},
});
console.log(`backend ${engine.backend()}`);

// 1. 선형 층 하나. 지문이 기준 구현의 것과 같은가.
for (const [rows, cols, count] of [[13, 203, 9], [1536, 576, 1], [1536, 576, 64]]) {
  const { seconds, digest } = engine.benchLinear(rows, cols, count, 30);
  console.log(`linear ${rows}x${cols}x${count} 지문 ${digest} 시간 ${(seconds * 1e6).toFixed(1)}`);
}

// 2. 손글씨 숫자 하나. MNIST 시험 사진의 첫 장이다.
const images = await file('mnist/t10k-images-idx3-ubyte');
const graph = engine.openGraph('/mnist/mlp.tflite');
graph.input().set(Array.from(images.subarray(16, 16 + 784), (pixel) => pixel / 255));
graph.run();
const scores = Array.from(graph.output());
const bits = Array.from(new Uint32Array(new Float32Array(scores).buffer), (word) => word.toString(16).padStart(8, '0'));
console.log(`mnist 답 ${scores.indexOf(Math.max(...scores))} 비트 ${bits.join(' ')}`);

// 3. 언어 모델.
if (what === 'llm') {
  const llm = engine.openLlm('/smollm2_135m', 0);
  let answer = '';
  const started = performance.now();
  const written = llm.chat(
    'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
    'What is the capital of France?', 32, (piece) => { answer += piece; });
  const hex = Buffer.from(answer, 'utf8').toString('hex');
  console.log(`llm ${written} ${hex}`);
  console.log(`llm-time ${(performance.now() - started).toFixed(0)}ms 메모리 ${(engine.memoryBytes() / 1e6).toFixed(0)}MB`);
  llm.close();
}
process.exit(0);
