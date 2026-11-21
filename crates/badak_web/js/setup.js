// 데모와 측정이 함께 쓰는 것들: 모델이 있는 곳, 엔진을 올리는 순서 (48장).

import { Badak } from './client.js';
import { detect, pick } from './features.js';

const HUB = 'https://huggingface.co';
const MNIST = `${HUB}/building-an-inference-engine-from-scratch/mnist-tflite/resolve/9bb2098554f5c5e3cdb3b4c060eebf7abdf7f166`;
const SMOLLM2 = `${HUB}/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac`;

// 모델 파일들: 엔진에게 보일 길, 받을 주소, SHA-256. 판(revision)을 고정해
// 원래 있는 곳에서 받는다. 해시는 scripts/download_ch26.sh 와
// scripts/download_smollm2.sh 에 적힌 것과 같다.
// local을 주면 scripts/serve_web.py 가 내주는 models/ 폴더에서 받는다.
export function sources(local = false) {
  const mnist = local ? '/models/mnist_tflite' : MNIST;
  const llm = local ? '/models/smollm2_135m' : SMOLLM2;
  const file = (path, url, sha256) => ({ path, url, sha256 });
  return {
    mnist: [
      file('mnist/mlp.tflite', `${mnist}/mlp.tflite`, '1c38b87144e8e5ab75db517a6482cfb91ac1a80c4429df9e1dc8b5ee2ca0a13d'),
    ],
    llm: [
      file('smollm2_135m/config.json', `${llm}/config.json`, '8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a'),
      file('smollm2_135m/tokenizer.json', `${llm}/tokenizer.json`, '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'),
      file('smollm2_135m/model.safetensors', `${llm}/model.safetensors`, '5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'),
    ],
  };
}

export const SYSTEM = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face';

// 이 브라우저에 맞는 빌드를 골라 워커에 올린다.
export async function start({ build, threads, waiting, gpu = false, features = detect() } = {}) {
  build ??= pick(features);
  const threaded = build.includes('_threads');
  // 스레드는 여덟까지만 쓴다. 코어가 더 있어도 이 크기의 모델에서는 얻는 것이 적다.
  threads ??= threaded ? Math.min(features.cores, 8) : 1;
  const badak = new Badak();
  try {
    const started = await badak.call('start', {
      wasm: new URL(build, import.meta.url).href,
      threads,
      gpu,
      // 일을 기다리는 방법. 주지 않으면 엔진이 늘 하던 대로 한다: 잠깐 돌며
      // 기다리다가 일이 오지 않으면 잠든다 (16장).
      env: waiting ? { BADAK_WAITING: waiting } : {},
    });
    return { badak, build, threads: threaded ? threads : 1, features, ...started };
  } catch (error) {
    badak.stop();
    if (!threaded) throw error;
    // 스레드를 쓰는 빌드를 올리지 못했다. 쓰지 않는 빌드로 물러난다.
    console.warn(`${build}: ${error.message}`);
    return start({ build: build.replace('_threads', ''), waiting, gpu, features });
  }
}
