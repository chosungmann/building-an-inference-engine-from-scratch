// 이 브라우저의 GPU로 선형 층 하나를 계산해 보는 작은 데모 (49장).
//
// 쓰는 법: <div id="badak-gpu"></div> 를 두고 이 파일을 모듈로 읽는다.

import { start } from './setup.js';

// 벡터 256개, 2048 → 2048. Llama 3.2 1B의 어텐션에 있는 선형 층의 크기다.
const [COUNT, ROWS, COLS] = [256, 2048, 2048];

export function mount(root) {
  root.innerHTML = `
    <style>
      .badak-gpu { margin: 1em 0; }
      .badak-gpu table { border-collapse: collapse; margin: 8px 0; font-variant-numeric: tabular-nums; }
      .badak-gpu td, .badak-gpu th { border: 1px solid var(--table-border-color, #999); padding: 3px 12px; text-align: right; }
      .badak-gpu th { text-align: left; font-weight: normal; }
      .badak-gpu .status { font-size: 0.85em; opacity: 0.8; }
    </style>
    <div class="badak-gpu">
      <button type="button">이 브라우저의 GPU로 계산해 보기</button>
      <div class="result"></div>
      <div class="status"></div>
    </div>`;
  const button = root.querySelector('button');
  const result = root.querySelector('.result');
  const status = root.querySelector('.status');
  const milliseconds = (seconds) => `${(seconds * 1000).toFixed(2)}ms`;

  button.addEventListener('click', async () => {
    button.disabled = true;
    result.innerHTML = '';
    status.textContent = '엔진을 올리는 중…';
    let engine = null;
    try {
      engine = await start({ gpu: true });
      if (!engine.gpu) {
        status.textContent = typeof WebAssembly.Suspending === 'function'
          ? '이 브라우저에서는 GPU를 열지 못했습니다(WebGPU가 없거나 꺼져 있습니다).'
          : '이 브라우저에는 엔진이 GPU를 기다리는 데 쓰는 기능(JSPI)이 없습니다.';
        return;
      }
      status.textContent = 'GPU와 CPU에서 같은 선형 층을 계산하는 중…';
      const gpu = await engine.badak.call('gpuLinear', { variant: 3, count: COUNT, rows: ROWS, cols: COLS, repeats: 20, warm: 0.5 });
      const cpu = await engine.badak.call('linear', { rows: ROWS, cols: COLS, count: COUNT, repeats: 3 });
      const inside = gpu.inside >= 0 ? milliseconds(gpu.inside) : '잴 수 없음';
      result.innerHTML = `
        <table>
          <tr><th>GPU: 명령을 보내고 끝나기를 기다린 시간</th><td>${milliseconds(gpu.wall)}</td></tr>
          <tr><th>GPU: 그 가운데 GPU가 스스로 잰 계산 시간</th><td>${inside}</td></tr>
          <tr><th>GPU: 결과 2MB를 읽어 오는 시간</th><td>${milliseconds(gpu.read)}</td></tr>
          <tr><th>CPU: 같은 계산 (스레드 ${engine.threads}개)</th><td>${milliseconds(cpu.seconds)}</td></tr>
          <tr><th>두 결과의 가장 큰 상대 오차</th><td>${gpu.error.toExponential(1)}</td></tr>
        </table>`;
      status.textContent = `GPU: ${engine.gpu}. 엔진: ${engine.build} (${engine.backend}).`;
    } catch (error) {
      status.textContent = `실패했습니다: ${error.message}`;
    } finally {
      engine?.badak.stop();
      button.disabled = false;
    }
  });
}

const root = document.getElementById('badak-gpu');
if (root) mount(root);
