// 숫자를 그리면 badak이 알아맞히는 작은 데모 (48장).
//
// 쓰는 법: <div id="badak-mnist"></div> 를 두고 이 파일을 모듈로 읽는다.

import { sources, start } from './setup.js';

const SIZE = 252; // 그리는 판의 한 변(픽셀). 28의 9배.

// ANCHOR: prepare
// 그린 그림을 모델이 배운 꼴로 바꾼다. MNIST의 숫자는 28×28 사진의 한가운데
// 20×20 안에 들어 있고, 무게 중심이 사진의 가운데에 온다. 크게 그렸든 구석에
// 그렸든 그렇게 맞춰 준다.
function prepare(canvas) {
  const context = canvas.getContext('2d');
  const { data } = context.getImageData(0, 0, SIZE, SIZE);
  let [left, top, right, bottom] = [SIZE, SIZE, -1, -1];
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (data[(y * SIZE + x) * 4 + 3] > 0) {
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < 0) return null;
  const [width, height] = [right - left + 1, bottom - top + 1];
  const scale = 20 / Math.max(width, height);
  const small = new OffscreenCanvas(28, 28);
  const draw = small.getContext('2d', { willReadFrequently: true });
  const read = (shiftX, shiftY) => {
    draw.clearRect(0, 0, 28, 28);
    draw.drawImage(canvas, left, top, width, height,
      14 - (width * scale) / 2 + shiftX, 14 - (height * scale) / 2 + shiftY, width * scale, height * scale);
    const pixels = draw.getImageData(0, 0, 28, 28).data;
    return Float32Array.from({ length: 784 }, (_, index) => pixels[index * 4 + 3] / 255);
  };
  // 한 번 그려 무게 중심을 재고, 그만큼 옮겨 다시 그린다.
  const first = read(0, 0);
  let [mass, centerX, centerY] = [0, 0, 0];
  first.forEach((value, index) => {
    mass += value;
    centerX += value * (index % 28);
    centerY += value * Math.floor(index / 28);
  });
  return read(13.5 - centerX / mass, 13.5 - centerY / mass);
}
// ANCHOR_END: prepare

export async function mount(root, { local = false, engine } = {}) {
  root.innerHTML = `
    <style>
      .badak-mnist { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; margin: 1em 0; }
      .badak-mnist canvas.pad { border: 1px solid var(--table-border-color, #999); border-radius: 6px; touch-action: none;
        cursor: crosshair; background: var(--bg, #fff); }
      .badak-mnist .bars { display: inline-grid; grid-template-columns: auto 160px auto; gap: 3px 8px; align-items: center;
        font-variant-numeric: tabular-nums; font-size: 0.9em; }
      .badak-mnist .bar { height: 12px; border-radius: 0 3px 3px 0; background: #2a78d6; min-width: 1px; }
      .badak-mnist .bars span:nth-child(3n) { text-align: right; }
      .badak-mnist .best { font-weight: bold; }
      .badak-mnist .status { font-size: 0.85em; opacity: 0.8; margin-top: 8px; max-width: 30em; }
      .badak-mnist button { margin-top: 6px; }
    </style>
    <div class="badak-mnist">
      <div><canvas class="pad" width="${SIZE}" height="${SIZE}" aria-label="숫자를 그리는 판"></canvas><br>
        <button type="button">지우기</button></div>
      <div><div class="bars"></div><div class="status">엔진을 올리는 중…</div></div>
    </div>`;
  const canvas = root.querySelector('canvas');
  const bars = root.querySelector('.bars');
  const status = root.querySelector('.status');
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const show = (shares) => {
    const best = shares.indexOf(Math.max(...shares));
    bars.innerHTML = shares.map((share, digit) => `
      <span class="${digit === best && share > 0 ? 'best' : ''}">${digit}</span>
      <span><div class="bar" style="width:${(share * 100).toFixed(1)}%"></div></span>
      <span class="${digit === best && share > 0 ? 'best' : ''}">${(share * 100).toFixed(1)}%</span>`).join('');
  };
  show(new Array(10).fill(0));

  let ready;
  try {
    ready = engine ?? (await start());
    for (const file of sources(local).mnist) await ready.badak.call('file', file);
  } catch (error) {
    status.textContent = `엔진을 올리지 못했습니다: ${error.message}`;
    return null;
  }
  const about = `${ready.build} (${ready.backend}${ready.threads > 1 ? `, 스레드 ${ready.threads}개` : ''})`;
  status.textContent = `${about}을 올렸습니다. 숫자를 하나 그려 보세요.`;

  let busy = false;
  let again = false;
  async function recognize() {
    if (busy) {
      again = true;
      return;
    }
    const pixels = prepare(canvas);
    if (!pixels) return;
    busy = true;
    const { scores, ms } = await ready.badak.call('mnist', { pixels, repeats: 100 });
    // 이 모델의 마지막 층은 소프트맥스다. 나온 값이 그대로 확률이다.
    show(scores);
    status.textContent = `${about} · 계산 한 번에 ${(ms * 1000).toFixed(0)}µs`;
    busy = false;
    if (again) {
      again = false;
      recognize();
    }
  }

  let last = null;
  const point = (event) => {
    const box = canvas.getBoundingClientRect();
    return [((event.clientX - box.left) * SIZE) / box.width, ((event.clientY - box.top) * SIZE) / box.height];
  };
  canvas.addEventListener('pointerdown', (event) => {
    canvas.setPointerCapture(event.pointerId);
    last = point(event);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!last) return;
    const next = point(event);
    context.lineWidth = 18;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = getComputedStyle(canvas).color;
    context.beginPath();
    context.moveTo(...last);
    context.lineTo(...next);
    context.stroke();
    last = next;
    recognize();
  });
  const lift = () => {
    last = null;
  };
  canvas.addEventListener('pointerup', lift);
  canvas.addEventListener('pointercancel', lift);
  root.querySelector('button').addEventListener('click', () => {
    context.clearRect(0, 0, SIZE, SIZE);
    show(new Array(10).fill(0));
  });
  return ready;
}

const root = document.getElementById('badak-mnist');
if (root && !root.dataset.manual) mount(root, { local: new URLSearchParams(location.search).get('models') === 'local' });
