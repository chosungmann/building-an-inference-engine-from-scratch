// 엔진이 GPU에 시키는 일을 브라우저의 WebGPU로 해 준다 (49장).
//
// 엔진은 "gpu"라는 묶음에서 함수 열두 개를 들여온다
// (crates/badak_gpu/src/web.rs). 여기에서 그것들을 만든다. 버퍼와 커널은
// 이쪽에 두고 엔진에게는 번호만 준다.
//
// 맥에서는 같은 자리에 wgpu가 있었다. wgpu가 Metal에게 하던 말을 여기에서는
// 브라우저에게 한다. 브라우저가 그 말을 다시 Metal에게 전한다.

// ANCHOR: open
// GPU를 고르고 연다. 없으면 null. 이 일은 "나중에 끝나는" 일이어서 엔진을
// 올리기 전에 끝내 둔다.
export async function openGpu() {
  const adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const limits = adapter.limits;
  const features = ['timestamp-query'].filter((name) => adapter.features.has(name));
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    // WebGPU의 기본 한도는 버퍼 하나를 256MB로, 셰이더에 묶는 버퍼를 128MB로
    // 묶는다. 가중치 행렬이 그보다 크므로 이 GPU가 허락하는 만큼 달라고 한다.
    requiredLimits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    },
  });
  return new Gpu(adapter, device);
}
// ANCHOR_END: open

export class Gpu {
  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.memory = null;
    this.buffers = new Map(); // 번호 → GPUBuffer
    this.kernels = []; // 번호 → GPUComputePipeline
    this.next = 1;
    this.encoder = null; // 아직 보내지 않은 명령 묶음.
    this.staging = new Map(); // 크기 → 읽어 올 때 쓰는 버퍼.
    this.timer = device.features.has('timestamp-query')
      ? {
          queries: device.createQuerySet({ type: 'timestamp', count: 2 }),
          times: device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
          readable: device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        }
      : null;
    // 오류는 버리지 않고 모아 둔다. WebGPU의 잘못된 호출은 예외를 던지지 않는다.
    this.errors = [];
    device.addEventListener?.('uncapturederror', (event) => this.errors.push(event.error.message));
    this.lost = null;
    device.lost.then((info) => {
      this.lost = info.message || info.reason;
    });
  }

  // 화면에 적을 이름. 브라우저는 GPU의 자세한 이름을 알려 주지 않는다.
  name() {
    const info = this.adapter.info ?? {};
    return [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || '이름 없음';
  }

  bind(memory) {
    this.memory = memory;
  }

  bytes(pointer, length) {
    return new Uint8Array(this.memory.buffer, pointer, length);
  }

  // 결과를 읽어 올 때 쓰는 버퍼. 계산에 쓰는 버퍼는 CPU가 열 수 없어서,
  // 열 수 있는 버퍼로 한 번 옮겨 적는다. 같은 크기는 돌려 쓴다.
  stagingFor(size) {
    let buffer = this.staging.get(size);
    if (!buffer) {
      buffer = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.staging.set(size, buffer);
    }
    return buffer;
  }

  // 엔진에게 건넬 함수들. suspending은 "나중에 끝나는" 함수를 엔진이 그냥
  // 부를 수 있게 표시하는 것이다(badak.js).
  imports(suspending) {
    const self = this;
    const { device } = this;
    const functions = {
      open: () => 1,
      name(pointer, capacity) {
        const text = new TextEncoder().encode(self.name());
        self.bytes(pointer, Math.min(capacity, text.length)).set(text.subarray(0, capacity));
        return text.length;
      },
      can_time: () => (self.timer ? 1 : 0),
      // ANCHOR: kernel
      // WGSL 소스를 커널로 만든다. 23장의 셰이더 파일 그대로다.
      kernel(source, sourceLength, entry, entryLength) {
        const decode = (pointer, length) => new TextDecoder().decode(self.bytes(pointer, length).slice());
        const module = device.createShaderModule({ code: decode(source, sourceLength) });
        self.kernels.push(
          device.createComputePipeline({
            // 버퍼를 어떤 자리에 묶는지는 셰이더의 선언에서 읽어 낸다.
            layout: 'auto',
            compute: { module, entryPoint: decode(entry, entryLength) },
          }),
        );
        return self.kernels.length - 1;
      },
      // ANCHOR_END: kernel
      buffer(bytes, uniform) {
        const usage = uniform
          ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
          : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        self.buffers.set(self.next, device.createBuffer({ size: bytes, usage }));
        return self.next++;
      },
      // ANCHOR: write
      // 모듈의 메모리에 있는 값을 버퍼에 적는다. 브라우저가 값을 복사해
      // 가므로, 돌아온 뒤에 엔진이 그 메모리를 다시 써도 된다.
      write(buffer, offset, pointer, length) {
        device.queue.writeBuffer(self.buffers.get(buffer), offset, self.bytes(pointer, length));
      },
      // ANCHOR_END: write
      // ANCHOR: encode
      encode(kernel, ids, count, x, y, z, timed) {
        const pipeline = self.kernels[kernel];
        const list = new Uint32Array(self.memory.buffer, ids, count);
        // 1. 묶음: 셰이더의 자리마다 어느 버퍼를 줄지.
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: Array.from(list, (id, binding) => ({ binding, resource: { buffer: self.buffers.get(id) } })),
        });
        // 2. 명령: 어느 커널을, 어느 버퍼로, 몇 그룹 실행할지.
        self.encoder ??= device.createCommandEncoder();
        const pass = self.encoder.beginComputePass(
          timed && self.timer
            ? { timestampWrites: { querySet: self.timer.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
            : undefined,
        );
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
      },
      submit() {
        if (!self.encoder) return;
        device.queue.submit([self.encoder.finish()]);
        self.encoder = null;
      },
      // ANCHOR_END: encode
      // ANCHOR: read
      // 보낸 명령이 모두 끝날 때까지 기다린다.
      wait: () => device.queue.onSubmittedWorkDone(),
      // 버퍼의 값을 모듈의 메모리로 읽어 온다.
      async read(buffer, pointer, length) {
        const size = Math.ceil(length / 4) * 4;
        const staging = self.stagingFor(size);
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(self.buffers.get(buffer), 0, staging, 0, size);
        device.queue.submit([encoder.finish()]);
        // 열어 달라고 부탁하고, GPU가 앞선 일을 끝내고 열어 줄 때까지 기다린다.
        await staging.mapAsync(GPUMapMode.READ);
        // 기다리는 동안 모듈의 메모리가 자랐을 수 있다. 지금의 것을 본다.
        self.bytes(pointer, length).set(new Uint8Array(staging.getMappedRange(), 0, length));
        staging.unmap();
      },
      // ANCHOR_END: read
      // 시계를 달고 실행한 마지막 커널이 걸린 시간(나노초).
      async elapsed() {
        const { queries, times, readable } = self.timer;
        const encoder = device.createCommandEncoder();
        encoder.resolveQuerySet(queries, 0, 2, times, 0);
        encoder.copyBufferToBuffer(times, 0, readable, 0, 16);
        device.queue.submit([encoder.finish()]);
        await readable.mapAsync(GPUMapMode.READ);
        const [began, ended] = new BigUint64Array(readable.getMappedRange().slice(0));
        readable.unmap();
        return Number(ended - began);
      },
      release(buffer) {
        self.buffers.get(buffer)?.destroy();
        self.buffers.delete(buffer);
      },
    };
    // 주소는 부호 없는 수로 바꿔 받는다(wasi.js와 같은 까닭).
    const unsigned = (value) => (typeof value === 'number' && Number.isInteger(value) ? value >>> 0 : value);
    // 이 셋은 약속(Promise)을 돌려준다. 엔진이 그 약속이 지켜질 때까지 멈춰
    // 있게 하려면 따로 표시해서 건네야 한다.
    const waits = new Set(['wait', 'read', 'elapsed']);
    for (const [name, call] of Object.entries(functions)) {
      const plain = (...values) => call(...values.map(unsigned));
      functions[name] = waits.has(name) ? suspending(plain) : plain;
    }
    return functions;
  }
}

// GPU가 없을 때 엔진에게 건네는 함수들. 엔진은 open이 0인 것을 보고 GPU를
// 쓰지 않는다. 나머지는 불릴 일이 없다.
export function noGpu() {
  const never = () => {
    throw new Error('GPU가 없습니다');
  };
  return {
    open: () => 0, name: () => 0, can_time: () => 0, kernel: never, buffer: never, write: never,
    encode: never, submit: never, wait: never, read: never, elapsed: never, release: never,
  };
}
