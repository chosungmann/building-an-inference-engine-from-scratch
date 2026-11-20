// 페이지에서 엔진 워커를 부른다 (48장).

export class Badak {
  constructor(base = new URL('./', import.meta.url)) {
    this.base = base;
    this.worker = new Worker(new URL('engine.js', base), { type: 'module' });
    this.waiting = new Map();
    this.next = 1;
    this.worker.onmessage = ({ data }) => {
      const call = this.waiting.get(data.id);
      if (!call) return;
      if (data.kind === 'done') {
        this.waiting.delete(data.id);
        call.resolve(data.result);
      } else if (data.kind === 'error') {
        this.waiting.delete(data.id);
        call.reject(new Error(data.message));
      } else {
        call.listen?.(data);
      }
    };
    this.worker.onerror = (event) => {
      for (const call of this.waiting.values()) call.reject(new Error(event.message || '엔진 워커가 죽었습니다'));
      this.waiting.clear();
    };
  }

  // 요청 하나를 보내고 답을 기다린다. 도중의 소식(진행, 글 조각)은 listen이 받는다.
  call(kind, payload = {}, listen) {
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.waiting.set(id, { resolve, reject, listen });
      this.worker.postMessage({ ...payload, kind, id });
    });
  }

  stop() {
    this.worker.terminate();
  }
}
