// 엔진이 운영체제에 부탁하는 일들을 대신 해 준다 (48장).
//
// 엔진은 WASI라는 약속에 맞춰 빌드되어 있다. WASI는 "WebAssembly 모듈이
// 바깥에 부탁할 수 있는 일"의 목록이다. 파일을 열어 달라, 시각을 알려 달라,
// 난수를 달라. 브라우저에는 그런 것을 해 줄 운영체제가 없으므로 이 파일이
// 그 자리에 선다. 엔진이 들여오는 함수는 스물세 개이고, 여기에 다 있다.
//
// 파일은 메모리에 든 것만 있다. 읽기만 된다.

// WASI가 정한 오류의 번호.
const OK = 0, BADF = 8, INVAL = 28, NOENT = 44, NOSYS = 52, ROFS = 69;
// 파일의 종류.
const CHARACTER = 2, DIRECTORY = 3, REGULAR = 4;
// 미리 열려 있는 폴더("/")의 번호.
const ROOT = 3;

// ANCHOR: chunks
// 여러 조각으로 된 파일 하나 (49장).
//
// 받은 파일을 큰 배열 하나로 이어 붙이지 않고, 받은 조각들을 그대로 둔다.
// 이어 붙이려면 파일만 한 배열이 하나 더 필요하고(잠깐이지만 두 벌이 된다),
// 2GB가 넘는 배열은 만들지 못하는 브라우저도 있다.
export class Chunks {
  constructor(chunks) {
    this.chunks = chunks;
    this.starts = []; // 조각마다, 파일의 몇 번째 바이트에서 시작하는지.
    this.length = 0;
    for (const chunk of chunks) {
      this.starts.push(this.length);
      this.length += chunk.length;
    }
  }

  // position부터 target.length바이트를 target에 복사하고, 복사한 수를 돌려준다.
  copy(position, target) {
    // position이 든 조각을 찾는다(이분 탐색).
    let [low, high] = [0, this.chunks.length - 1];
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (this.starts[middle] <= position) low = middle;
      else high = middle - 1;
    }
    let copied = 0;
    for (let index = low; index < this.chunks.length && copied < target.length; index++) {
      const from = position + copied - this.starts[index];
      const piece = this.chunks[index].subarray(from, from + target.length - copied);
      target.set(piece, copied);
      copied += piece.length;
    }
    return copied;
  }
}
// ANCHOR_END: chunks

export class Wasi {
  // files: 길 → Uint8Array 또는 Chunks. env: 환경 변수. print: 엔진이 적은
  // 줄을 받는 함수.
  constructor({ files = new Map(), env = {}, print = console.log } = {}) {
    this.files = files;
    this.env = Object.entries(env).map(([key, value]) => `${key}=${value}`);
    this.print = print;
    this.open = new Map(); // 번호 → { path, bytes, position } 또는 { path, directory: true }
    this.next = ROOT + 1;
    this.pending = ''; // 아직 줄이 끝나지 않은 출력.
    this.memory = null;
  }

  // 모듈의 메모리를 알려 준다. 메모리가 자라면 배열을 다시 만들어야 하므로
  // 쓸 때마다 새로 본다.
  bind(memory) {
    this.memory = memory;
  }

  view() {
    return new DataView(this.memory.buffer);
  }

  bytes(pointer, length) {
    return new Uint8Array(this.memory.buffer, pointer, length);
  }

  // 공유 메모리에서는 글을 바로 읽지 못하는 브라우저가 있다. 복사해서 읽는다.
  text(pointer, length) {
    return new TextDecoder().decode(this.bytes(pointer, length).slice());
  }

  // "/a/./b/" 같은 길을 "a/b"로 다듬는다.
  normalize(path) {
    const parts = [];
    for (const part of path.split('/')) {
      if (part === '' || part === '.') continue;
      if (part === '..') parts.pop();
      else parts.push(part);
    }
    return parts.join('/');
  }

  isDirectory(path) {
    if (path === '') return true;
    for (const name of this.files.keys()) if (name.startsWith(path + '/')) return true;
    return false;
  }

  // 파일 하나의 정보를 64바이트에 적는다.
  stat(buffer, type, size) {
    const view = this.view();
    for (let offset = 0; offset < 64; offset += 8) view.setBigUint64(buffer + offset, 0n, true);
    view.setUint8(buffer + 16, type);
    view.setBigUint64(buffer + 24, 1n, true);
    view.setBigUint64(buffer + 32, BigInt(size), true);
  }

  imports() {
    const self = this;
    const functions = {
      // 환경 변수. 엔진은 BADAK_WAITING 같은 것을 여기에서 읽는다.
      environ_sizes_get(count, size) {
        const view = self.view();
        view.setUint32(count, self.env.length, true);
        view.setUint32(size, self.env.reduce((sum, entry) => sum + new TextEncoder().encode(entry).length + 1, 0), true);
        return OK;
      },
      environ_get(pointers, buffer) {
        const view = self.view();
        for (const entry of self.env) {
          const bytes = new TextEncoder().encode(entry);
          view.setUint32(pointers, buffer, true);
          self.bytes(buffer, bytes.length).set(bytes);
          view.setUint8(buffer + bytes.length, 0);
          pointers += 4;
          buffer += bytes.length + 1;
        }
        return OK;
      },
      // ANCHOR: clock
      // 시각. 0번은 벽시계, 1번은 거꾸로 가지 않는 시계다. 나노초로 답한다.
      clock_time_get(id, _precision, result) {
        const now = id === 0 ? Date.now() : performance.now();
        self.view().setBigUint64(result, BigInt(Math.round(now * 1e6)), true);
        return OK;
      },
      // ANCHOR_END: clock
      // 난수. 표준 라이브러리의 해시 표가 시드로 쓴다.
      random_get(buffer, length) {
        const random = new Uint8Array(length);
        for (let offset = 0; offset < length; offset += 65536) {
          crypto.getRandomValues(random.subarray(offset, Math.min(offset + 65536, length)));
        }
        self.bytes(buffer, length).set(random);
        return OK;
      },
      sched_yield() {
        return OK;
      },
      proc_exit(code) {
        throw new Error(`엔진이 끝났습니다 (${code})`);
      },

      // 미리 열려 있는 폴더를 묻는다. 3번이 "/"이고 그 뒤로는 없다.
      fd_prestat_get(fd, buffer) {
        if (fd !== ROOT) return BADF;
        const view = self.view();
        view.setUint8(buffer, 0);
        view.setUint32(buffer + 4, 1, true);
        return OK;
      },
      fd_prestat_dir_name(fd, buffer, _length) {
        if (fd !== ROOT) return BADF;
        self.bytes(buffer, 1)[0] = 47; // '/'
        return OK;
      },
      fd_fdstat_get(fd, buffer) {
        const entry = self.open.get(fd);
        const type = fd < ROOT ? CHARACTER : fd === ROOT || entry?.directory ? DIRECTORY : entry ? REGULAR : 0;
        if (!type) return BADF;
        const view = self.view();
        view.setUint8(buffer, type);
        view.setUint16(buffer + 2, 0, true);
        view.setBigUint64(buffer + 8, ~0n & 0xffffffffffffffffn, true);
        view.setBigUint64(buffer + 16, ~0n & 0xffffffffffffffffn, true);
        return OK;
      },
      // ANCHOR: files
      // 파일을 연다. 길은 폴더 번호에서 시작한다.
      path_open(dirfd, _flags, pathPointer, pathLength, oflags, _base, _inheriting, _fdflags, result) {
        const base = dirfd === ROOT ? '' : self.open.get(dirfd)?.path;
        if (base === undefined) return BADF;
        const path = self.normalize(base + '/' + self.text(pathPointer, pathLength));
        // 1: 없으면 만들기, 8: 비우기. 읽기만 되는 곳이다.
        if (oflags & 9) return ROFS;
        const bytes = self.files.get(path);
        if (bytes) self.open.set(self.next, { path, bytes, position: 0 });
        else if (self.isDirectory(path)) self.open.set(self.next, { path, directory: true });
        else return NOENT;
        self.view().setUint32(result, self.next++, true);
        return OK;
      },
      fd_close(fd) {
        return self.open.delete(fd) || fd <= ROOT ? OK : BADF;
      },
      // 읽는다. 받을 자리가 여럿일 수 있다(자리마다 주소와 길이).
      fd_read(fd, vectors, count, result) {
        const entry = self.open.get(fd);
        if (!entry || entry.directory) return BADF;
        const view = self.view();
        let total = 0;
        for (let index = 0; index < count; index++) {
          const pointer = view.getUint32(vectors + index * 8, true);
          const length = view.getUint32(vectors + index * 8 + 4, true);
          let copied;
          if (entry.bytes instanceof Chunks) {
            const most = Math.max(0, Math.min(length, entry.bytes.length - entry.position));
            copied = entry.bytes.copy(entry.position, self.bytes(pointer, most));
          } else {
            const piece = entry.bytes.subarray(entry.position, entry.position + length);
            self.bytes(pointer, piece.length).set(piece);
            copied = piece.length;
          }
          entry.position += copied;
          total += copied;
          if (copied < length) break;
        }
        view.setUint32(result, total, true);
        return OK;
      },
      // ANCHOR_END: files
      fd_seek(fd, offset, whence, result) {
        const entry = self.open.get(fd);
        if (!entry || entry.directory) return BADF;
        const from = [0, entry.position, entry.bytes.length][whence];
        if (from === undefined) return INVAL;
        entry.position = Math.max(0, from + Number(offset));
        self.view().setBigUint64(result, BigInt(entry.position), true);
        return OK;
      },
      fd_filestat_get(fd, buffer) {
        const entry = self.open.get(fd);
        if (fd === ROOT || entry?.directory) self.stat(buffer, DIRECTORY, 0);
        else if (entry) self.stat(buffer, REGULAR, entry.bytes.length);
        else if (fd < ROOT) self.stat(buffer, CHARACTER, 0);
        else return BADF;
        return OK;
      },
      path_filestat_get(dirfd, _flags, pathPointer, pathLength, buffer) {
        const base = dirfd === ROOT ? '' : self.open.get(dirfd)?.path;
        if (base === undefined) return BADF;
        const path = self.normalize(base + '/' + self.text(pathPointer, pathLength));
        const bytes = self.files.get(path);
        if (bytes) self.stat(buffer, REGULAR, bytes.length);
        else if (self.isDirectory(path)) self.stat(buffer, DIRECTORY, 0);
        else return NOENT;
        return OK;
      },
      // 쓴다. 1번(표준 출력)과 2번(표준 오류)만 받아, 줄 단위로 넘긴다.
      fd_write(fd, vectors, count, result) {
        if (fd !== 1 && fd !== 2) return BADF;
        const view = self.view();
        let total = 0;
        for (let index = 0; index < count; index++) {
          const pointer = view.getUint32(vectors + index * 8, true);
          const length = view.getUint32(vectors + index * 8 + 4, true);
          self.pending += self.text(pointer, length);
          total += length;
        }
        const lines = self.pending.split('\n');
        self.pending = lines.pop();
        for (const line of lines) self.print(line);
        view.setUint32(result, total, true);
        return OK;
      },
      fd_sync() {
        return OK;
      },
      fd_readdir() {
        return NOSYS;
      },
      // 바꾸는 일은 모두 거절한다.
      path_create_directory() {
        return ROFS;
      },
      path_remove_directory() {
        return ROFS;
      },
      path_rename() {
        return ROFS;
      },
      path_unlink_file() {
        return ROFS;
      },
    };
    // ANCHOR: unsigned
    // 모듈이 넘기는 32비트 정수는 JavaScript에 부호 있는 수로 온다. 메모리가
    // 2GB를 넘으면 그 위의 주소가 음수로 보인다. 주소도 길이도 번호도 음수일
    // 일이 없으므로, 받은 수를 모두 부호 없는 수로 바꿔서 넘긴다 (49장).
    const unsigned = (value) => (typeof value === 'number' ? value >>> 0 : value);
    for (const [name, call] of Object.entries(functions)) {
      functions[name] = (...values) => call(...values.map(unsigned));
    }
    return functions;
    // ANCHOR_END: unsigned
  }
}

// 모듈이 메모리를 들여온다면, 그 메모리의 처음 크기와 한도를 모듈의
// 바이트에서 읽는다. 브라우저는 들여오기의 이름까지만 알려 주고 크기는
// 알려 주지 않는다. 메모리를 우리가 만들어 건네야 하므로 직접 읽는다.
export function importedMemory(wasm) {
  const bytes = new Uint8Array(wasm);
  let position = 8; // 마법의 수와 버전.
  const number = () => {
    let value = 0, shift = 0, byte;
    do {
      byte = bytes[position++];
      value += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    } while (byte & 0x80);
    return value;
  };
  const skipName = () => {
    // 길이를 먼저 읽는다. `position += number()`라고 쓰면 길이를 읽기 전의
    // 위치에 더하게 된다.
    const length = number();
    position += length;
  };
  while (position < bytes.length) {
    const id = bytes[position++];
    const size = number();
    if (id !== 2) {
      position += size; // 들여오기 구획이 아니면 건너뛴다.
      continue;
    }
    for (let count = number(); count > 0; count--) {
      skipName();
      skipName();
      const kind = bytes[position++];
      if (kind === 0) number(); // 함수: 타입 번호.
      else if (kind === 1) { position++; const flags = number(); number(); if (flags & 1) number(); } // 표.
      else if (kind === 2) {
        const flags = number();
        const initial = number();
        const maximum = flags & 1 ? number() : undefined;
        return { initial, maximum, shared: Boolean(flags & 2) };
      } else if (kind === 3) position += 2; // 전역.
    }
    return null;
  }
  return null;
}
