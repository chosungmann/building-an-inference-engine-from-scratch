// 이 브라우저가 무엇을 할 수 있는지 묻고, 올릴 빌드를 고른다 (48장).
//
// WebAssembly에는 "이 명령을 쓸 수 있는가"를 실행 중에 묻는 방법이 없다.
// 모르는 명령이 하나라도 든 모듈은 올리는 단계에서 거절된다. 그래서 그
// 명령 하나만 든 아주 작은 모듈을 만들어 "이것을 올릴 수 있는가"를 묻는다.

// 함수 하나짜리 모듈을 만든다. body는 그 함수의 몸이다.
function probe(body) {
  const code = [0, ...body, 0x0b]; // 지역 변수 없음, 몸, 끝.
  return new Uint8Array([
    0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, // "\0asm", 버전 1.
    1, 4, 1, 0x60, 0, 0, // 타입 구획: 인자도 결과도 없는 함수.
    3, 2, 1, 0, // 함수 구획: 그 타입의 함수 하나.
    10, code.length + 2, 1, code.length, ...code, // 코드 구획.
  ]);
}

const zeros = new Array(16).fill(0);
const constant = [0xfd, 0x0c, ...zeros]; // v128.const 0
// v128.const 하나를 만들고 버린다.
const SIMD = probe([...constant, 0x1a]);
// 셋을 만들어 f32x4.relaxed_madd(0xfd 0x105)를 하고 버린다.
const RELAXED = probe([...constant, ...constant, ...constant, 0xfd, 0x85, 0x02, 0x1a]);

// ANCHOR: detect
export function detect() {
  return {
    simd: WebAssembly.validate(SIMD),
    relaxed: WebAssembly.validate(RELAXED),
    // 공유 메모리는 "교차 출처 격리"된 페이지에서만 쓸 수 있다.
    threads: typeof SharedArrayBuffer !== 'undefined' && globalThis.crossOriginIsolated === true,
    cores: globalThis.navigator?.hardwareConcurrency ?? 1,
  };
}

// 여섯 빌드 가운데 이 브라우저에서 가장 빠를 것의 이름.
export function pick(features, { threads = true, relaxed = true } = {}) {
  const kernel = features.relaxed && relaxed ? 'relaxed' : features.simd ? 'simd' : 'scalar';
  return `badak_${kernel}${features.threads && threads ? '_threads' : ''}.wasm`;
}
// ANCHOR_END: detect
