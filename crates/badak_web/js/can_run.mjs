// 이 Node가 주어진 빌드(scalar, simd, relaxed, …_threads)를 올릴 수 있는가.
// 있으면 0으로, 없으면 1로 끝난다. scripts/check_web.sh 가 쓴다.
import { detect } from './features.js';

const features = detect();
const name = process.argv[2] ?? '';
const able = name.startsWith('relaxed') ? features.relaxed : name.startsWith('simd') ? features.simd : true;
process.exit(able ? 0 : 1);
