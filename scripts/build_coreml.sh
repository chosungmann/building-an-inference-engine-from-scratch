#!/bin/sh
# 45장: badak이 Core ML을 부를 때 쓰는 동적 라이브러리를 만든다. macOS에서만
# 된다.
#
# 쓰는 법: sh scripts/build_coreml.sh
# 결과: target/coreml/libbadak_coreml.dylib
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/target/coreml"
mkdir -p "$OUT"
xcrun swiftc -O -parse-as-library -module-name BadakCoreML -emit-library \
    -o "$OUT/libbadak_coreml.dylib" "$ROOT/apps/coreml/BadakCoreML.swift"
rm -f "$OUT"/BadakCoreML.swiftmodule "$OUT"/BadakCoreML.swiftdoc \
    "$OUT"/BadakCoreML.abi.json "$OUT"/BadakCoreML.swiftsourceinfo
echo "만들었습니다: target/coreml/libbadak_coreml.dylib"
