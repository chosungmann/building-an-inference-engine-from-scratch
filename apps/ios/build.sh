#!/bin/sh
# iOS 시뮬레이터용 앱을 Xcode 프로젝트 없이 만든다 (44장).
#
# 쓰는 법: sh apps/ios/build.sh [run|device]
#
# 하는 일:
#   1. 엔진을 iOS 시뮬레이터용 정적 라이브러리로 빌드한다.
#   2. Swift 한 파일을 컴파일해 그 라이브러리와 묶는다.
#   3. 실행 파일, Info.plist, 모델, 사진을 폴더 하나(.app)에 모은다.
#   4. run 을 주면 켜져 있는 시뮬레이터에 넣고 돌려, 맥에서 돌린 결과와
#      견준다. device 를 주면 실기기용으로 묶이는지 보고, 기기용과
#      시뮬레이터용 라이브러리를 한 묶음(XCFramework)으로 만든다.
#
# 모델과 사진은 scripts/download_ch27.sh 가 받아 둔 것을 쓴다.
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET=aarch64-apple-ios-sim
OUT="$ROOT/target/apps/ios"
APP="$OUT/BadakDemo.app"
MODELS="${BADAK_MODELS:-$ROOT/models}"

cd "$ROOT"
cargo build --release -p badak_ffi --target "$TARGET"

rm -rf "$APP"
mkdir -p "$APP"
# -import-objc-header 가 C 헤더의 선언을 Swift에 그대로 보여 준다.
xcrun -sdk iphonesimulator swiftc \
    -target arm64-apple-ios17.0-simulator \
    -parse-as-library -O -Xlinker -dead_strip \
    -import-objc-header crates/badak_ffi/include/badak.h \
    apps/ios/BadakDemo.swift \
    "target/$TARGET/release/libbadak_ffi.a" \
    -o "$APP/BadakDemo"
cp apps/ios/Info.plist "$APP/"
cp "$MODELS/vision_tflite/mobilenetv2.tflite" "$MODELS/vision_tflite/labels.txt" \
    "$MODELS/images/cup.jpg" "$APP/"
# 시뮬레이터는 서명이 없는 앱을 받지 않는다. 신원 없이("-") 서명한다.
codesign --force --sign - "$APP" 2> /dev/null
echo "만들었습니다: ${APP#"$ROOT"/}"
du -sk "$APP"/* | awk '{ printf "  %6d KB  %s\n", $1, $2 }' | sed "s|$APP/||"

if [ "${1:-}" = device ]; then
    # 실기기용. 같은 Swift 파일을 iPhone의 SDK로 컴파일해 묶어 본다. 기기에
    # 넣으려면 Apple이 내준 인증서로 서명해야 하므로 여기에서는 묶이는
    # 데까지만 본다.
    cargo build --release -p badak_ffi --target aarch64-apple-ios
    xcrun -sdk iphoneos swiftc -target arm64-apple-ios17.0 -parse-as-library -O \
        -Xlinker -dead_strip -import-objc-header crates/badak_ffi/include/badak.h \
        apps/ios/BadakDemo.swift target/aarch64-apple-ios/release/libbadak_ffi.a \
        -o "$OUT/BadakDemo_device"
    file "$OUT/BadakDemo_device" | sed "s|$OUT/||"
    # 기기용과 시뮬레이터용 라이브러리를 헤더와 함께 한 묶음으로 만든다.
    # Xcode 프로젝트에 끌어다 넣는 것은 이 묶음이다.
    rm -rf "$OUT/Badak.xcframework"
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libbadak_ffi.a -headers crates/badak_ffi/include \
        -library "target/$TARGET/release/libbadak_ffi.a" -headers crates/badak_ffi/include \
        -output "$OUT/Badak.xcframework" > /dev/null
    find "$OUT/Badak.xcframework" -maxdepth 2 | sed "s|$OUT/||" | sort
    exit 0
fi
[ "${1:-}" = run ] || exit 0

# 맥에서 같은 일을 한 결과. C 프로그램이 같은 라이브러리의 맥용 빌드를 부른다.
cargo build --release -p badak_ffi
cc -std=c11 -o "$OUT/host" crates/badak_ffi/tests/host.c \
    -I crates/badak_ffi/include target/release/libbadak_ffi.a
"$OUT/host" "$MODELS/vision_tflite/mobilenetv2.tflite" "$MODELS/images/cup.jpg" \
    | grep '^BADAK top' > "$OUT/expected.txt"

xcrun simctl install booted "$APP"
# 앱이 적는 줄을 받아, 다 적었다는 줄이 나오면 끝낸다.
xcrun simctl launch --console-pty --terminate-running-process booted book.badak.demo \
    > "$OUT/console.txt" 2>&1 &
for _ in $(seq 60); do
    grep -q '^BADAK done' "$OUT/console.txt" 2> /dev/null && break
    sleep 1
done
# 시뮬레이터가 끼워 넣는 글자를 걸러, 앱이 적은 줄만 남긴다.
tr -d '\r' < "$OUT/console.txt" | grep -o 'BADAK [a-z]* .*' || true
grep -o 'BADAK top [0-9]* [0-9a-f]\{8\}' "$OUT/console.txt" > "$OUT/actual.txt" || true
if cmp -s "$OUT/expected.txt" "$OUT/actual.txt"; then
    echo "맥에서 돌린 결과와 비트까지 같습니다."
else
    echo "맥에서 돌린 결과와 다릅니다:" >&2
    diff "$OUT/expected.txt" "$OUT/actual.txt" >&2 || true
    exit 1
fi
