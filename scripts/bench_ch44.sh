#!/bin/sh
# 44장: 엔진이 앱에 더하는 크기를 빌드 설정마다 재고, panic=abort 가 문에서
# 무엇을 바꾸는지 본다.
#
# 쓰는 법: ANDROID_HOME=<SDK> sh scripts/bench_ch44.sh
# 결과: bench/ch44/sizes.txt
#
# 속도는 재지 않는다. 크기와 동작만 본다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ANDROID_HOME:?Android SDK의 자리를 ANDROID_HOME에 주세요}"
latest() { ls -d "$1"/* | sort -V | tail -n 1; }
NDK="${ANDROID_NDK_HOME:-$(latest "$ANDROID_HOME/ndk")}"
LLVM="$(ls -d "$NDK"/toolchains/llvm/prebuilt/*)/bin"
OUT="$ROOT/bench/ch44"
WORK="$ROOT/target/ch44_sizes"
mkdir -p "$OUT" "$WORK"
cd "$ROOT"

LTO='--config profile.release.lto="fat" --config profile.release.codegen-units=1'
ABORT='--config profile.release.panic="abort"'

# 이름과 cargo에 더 줄 설정.
variants() {
    echo "기본"
    echo "lto $LTO"
    echo "lto_abort $LTO $ABORT"
}

{
    echo "# 이름 대상 파일 바이트"
    variants | while read -r name flags; do
        [ "$name" = 기본 ] && flags=""
        # Android: 기호를 떼어 낸 .so.
        # shellcheck disable=SC2086
        env CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$LLVM/aarch64-linux-android24-clang" \
            CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="-C link-arg=-Wl,-z,max-page-size=16384" \
            cargo build --quiet --release -p badak_ffi --target aarch64-linux-android \
            --target-dir "$WORK/$name" $flags
        so="$WORK/$name/aarch64-linux-android/release/libbadak_ffi.so"
        "$LLVM/llvm-strip" --strip-unneeded -o "$WORK/$name.so" "$so"
        echo "$name android libbadak_ffi.so $(wc -c < "$WORK/$name.so" | tr -d ' ')"
        gzip -9 -c "$WORK/$name.so" > "$WORK/$name.so.gz"
        echo "$name android libbadak_ffi.so.gz $(wc -c < "$WORK/$name.so.gz" | tr -d ' ')"

        # iOS: Swift 앱에 묶은 실행 파일에서 기호를 떼어 낸 것.
        # shellcheck disable=SC2086
        cargo build --quiet --release -p badak_ffi --target aarch64-apple-ios \
            --target-dir "$WORK/$name" $flags
        # 링커에게 쓰이지 않는 함수를 버리라고 할 때(-dead_strip)와 아닐 때.
        for dead in "" "-Xlinker -dead_strip"; do
            # shellcheck disable=SC2086
            xcrun -sdk iphoneos swiftc -target arm64-apple-ios17.0 -parse-as-library -O $dead \
                -import-objc-header crates/badak_ffi/include/badak.h apps/ios/BadakDemo.swift \
                "$WORK/$name/aarch64-apple-ios/release/libbadak_ffi.a" -o "$WORK/$name.ios"
            strip "$WORK/$name.ios" 2> /dev/null
            echo "$name ios BadakDemo${dead:+_dead_strip} $(wc -c < "$WORK/$name.ios" | tr -d ' ')"
        done
    done
    # 견줄 것: 엔진을 부르지 않는 같은 모양의 Swift 앱.
    cat > "$WORK/empty.swift" <<'SWIFT'
import SwiftUI

@main
struct EmptyApp: App {
    var body: some Scene {
        WindowGroup { Text("empty") }
    }
}
SWIFT
    xcrun -sdk iphoneos swiftc -target arm64-apple-ios17.0 -parse-as-library -O \
        -Xlinker -dead_strip "$WORK/empty.swift" -o "$WORK/empty.ios"
    strip "$WORK/empty.ios" 2> /dev/null
    echo "엔진없음 ios EmptyApp $(wc -c < "$WORK/empty.ios" | tr -d ' ')"
} > "$OUT/sizes.txt"
cat "$OUT/sizes.txt"

# panic=abort: 맥에서 본다. 문 안에서 일부러 패닉을 낸다.
{
    for name in 기본 lto_abort; do
        flags=""
        [ "$name" = lto_abort ] && flags="$LTO $ABORT"
        # shellcheck disable=SC2086
        cargo build --quiet --release -p badak_ffi --target-dir "$WORK/host_$name" $flags
        cc -o "$WORK/probe_$name" crates/badak_ffi/tests/panic_probe.c \
            "$WORK/host_$name/release/libbadak_ffi.a"
        echo "## $name"
        status=0
        "$WORK/probe_$name" 2>&1 | grep -v '^note:' | sed "s|$ROOT/||" || status=$?
        # 파이프의 끝이 아니라 프로그램이 어떻게 끝났는지 따로 본다.
        ("$WORK/probe_$name" > /dev/null 2>&1) 2> /dev/null || status=$?
        echo "프로그램이 끝난 값: $status"
    done
} > "$OUT/panic.txt"
cat "$OUT/panic.txt"
