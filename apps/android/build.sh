#!/bin/sh
# Android 앱을 Gradle 없이 만든다 (44장).
#
# 쓰는 법: ANDROID_HOME=<SDK> JAVA_HOME=<JDK> sh apps/android/build.sh [run]
#
# 하는 일:
#   1. 엔진을 Android용 동적 라이브러리(.so)로 빌드한다. 링커는 NDK의 것.
#   2. Java 두 파일을 컴파일하고(javac), Android가 읽는 꼴로 바꾼다(d8).
#   3. 선언 파일과 자산으로 꾸러미의 뼈대를 만들고(aapt2), 코드와 .so를
#      넣고, 줄을 맞추고(zipalign), 서명한다(apksigner).
#   4. run 을 주면 켜져 있는 에뮬레이터에 넣고 돌려, 맥에서 돌린 결과와
#      견준다.
#
# 필요한 것: Android SDK의 build-tools, platforms, platform-tools, ndk.
# 모델과 사진은 scripts/download_ch27.sh 가 받아 둔 것을 쓴다.
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ANDROID_HOME:?Android SDK의 자리를 ANDROID_HOME에 주세요}"
: "${JAVA_HOME:?JDK의 자리를 JAVA_HOME에 주세요}"
latest() { ls -d "$1"/* | sort -V | tail -n 1; }
NDK="${ANDROID_NDK_HOME:-$(latest "$ANDROID_HOME/ndk")}"
TOOLS="$(latest "$ANDROID_HOME/build-tools")"
PLATFORM="$(latest "$ANDROID_HOME/platforms")/android.jar"
LLVM="$(ls -d "$NDK"/toolchains/llvm/prebuilt/*)/bin"
ADB="$ANDROID_HOME/platform-tools/adb"
PATH="$JAVA_HOME/bin:$PATH"
TARGET=aarch64-linux-android
# 앱이 돌 수 있는 가장 낮은 Android. 선언 파일의 minSdkVersion과 같아야 한다.
API=24
OUT="$ROOT/target/apps/android"
MODELS="${BADAK_MODELS:-$ROOT/models}"

cd "$ROOT"
rm -rf "$OUT"
mkdir -p "$OUT/classes" "$OUT/assets" "$OUT/lib/arm64-v8a"

# 1. 엔진. cargo는 링커를 모르므로 NDK의 clang을 알려 준다. 페이지가 16KB인
#    기기에서도 올라가도록 줄 맞춤을 16KB로 한다.
CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$LLVM/$TARGET$API-clang" \
CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="-C link-arg=-Wl,-z,max-page-size=16384" \
    cargo build --release -p badak_ffi --target "$TARGET"
"$LLVM/llvm-strip" --strip-unneeded -o "$OUT/lib/arm64-v8a/libbadak_ffi.so" \
    "target/$TARGET/release/libbadak_ffi.so"

# 2. Java.
javac --release 11 -classpath "$PLATFORM" \
    -d "$OUT/classes" $(find apps/android/java -name '*.java')
"$TOOLS/d8" --release --min-api "$API" --lib "$PLATFORM" --output "$OUT" \
    $(find "$OUT/classes" -name '*.class')

# 3. 꾸러미. -0 은 그 확장자의 자산을 누르지 않고 넣으라는 뜻이다. 눌린
#    파일은 메모리에 비출 수 없다.
cp "$MODELS/vision_tflite/mobilenetv2.tflite" "$MODELS/vision_tflite/labels.txt" \
    "$MODELS/images/cup.jpg" "$OUT/assets/"
"$TOOLS/aapt2" link -o "$OUT/unaligned.apk" \
    --manifest apps/android/AndroidManifest.xml -I "$PLATFORM" \
    -A "$OUT/assets" -0 tflite -0 jpg
(cd "$OUT" && zip -q unaligned.apk classes.dex && zip -q -0 unaligned.apk lib/arm64-v8a/libbadak_ffi.so)
# -P 16: 눌리지 않은 .so를 16KB의 경계에 맞춘다. 4: 다른 눌리지 않은 파일을
# 4바이트의 경계에 맞춘다.
"$TOOLS/zipalign" -f -P 16 4 "$OUT/unaligned.apk" "$OUT/aligned.apk"
# 서명. 개발용 열쇠를 처음 한 번 만든다. 가게에 올릴 앱의 열쇠가 아니다.
# 열쇠가 바뀌면 기기는 같은 앱의 새 판으로 받아 주지 않으므로 남겨 둔다.
KEYS="$ROOT/target/apps/android_debug.keystore"
[ -f "$KEYS" ] || keytool -genkeypair -keystore "$KEYS" -storepass android -keypass android \
    -alias debug -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=badak debug" > /dev/null 2>&1
"$TOOLS/apksigner" sign --ks "$KEYS" --ks-pass pass:android \
    --out "$OUT/badak_demo.apk" "$OUT/aligned.apk"
echo "만들었습니다: ${OUT#"$ROOT"/}/badak_demo.apk"
unzip -lv "$OUT/badak_demo.apk" | awk 'NR > 3 && $8 != "" && $1 ~ /^[0-9]+$/ \
    { printf "  %8d → %8d 바이트  %-7s %s\n", $1, $3, $2, $8 }'

[ "${1:-}" = run ] || exit 0

# 맥에서 같은 일을 한 결과.
cargo build --release -p badak_ffi
cc -std=c11 -o "$OUT/host" crates/badak_ffi/tests/host.c \
    -I crates/badak_ffi/include target/release/libbadak_ffi.a
"$OUT/host" "$MODELS/vision_tflite/mobilenetv2.tflite" "$MODELS/images/cup.jpg" \
    | grep '^BADAK top' > "$OUT/expected.txt"

"$ADB" install -r "$OUT/badak_demo.apk" > /dev/null
"$ADB" logcat -c > /dev/null 2>&1
"$ADB" shell am start -n book.badak.demo/.MainActivity > /dev/null
for _ in $(seq 60); do
    "$ADB" logcat -d -s badak:I | grep -q 'BADAK done' && break
    sleep 1
done
"$ADB" logcat -d -s badak:I | sed -n 's/.*\(BADAK .*\)/\1/p' | tr -d '\r' > "$OUT/console.txt"
cat "$OUT/console.txt"
grep '^BADAK top' "$OUT/console.txt" > "$OUT/actual.txt" || true
if cmp -s "$OUT/expected.txt" "$OUT/actual.txt"; then
    echo "맥에서 돌린 결과와 비트까지 같습니다."
else
    echo "맥에서 돌린 결과와 다릅니다:" >&2
    diff "$OUT/expected.txt" "$OUT/actual.txt" >&2 || true
    exit 1
fi
