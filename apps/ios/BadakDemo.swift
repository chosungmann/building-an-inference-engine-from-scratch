// badak을 부르는 가장 작은 iOS 앱 (44장).
//
// 앱 꾸러미에 든 모델로 꾸러미에 든 사진을 분류해 화면에 보여 준다.
// 같은 결과를 "BADAK"으로 시작하는 줄로도 적는다. 맥에서 돌린 결과와
// 글자 그대로 견주려는 것이다.

import SwiftUI

// ANCHOR: wrapper
/// C의 오류 숫자를 Swift의 오류로 바꾼 것.
struct BadakError: Error, CustomStringConvertible {
    let status: Int32
    let description: String

    /// 숫자가 0이 아니면, 같은 스레드에서 설명을 읽어 던진다.
    static func check(_ status: Int32) throws {
        guard status != BADAK_OK else { return }
        var buffer = [CChar](repeating: 0, count: 512)
        _ = badak_last_error(&buffer, buffer.count)
        throw BadakError(status: status, description: String(cString: buffer))
    }
}

/// C의 분류기를 감싼 것. 사라질 때 닫는다.
final class Classifier {
    private let handle: OpaquePointer

    init(path: String) throws {
        var handle: OpaquePointer?
        try BadakError.check(badak_classifier_open(path, &handle))
        self.handle = handle!
    }

    deinit {
        badak_classifier_close(handle)
    }

    /// JPEG 파일의 바이트를 받아 확률이 높은 부류부터 돌려준다.
    func classify(jpeg: Data, count: Int) throws -> [badak_score] {
        var top = [badak_score](repeating: badak_score(), count: count)
        var written = 0
        try jpeg.withUnsafeBytes { bytes in
            try BadakError.check(
                badak_classifier_classify_jpeg(
                    handle, bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count,
                    &top, count, &written))
        }
        return Array(top.prefix(written))
    }
}
// ANCHOR_END: wrapper

struct Outcome {
    var lines: [String] = []
    var failure: String?
}

/// 꾸러미에 든 것들로 분류를 한 번 한다.
func classifyBundledPhoto() -> Outcome {
    var outcome = Outcome()
    do {
        let bundle = Bundle.main
        guard let model = bundle.path(forResource: "mobilenetv2", ofType: "tflite"),
            let photo = bundle.url(forResource: "cup", withExtension: "jpg"),
            let labels = bundle.url(forResource: "labels", withExtension: "txt")
        else {
            throw BadakError(status: -1, description: "꾸러미에 파일이 없습니다")
        }
        let names = try String(contentsOf: labels, encoding: .utf8).split(separator: "\n")
        let classifier = try Classifier(path: model)
        let top = try classifier.classify(jpeg: try Data(contentsOf: photo), count: 5)
        print("BADAK version \(String(cString: badak_version()))")
        var cpu = [CChar](repeating: 0, count: 256)
        _ = badak_cpu_features(&cpu, cpu.count)
        print("BADAK cpu \(String(cString: cpu))")
        for score in top {
            let bits = String(format: "%08x", score.probability.bitPattern)
            print("BADAK top \(score.index) \(bits)")
            let percent = String(format: "%.1f%%", 100 * score.probability)
            outcome.lines.append("\(names[Int(score.index)])  \(percent)")
        }
    } catch {
        print("BADAK error \(error)")
        outcome.failure = "\(error)"
    }
    print("BADAK done")
    return outcome
}

struct ContentView: View {
    @State private var outcome = Outcome()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("badak \(String(cString: badak_version()))").font(.headline)
            if let path = Bundle.main.path(forResource: "cup", ofType: "jpg"),
                let image = UIImage(contentsOfFile: path)
            {
                Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 280)
            }
            ForEach(outcome.lines, id: \.self) { line in
                Text(line).font(.system(.body, design: .monospaced))
            }
            if let failure = outcome.failure {
                Text(failure).foregroundStyle(.red)
            }
            Spacer()
        }
        .padding()
        .task {
            // 엔진은 화면을 그리는 스레드 밖에서 돌린다.
            outcome = await Task.detached { classifyBundledPhoto() }.value
        }
    }
}

@main
struct BadakDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
