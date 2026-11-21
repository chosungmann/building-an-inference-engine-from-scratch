// badak을 부르는 가장 작은 iOS 앱 (44장).
//
// 앱 번들에 든 모델로 번들에 든 사진을 분류해 화면에 보여 준다.
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

// ANCHOR: llm_wrapper
/// C의 언어 모델을 감싼 것 (47장).
final class Llm {
    private let handle: OpaquePointer

    init(directory: String) throws {
        var handle: OpaquePointer?
        // 0: 파일에 적힌 그대로. 파일을 메모리에 매핑한다.
        try BadakError.check(badak_llm_open(directory, 0, &handle))
        self.handle = handle!
    }

    deinit {
        badak_llm_close(handle)
    }

    /// 조각을 받을 Swift의 함수를 담아 두는 상자. C의 함수는 Swift의
    /// 클로저를 직접 받지 못한다. 상자의 주소를 `user`로 넘기고, C의 함수
    /// 안에서 그 주소로 상자를 되찾는다.
    private final class Listener {
        let body: (String) -> Bool
        init(_ body: @escaping (String) -> Bool) { self.body = body }
    }

    /// 질문 하나에 답한다. 글이 완성되는 대로 `onPiece`를 부르고, 그것이
    /// 거짓을 돌려주면 그만 쓴다.
    func chat(
        system: String?, _ question: String, maxTokens: Int, onPiece: @escaping (String) -> Bool
    ) throws -> Int {
        let listener = Listener(onPiece)
        var written = 0
        let status = withExtendedLifetime(listener) {
            badak_llm_chat(
                handle, system, question, maxTokens,
                { user, piece, length in
                    // 여기는 C의 함수다. 바깥의 변수를 잡을 수 없다.
                    let listener = Unmanaged<Listener>.fromOpaque(user!).takeUnretainedValue()
                    let bytes = UnsafeRawBufferPointer(start: piece, count: length)
                    return listener.body(String(decoding: bytes, as: UTF8.self)) ? 0 : 1
                },
                Unmanaged.passUnretained(listener).toOpaque(), &written)
        }
        try BadakError.check(status)
        return written
    }
}
// ANCHOR_END: llm_wrapper

struct Outcome {
    var lines: [String] = []
    var failure: String?
}

let question = "What is the capital of France?"
/// SmolLM2가 학습할 때 쓴 기본 지시문. 모델의 대화 틀에 적혀 있다.
let systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

/// 번들에 언어 모델이 들어 있으면 질문 하나를 보낸다. 조각이 올 때마다
/// `show`를 부른다.
func askBundledModel(show: @escaping @Sendable (String) -> Void) {
    guard let directory = Bundle.main.resourceURL?.appendingPathComponent("smollm2_135m").path,
        FileManager.default.fileExists(atPath: directory + "/model.safetensors")
    else {
        print("BADAK llm 없음")
        return
    }
    do {
        let llm = try Llm(directory: directory)
        var answer = ""
        let written = try llm.chat(system: systemPrompt, question, maxTokens: 32) { piece in
            answer += piece
            show(answer)
            return true
        }
        let hex = answer.utf8.map { String(format: "%02x", $0) }.joined()
        print("BADAK llm \(written) \(hex)")
    } catch {
        print("BADAK error \(error)")
        show("\(error)")
    }
    print("BADAK llm-done")
}

/// 번들에 든 것들로 분류를 한 번 한다.
func classifyBundledPhoto() -> Outcome {
    var outcome = Outcome()
    do {
        let bundle = Bundle.main
        guard let model = bundle.path(forResource: "mobilenetv2", ofType: "tflite"),
            let photo = bundle.url(forResource: "cup", withExtension: "jpg"),
            let labels = bundle.url(forResource: "labels", withExtension: "txt")
        else {
            throw BadakError(status: -1, description: "번들에 파일이 없습니다")
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
    @State private var answer = ""

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
            if !answer.isEmpty {
                Divider()
                Text(question).font(.subheadline).bold()
                Text(answer).font(.subheadline)
            }
            Spacer()
        }
        .padding()
        .task {
            // 엔진은 화면을 그리는 스레드 밖에서 돌린다.
            outcome = await Task.detached { classifyBundledPhoto() }.value
            // 토큰이 하나 올 때마다 화면을 고친다. 화면은 주 스레드에서만
            // 고칠 수 있으므로 그리로 넘긴다.
            await Task.detached {
                askBundledModel { text in
                    Task { @MainActor in answer = text }
                }
            }.value
        }
    }
}

@main
struct BadakDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
