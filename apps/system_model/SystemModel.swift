// 운영체제에 들어 있는 언어 모델을 부른다 (47장).
//
// Apple의 기기에는 운영체제가 관리하는 언어 모델이 있고, 앱은
// FoundationModels 프레임워크로 그것을 부른다. 앱은 모델 파일을 갖고 있지
// 않다. 이 프로그램은 질문 하나를 보내고, 첫 글이 오기까지의 시간과 전체
// 시간을 적는다.
//
// 빌드와 실행: sh scripts/bench_ch47.sh system   (macOS 26 이상)

import Foundation
import FoundationModels

@main
struct SystemModel {
    static func main() async {
        let prompt = CommandLine.arguments.dropFirst().joined(separator: " ")
        // ANCHOR: availability
        // 모델이 이 기기에 있는지부터 묻는다. 기기가 지원하지 않거나, 사용자가
        // 기능을 꺼 두었거나, 아직 내려받는 중일 수 있다.
        guard case .available = SystemLanguageModel.default.availability else {
            print("system 쓸수없음 \(SystemLanguageModel.default.availability)")
            return
        }
        // ANCHOR_END: availability
        guard !prompt.isEmpty else {
            print("system 쓸수있음")
            return
        }
        // ANCHOR: stream
        let session = LanguageModelSession()
        let started = Date()
        var first: TimeInterval?
        var text = ""
        do {
            // 답이 만들어지는 대로 받는다. 받을 때마다 지금까지의 답 전체가 온다.
            for try await partial in session.streamResponse(to: prompt) {
                if first == nil {
                    first = Date().timeIntervalSince(started)
                }
                text = partial.content
            }
        } catch {
            print("system 오류 \(error)")
            return
        }
        // ANCHOR_END: stream
        let total = Date().timeIntervalSince(started)
        let words = text.split(whereSeparator: { $0.isWhitespace }).count
        print(
            String(
                format: "system 첫글 %.0f 전체 %.0f 글자 %d 낱말 %d",
                (first ?? total) * 1e3, total * 1e3, text.count, words))
        print(text)
    }
}
