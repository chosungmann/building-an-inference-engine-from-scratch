// badak이 Core ML을 부르는 다리 (45장).
//
// Core ML은 Swift와 Objective-C로만 부를 수 있다. 이 파일은 C 함수 넷을
// 내놓는 동적 라이브러리가 되고, 엔진(crates/badak_graph/src/coreml.rs)이
// 실행할 때 그것을 올려서 부른다. 44장과 방향이 반대다. 거기서는 Swift가
// Rust를 불렀고, 여기서는 Rust가 Swift를 부른다.
//
// 빌드: sh scripts/build_coreml.sh

import CoreML
import Foundation

/// 올려 둔 모델. 컴파일한 폴더는 모델을 닫을 때 지운다.
final class Loaded {
    let model: MLModel
    let compiled: URL

    init(model: MLModel, compiled: URL) {
        self.model = model
        self.compiled = compiled
    }

    deinit {
        try? FileManager.default.removeItem(at: compiled)
    }
}

/// 글을 부르는 쪽이 준 자리에 채운다. 끝에 0을 붙인다.
func fill(_ buffer: UnsafeMutablePointer<CChar>?, _ capacity: Int, _ text: String) {
    guard let buffer, capacity > 0 else { return }
    let bytes = Array(text.utf8.prefix(capacity - 1))
    for (index, byte) in bytes.enumerated() {
        buffer[index] = CChar(bitPattern: byte)
    }
    buffer[bytes.count] = 0
}

func configuration(_ units: Int32) -> MLModelConfiguration {
    let configuration = MLModelConfiguration()
    // 0: CPU만, 1: CPU와 GPU, 2: 모두, 3: CPU와 뉴럴 엔진.
    configuration.computeUnits = MLComputeUnits(rawValue: Int(units)) ?? .all
    return configuration
}

// ANCHOR: load
/// `.mlmodel` 파일을 이 기기에 맞게 컴파일하고 올린다.
@_cdecl("badak_coreml_load")
public func badakCoreMLLoad(
    _ path: UnsafePointer<CChar>, _ units: Int32,
    _ message: UnsafeMutablePointer<CChar>?, _ capacity: Int
) -> UnsafeMutableRawPointer? {
    do {
        let source = URL(fileURLWithPath: String(cString: path))
        // 1. 파일을 Core ML이 쓰는 꼴(.mlmodelc 폴더)로 바꾼다.
        let compiled = try MLModel.compileModel(at: source)
        // 2. 올린다. 어느 장치에서 돌릴지는 여기에서 정해진다.
        let model = try MLModel(contentsOf: compiled, configuration: configuration(units))
        return Unmanaged.passRetained(Loaded(model: model, compiled: compiled)).toOpaque()
    } catch {
        fill(message, capacity, "\(error)")
        return nil
    }
}
// ANCHOR_END: load

@_cdecl("badak_coreml_free")
public func badakCoreMLFree(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<Loaded>.fromOpaque(handle).release()
}

struct ShimError: Error, CustomStringConvertible {
    let description: String
}

// ANCHOR: run
/// 입력들을 넣고 출력들을 받는다. 모든 값은 네 축의 `Float` 배열이다.
@_cdecl("badak_coreml_run")
public func badakCoreMLRun(
    _ handle: UnsafeMutableRawPointer,
    _ inputCount: Int32,
    _ inputNames: UnsafePointer<UnsafePointer<CChar>>,
    _ inputs: UnsafePointer<UnsafePointer<Float>>,
    _ shapes: UnsafePointer<Int64>,
    _ outputCount: Int32,
    _ outputNames: UnsafePointer<UnsafePointer<CChar>>,
    _ outputs: UnsafePointer<UnsafeMutablePointer<Float>>,
    _ lengths: UnsafePointer<Int>,
    _ message: UnsafeMutablePointer<CChar>?, _ capacity: Int
) -> Int32 {
    let loaded = Unmanaged<Loaded>.fromOpaque(handle).takeUnretainedValue()
    do {
        var features: [String: MLFeatureValue] = [:]
        for index in 0..<Int(inputCount) {
            let shape = (0..<4).map { Int(shapes[index * 4 + $0]) }
            let strides = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]
            // 엔진의 메모리를 복사하지 않고 그대로 감싼다.
            let array = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: inputs[index]),
                shape: shape.map { NSNumber(value: $0) },
                dataType: .float32,
                strides: strides.map { NSNumber(value: $0) },
                deallocator: nil)
            features[String(cString: inputNames[index])] = MLFeatureValue(multiArray: array)
        }
        let result = try loaded.model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: features))
        for index in 0..<Int(outputCount) {
            let name = String(cString: outputNames[index])
            guard let array = result.featureValue(for: name)?.multiArrayValue else {
                throw ShimError(description: "출력 \(name)이 없습니다")
            }
            guard array.count == lengths[index] else {
                throw ShimError(
                    description: "출력 \(name)의 길이가 \(array.count)입니다 (\(lengths[index])을 기다림)")
            }
            let target = UnsafeMutableBufferPointer(start: outputs[index], count: array.count)
            if array.dataType == .float32 {
                array.withUnsafeBufferPointer(ofType: Float.self) { source in
                    _ = target.update(fromContentsOf: source)
                }
            } else {
                for position in 0..<array.count {
                    target[position] = array[position].floatValue
                }
            }
        }
        return 0
    } catch {
        fill(message, capacity, "\(error)")
        return 1
    }
}
// ANCHOR_END: run

func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "NPU"
    @unknown default: return "?"
    }
}

// ANCHOR: plan
/// 층마다 Core ML이 어느 장치에서 돌리려 하는지 적는다. 한 줄에 층 하나,
/// "이름<탭>고른 장치<탭>돌릴 수 있는 장치들"이다. 적은 층의 수를 돌려준다.
@_cdecl("badak_coreml_plan")
public func badakCoreMLPlan(
    _ path: UnsafePointer<CChar>, _ units: Int32,
    _ buffer: UnsafeMutablePointer<CChar>?, _ capacity: Int
) -> Int64 {
    let source = URL(fileURLWithPath: String(cString: path))
    let settings = configuration(units)
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var lines: [String] = []
    nonisolated(unsafe) var failure: String?
    Task.detached {
        defer { done.signal() }
        do {
            let compiled = try await MLModel.compileModel(at: source)
            defer { try? FileManager.default.removeItem(at: compiled) }
            let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: settings)
            guard case .neuralNetwork(let network) = plan.modelStructure else {
                failure = "신경망 형식의 모델이 아닙니다"
                return
            }
            for layer in network.layers {
                guard let usage = plan.deviceUsage(for: layer) else {
                    lines.append("\(layer.name)\t-\t-")
                    continue
                }
                let supported = usage.supported.map(deviceName).joined(separator: ",")
                lines.append("\(layer.name)\t\(deviceName(usage.preferred))\t\(supported)")
            }
        } catch {
            failure = "\(error)"
        }
    }
    done.wait()
    if let failure {
        fill(buffer, capacity, failure)
        return -1
    }
    fill(buffer, capacity, lines.joined(separator: "\n"))
    return Int64(lines.count)
}
// ANCHOR_END: plan
