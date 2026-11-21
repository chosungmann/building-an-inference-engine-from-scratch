"""51장: PyTorch 모델을 Python 밖으로 내보내는 길들.

실행:
    sh scripts/download_mnist.sh
    pip install -r reference/requirements_ch51.txt
    python reference/ch51_export.py [branch|export|onnx|executorch|ort ...]

    branch      입력에 따라 갈리는 코드를 세 가지 방법으로 내보내 본다
    export      MNIST의 MLP를 torch.export로 내보내 그래프를 적는다
    onnx        같은 모델을 ONNX 파일로 내보낸다. badak이 읽는다
    executorch  같은 모델을 ExecuTorch의 .pte로 내보내고 돌려 본다
    ort         SmolLM2-135M의 ONNX 파일을 ONNX Runtime으로 돌려 속도를 잰다
                (sh scripts/download_ch50.sh 가 받아 둔 파일)

출력: models/ch51/ 아래의 파일들과, 표준 출력의 줄들.
"""

import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/ch51"


# ANCHOR: branch
def clip_or_double(x):
    """입력을 보고 갈 길을 정한다. 합이 양수이면 두 배, 아니면 0."""
    if x.sum() > 0:
        return x * 2
    return x * 0


def branch():
    positive, negative = torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0])
    print(f"branch python {clip_or_double(positive).tolist()} {clip_or_double(negative).tolist()}")

    # 1. 추적: 한 번 실행해 보고 지나간 연산을 받아 적는다.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = torch.jit.trace(clip_or_double, positive)
    print(f"branch trace {traced(positive).tolist()} {traced(negative).tolist()} 경고 {len(caught)}개")
    # 경고 가운데 둘을 적는다: 이 방법이 낡았다는 것과, 갈림길이 굳었다는 것.
    for first in sorted({str(warning.message).splitlines()[0] for warning in caught}):
        if "deprecated" in first or "Python boolean" in first:
            print(f"branch trace-warning {first[:230]}")

    # 2. 스크립트: 함수의 소스를 읽어 옮긴다. 갈림길이 남는다.
    scripted = torch.jit.script(clip_or_double)
    print(f"branch script {scripted(positive).tolist()} {scripted(negative).tolist()}")

    # 3. torch.export: 추적하되, 입력에 따라 갈리는 곳을 만나면 멈춘다.
    class Module(torch.nn.Module):
        def forward(self, x):
            return clip_or_double(x)

    try:
        torch.export.export(Module(), (positive,))
        print("branch export 통과")
    except Exception as error:  # noqa: BLE001
        lines = [line for line in str(error).splitlines() if line.strip()]
        print(f"branch export 거절 {type(error).__name__}: {lines[0][:200]}")

    # 갈림길을 텐서 연산으로 적으면 어느 길로도 내보낼 수 있다.
    class Rewritten(torch.nn.Module):
        def forward(self, x):
            return torch.where(x.sum() > 0, x * 2, x * 0)

    program = torch.export.export(Rewritten(), (positive,))
    module = program.module()
    print(f"branch export-where {module(positive).tolist()} {module(negative).tolist()}")
# ANCHOR_END: branch


class Mlp(torch.nn.Module):
    """7장의 그 신경망."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def mlp():
    model = Mlp()
    # 7장에서 학습해 둔 가중치를 그대로 읽는다. 이름이 하나라도 다르면 실패한다.
    model.load_state_dict(load_file(ROOT / "models/mnist_mlp/model.safetensors"), strict=True)
    return model.eval()


def images(count):
    data = (ROOT / "models/mnist/t10k-images-idx3-ubyte").read_bytes()
    pixels = torch.frombuffer(bytearray(data[16 : 16 + count * 784]), dtype=torch.uint8)
    return pixels.reshape(count, 784).float() / 255.0


# ANCHOR: export
def export():
    model, x = mlp(), images(4)
    # 묶음의 크기는 정하지 않고 내보낸다.
    batch = torch.export.Dim("batch")
    program = torch.export.export(model, (x,), dynamic_shapes={"x": {0: batch}})
    print("export graph")
    for node in program.graph.nodes:
        if node.op == "call_function":
            print(f"export node {node.name} = {node.target}({', '.join(str(argument) for argument in node.args)})")
        else:
            print(f"export node {node.op} {node.name}")
    print(f"export signature {[spec.kind.name for spec in program.graph_signature.input_specs]}")
    print(f"export shapes {program.range_constraints}")
    # 내보낸 것은 다른 크기의 묶음에도 쓸 수 있다.
    module = program.module()
    same = torch.equal(module(images(7)), model(images(7)))
    print(f"export batch-7 같은 답 {same}")
# ANCHOR_END: export


def onnx():
    OUT.mkdir(parents=True, exist_ok=True)
    model, x = mlp(), images(4)
    path = OUT / "mlp.onnx"
    torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"], dynamic_shapes={"x": {0: "batch"}}, dynamo=True, external_data=False)
    import onnx as onnx_format

    graph = onnx_format.load(path).graph
    print(f"onnx file {path.stat().st_size} bytes nodes {[node.op_type for node in graph.node]}")
    # badak이 견줄 입력과 정답.
    sample = images(16)
    with torch.inference_mode():
        expected = model(sample)
    save_file({"x": sample.contiguous(), "y": expected.contiguous()}, OUT / "mlp_reference.safetensors")
    import onnxruntime

    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = torch.from_numpy(session.run(None, {"x": sample.numpy()})[0])
    print(f"onnx onnxruntime 가장 큰 차이 {(got - expected).abs().max().item():.2e}")


# ANCHOR: executorch
def executorch():
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    OUT.mkdir(parents=True, exist_ok=True)
    model, x = mlp(), images(1)
    # 1. 내보낸다(그래프를 얻는다).
    program = torch.export.export(model, (x,))
    for label, partitioners in (("portable", []), ("xnnpack", [XnnpackPartitioner()])):
        # 2. 기기용 연산자로 낮추고, 가속기가 맡을 부분을 떼어 맡긴다.
        edge = to_edge_transform_and_lower(program, partitioner=partitioners)
        operators = [node.target.__name__ if hasattr(node.target, "__name__") else str(node.target) for node in edge.exported_program().graph.nodes if node.op == "call_function"]
        # 3. 실행기가 읽을 파일로 적는다. 메모리 계획이 이때 선다.
        binary = edge.to_executorch()
        path = OUT / f"mlp_{label}.pte"
        path.write_bytes(binary.buffer)
        # 4. 실행기로 돌린다.
        method = Runtime.get().load_program(path).load_method("forward")
        sample = images(16)
        with torch.inference_mode():
            expected = model(sample)
        worst = 0.0
        started = time.perf_counter()
        for row in range(16):
            got = method.execute([sample[row : row + 1]])[0]
            worst = max(worst, (got - expected[row : row + 1]).abs().max().item())
        each = (time.perf_counter() - started) / 16
        if label == "portable":
            # 견줄 것: PyTorch로 한 장씩.
            with torch.inference_mode():
                began = time.perf_counter()
                for _ in range(2000):
                    model(sample[:1])
                print(f"executorch pytorch 한 장에 {(time.perf_counter() - began) / 2000 * 1e6:.0f}us")
        print(f"executorch {label} file {path.stat().st_size} bytes 연산자 {operators} 가장 큰 차이 {worst:.2e} 한 장에 {each * 1e6:.0f}us")
# ANCHOR_END: executorch


def ort():
    """SmolLM2-135M을 ONNX Runtime으로. 50장의 브라우저 측정과 같은 반복문이다."""
    import numpy
    import onnxruntime

    path = ROOT / "models/ch50/hf/HuggingFaceTB/SmolLM2-135M-Instruct/onnx/model_fp16.onnx"
    options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    names = [f"past_key_values.{layer}.{kind}" for layer in range(30) for kind in ("key", "value")]
    outputs = ["logits"] + [name.replace("past_key_values", "present") for name in names]
    generator = numpy.random.default_rng(23)
    ids = generator.integers(0, 49152, size=(1, 256 + 32), dtype=numpy.int64)
    needs_positions = "position_ids" in [value.name for value in session.get_inputs()]

    def run(tokens, cache, seen):
        feeds = {"input_ids": tokens, "attention_mask": numpy.ones((1, seen + tokens.shape[1]), dtype=numpy.int64), **cache}
        if needs_positions:
            feeds["position_ids"] = numpy.arange(seen, seen + tokens.shape[1], dtype=numpy.int64)[None]
        result = session.run(outputs, feeds)
        return dict(zip(names, result[1:]))

    prompt, generation = [], []
    for repetition in range(5):
        time.sleep(3)
        empty = {name: numpy.zeros((1, 3, 0, 64), dtype=numpy.float16) for name in names}
        started = time.perf_counter()
        cache = run(ids[:, :256], empty, 0)
        took = time.perf_counter() - started
        started = time.perf_counter()
        for index in range(256, 288):
            cache = run(ids[:, index : index + 1], cache, index)
        each = time.perf_counter() - started
        if repetition:
            prompt.append(256 / took)
            generation.append(32 / each)
    print(f"ort speed threads default prompt {statistics.median(prompt):.0f} generation {statistics.median(generation):.1f} file-mb {path.stat().st_size / 1e6:.0f}")


if __name__ == "__main__":
    chosen = sys.argv[1:] or ["branch", "export", "onnx", "executorch", "ort"]
    for name in chosen:
        {"branch": branch, "export": export, "onnx": onnx, "executorch": executorch, "ort": ort}[name]()
