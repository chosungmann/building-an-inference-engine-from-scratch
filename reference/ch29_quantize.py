"""29장: MobileNetV2를 8비트 정수 모델로 바꾸고 LiteRT의 정답을 남긴다.

실행:
    sh scripts/download_ch27.sh
    pip install -r reference/requirements_ch29.txt
    python reference/ch29_quantize.py

출력 (models/vision_tflite/):
    mobilenetv2_int8.tflite     가중치와 중간 결과가 모두 int8인 모델
    reference_int8.safetensors  시험 사진마다의 정수 입력과 LiteRT의 정수 출력

27장의 mobilenetv2.tflite(f32)를 LiteRT의 양자화 도구로 바꾼다. 중간 결과의
범위를 정하려면 모델에 사진을 몇 장 넣어 봐야 한다(보정, calibration).
시험 사진 다섯 장에서 조각을 여러 개 잘라 쓴다.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file, save_file

from ai_edge_litert.interpreter import Interpreter, OpResolverType
from ai_edge_quantizer import quantizer, recipe

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "models/images"
OUT = ROOT / "models/vision_tflite"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE, CROP = 256, 224
SEED = 0
# 사진 한 장에서 잘라 낼 보정용 조각의 수.
CROPS = 8


def calibration_inputs():
    """보정에 쓸 입력들. 사진마다 무작위 위치에서 조각을 잘라 낸다."""
    rng = np.random.default_rng(SEED)
    inputs = []
    for path in sorted(IMAGES.glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = RESIZE / min(width, height)
        size = (max(RESIZE, round(width * scale)), max(RESIZE, round(height * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        # 가운데 조각(27장의 입력과 같은 것)을 먼저 넣는다. 시험할 때 넣을
        # 입력이 보정에 들어 있지 않으면 범위가 모자랄 수 있다.
        centre = ((size[0] - CROP) // 2, (size[1] - CROP) // 2)
        corners = [centre] + [
            (int(rng.integers(0, size[0] - CROP + 1)), int(rng.integers(0, size[1] - CROP + 1)))
            for _ in range(CROPS)
        ]
        for left, top in corners:
            pixels = np.asarray(image.crop((left, top, left + CROP, top + CROP)), dtype=np.float32) / 255.0
            inputs.append(((pixels - MEAN) / STD).transpose(2, 0, 1)[None])
    return inputs


def run(path, inputs, resolver):
    interpreter = Interpreter(model_path=str(path), experimental_op_resolver_type=resolver)
    interpreter.allocate_tensors()
    source = interpreter.get_input_details()[0]["index"]
    target = interpreter.get_output_details()[0]["index"]
    outputs = []
    for x in inputs:
        interpreter.set_tensor(source, x)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(target)[0].copy())
    return np.stack(outputs)


def main():
    source = OUT / "mobilenetv2.tflite"
    target = OUT / "mobilenetv2_int8.tflite"
    work = quantizer.Quantizer(str(source))
    # 가중치도 중간 결과도 8비트 정수. 합성곱의 가중치는 출력 채널마다 스케일을
    # 따로 둔다.
    work.load_quantization_recipe(recipe.static_wi8_ai8())
    samples = [{"args_0": x} for x in calibration_inputs()]
    result = work.quantize(work.calibrate({"serving_default": samples}))
    result.export_model(str(target), overwrite=True)
    print(f"{source.stat().st_size:,}바이트 → {target.stat().st_size:,}바이트, 보정 입력 {len(samples)}개")

    interpreter = Interpreter(model_path=str(target))
    interpreter.allocate_tensors()
    in_scale, in_zero = interpreter.get_input_details()[0]["quantization"]
    out_scale, out_zero = interpreter.get_output_details()[0]["quantization"]
    print(f"입력의 스케일 {in_scale:.6f}, 영점 {in_zero}. 출력의 스케일 {out_scale:.6f}, 영점 {out_zero}")

    float_reference = load_file(OUT / "reference.safetensors")
    labels = (OUT / "labels.txt").read_text(encoding="utf-8").splitlines()
    names = sorted(path.stem for path in IMAGES.glob("*.jpg"))
    # 27장의 입력(f32)을 이 모델의 정수 입력으로 바꾼다.
    inputs = [
        np.clip(np.round(float_reference[f"{name}.input"] / in_scale) + in_zero, -128, 127).astype(np.int8)
        for name in names
    ]
    # LiteRT의 기본 경로(XNNPACK)와, 넘겨주지 않고 LiteRT 자신의 커널로 계산한 것.
    delegated = run(target, inputs, OpResolverType.AUTO)
    builtin = run(target, inputs, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    reference = {}
    for index, name in enumerate(names):
        # 부호 있는 8비트를 바이트 그대로(u8로) 적는다. 책의 safetensors
        # 읽기 코드가 아는 자료형이다. 읽는 쪽이 부호를 붙여 해석한다.
        reference[f"{name}.input"] = inputs[index].view(np.uint8)
        reference[f"{name}.xnnpack"] = delegated[index].view(np.uint8)
        reference[f"{name}.builtin"] = builtin[index].view(np.uint8)
        best = int(builtin[index].argmax())
        before = int(float_reference[f"mobilenetv2.{name}.litert"].argmax())
        gap = int(np.abs(delegated[index].astype(np.int32) - builtin[index].astype(np.int32)).max())
        print(f"  {name:<18} {labels[best]:<28} f32 모델의 답: {labels[before]:<28} 두 경로의 차이 {gap}")
    # 첫 사진에 대해서는 LiteRT 자신의 커널이 계산한 중간 결과를 모두 남긴다.
    # 층마다 비트 단위로 견주는 데 쓴다. 이름의 번호는 파일의 텐서 번호다.
    interpreter = Interpreter(
        model_path=str(target),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inputs[0])
    interpreter.invoke()
    for detail in interpreter.get_tensor_details():
        if detail["dtype"] != np.int8 or len(detail["shape"]) == 0:
            continue
        try:
            value = interpreter.get_tensor(detail["index"])
        except ValueError:
            continue
        reference[f"{names[0]}.tensor.{detail['index']}"] = value.view(np.uint8)
    save_file(reference, OUT / "reference_int8.safetensors")


if __name__ == "__main__":
    main()
