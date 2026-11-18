//! 만들어 둔 합성곱 모델로 해석기 전체를 검증한다 (27장).
//!
//! 파일이 없으면 건너뛴다. `sh scripts/download_ch27.sh`로 받는다.

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::registry::Factory;
use badak_graph::{Graph, Interpreter, Registry, Strategy, conv, memory, models_dir};

const IMAGES: [&str; 5] = ["cup", "golden_retriever", "guitar", "pelican", "tabby_cat"];

#[test]
fn matches_litert_and_pytorch() {
    let directory = models_dir().join("vision_tflite");
    if !directory.join("resnet50.tflite").exists() {
        eprintln!("건너뜀: {}에 모델이 없습니다", directory.display());
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    for (name, steps) in [("resnet50", 78), ("mobilenetv2", 70)] {
        let model = Model::open(directory.join(format!("{name}.tflite"))).unwrap();
        let graph = Graph::from_tflite(&model).unwrap();
        let mut interpreter = Interpreter::new(&graph, &Registry::builtin()).unwrap();
        assert_eq!(interpreter.step_count(), steps);
        assert_eq!(interpreter.input_shape(0), [1, 3, 224, 224]);
        assert_eq!(interpreter.output_shape(0), [1, 1000]);
        // 계획한 자리가 하한과 같고, 따로 둘 때의 3분의 1보다 작다.
        let lifetimes = interpreter.lifetimes();
        assert_eq!(interpreter.arena_len(), memory::lower_bound(&lifetimes));
        assert!(interpreter.arena_len() * 3 < memory::plan(&lifetimes, Strategy::Separate).total);

        for image in IMAGES {
            let input = reference.tensor(&format!("{image}.input")).unwrap();
            // 앞의 실행이 남긴 값에 기대면 여기서 드러난다.
            interpreter.scribble(f32::NAN);
            interpreter
                .input_mut(0)
                .copy_from_slice(&input.to_vec::<f32>().unwrap());
            interpreter.run().unwrap();
            let scores = interpreter.output(0);
            for other in ["litert", "pytorch"] {
                let expected = reference
                    .tensor(&format!("{name}.{image}.{other}"))
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let best = |values: &[f32]| {
                    (0..values.len())
                        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
                        .unwrap()
                };
                assert_eq!(best(scores), best(&expected), "{name} {image} {other}");
                for (ours, theirs) in scores.iter().zip(&expected) {
                    assert!(
                        (ours - theirs).abs() < 2e-4,
                        "{name} {image}: {ours} != {theirs}"
                    );
                }
            }
        }
    }
}

#[test]
fn convolution_methods_agree_on_a_real_model() {
    let directory = models_dir().join("vision_tflite");
    if !directory.join("mobilenetv2.tflite").exists() {
        return;
    }
    let reference = SafeTensors::open(directory.join("reference.safetensors")).unwrap();
    let input = reference
        .tensor("pelican.input")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let model = Model::open(directory.join("mobilenetv2.tflite")).unwrap();
    let graph = Graph::from_tflite(&model).unwrap();
    let run = |factory: Factory| {
        let mut registry = Registry::builtin();
        registry.register("CONV_2D", factory);
        let mut interpreter = Interpreter::new(&graph, &registry).unwrap();
        interpreter.input_mut(0).copy_from_slice(&input);
        interpreter.run().unwrap();
        interpreter.output(0).to_vec()
    };
    let expected = run(conv::conv_2d_direct);
    for factory in [conv::conv_2d_im2col as Factory, conv::conv_2d] {
        for (ours, theirs) in run(factory).iter().zip(&expected) {
            assert!((ours - theirs).abs() < 2e-4);
        }
    }
}
