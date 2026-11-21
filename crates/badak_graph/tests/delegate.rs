//! 진짜 모델을 나눠서 돌려도 답이 같은지 본다 (45장).

use badak_formats::safetensors::SafeTensors;
use badak_formats::tflite::Model;
use badak_graph::delegate::{Hybrid, Options, Reference, partition};
use badak_graph::{Graph, Interpreter, Node, Registry, coreml, models_dir};

fn load(name: &str) -> (Graph, Vec<f32>) {
    let directory = models_dir().join("vision_tflite");
    let model = Model::open(directory.join(format!("{name}.tflite"))).unwrap();
    let input = SafeTensors::open(directory.join("reference.safetensors"))
        .unwrap()
        .tensor("cup.input")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    (Graph::from_tflite(&model).unwrap(), input)
}

#[test]
fn a_partitioned_mobilenet_gives_the_same_bits() {
    let (graph, input) = load("mobilenetv2");
    let registry = Registry::builtin();
    let mut plain = Interpreter::new(&graph, &registry).unwrap();
    plain.input_mut(0).copy_from_slice(&input);
    plain.run().unwrap();

    // (못 돌린다고 할 연산, 나누는 방법, 기다리는 조각의 수).
    let all = Options::default();
    let cases: [(&[&str], Options, usize); 5] = [
        (&[], all, 1),
        (&["ADD"], all, 21),
        (&["DEPTHWISE_CONV_2D"], all, 35),
        (&["ADD", "DEPTHWISE_CONV_2D"], all, 55),
        (
            &["ADD"],
            Options {
                max_delegated: 1,
                ..all
            },
            2,
        ),
    ];
    for (refused, options, count) in cases {
        let delegate = Reference::new(|_: &Graph, node: &Node| !refused.contains(&node.op.name()));
        let mut hybrid = Hybrid::new(&graph, &registry, &delegate, options).unwrap();
        assert_eq!(hybrid.segments().len(), count, "{refused:?}");
        // 모든 노드가 꼭 한 번씩 들어 있다.
        let mut nodes: Vec<usize> = hybrid
            .segments()
            .iter()
            .flat_map(|segment| segment.nodes.clone())
            .collect();
        nodes.sort_unstable();
        assert_eq!(nodes, (0..graph.nodes.len()).collect::<Vec<_>>());
        hybrid.input_mut(0).copy_from_slice(&input);
        hybrid.run().unwrap();
        assert_eq!(hybrid.output(0), plain.output(0), "{refused:?}");
    }
}

#[test]
fn both_models_lower_to_core_ml_whole() {
    // 층의 수: 노드에서 축 바꾸기 하나를 빼고, 활성화마다 층 하나를 더한다.
    for (name, layers) in [("mobilenetv2", 104), ("resnet50", 126)] {
        let (graph, _) = load(name);
        let supported: Vec<bool> = graph
            .nodes
            .iter()
            .map(|node| coreml::supports(&graph, node))
            .collect();
        assert!(supported.iter().all(|&yes| yes), "{name}");
        let segments = partition(&graph, &supported, Options::default()).unwrap();
        assert_eq!(segments.len(), 1);
        let lowered = coreml::lower(&graph, &segments[0]).unwrap();
        assert_eq!(lowered.layers, layers, "{name}");
        // 맨 앞의 축 바꾸기는 층이 되지 않고, 입력은 그대로 들어간다.
        assert_eq!(lowered.inputs[0].shape, [1, 3, 224, 224]);
        assert!(!lowered.inputs[0].channels_last);
        assert_eq!(lowered.outputs[0].shape, [1, 1000, 1, 1]);
        // 가중치가 거의 전부다.
        let file = std::fs::metadata(models_dir().join(format!("vision_tflite/{name}.tflite")))
            .unwrap()
            .len();
        let ratio = lowered.model.len() as f64 / file as f64;
        assert!((0.95..1.05).contains(&ratio), "{name}: {ratio}");
    }
}
