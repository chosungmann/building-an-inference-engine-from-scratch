//! 객체 검출 모델의 출력을 상자로 바꾼다 (32장).
//!
//! 분류 모델의 출력은 점수 1,000개였고 가장 큰 것을 고르면 끝이었다. 검출
//! 모델의 출력은 "칸마다의 예측" 수천 개다. 그것을 사진 위의 상자로
//! 바꾸고(해독), 같은 물체를 가리키는 상자들 가운데 하나만 남기는 일
//! (비최대 억제)이 모델 밖에 남아 있다.

// ANCHOR: detection
/// 찾아낸 물체 하나.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// 종류의 번호.
    pub class: usize,
    /// 점수. 0과 1 사이다.
    pub score: f32,
    /// 상자. `[왼쪽, 위, 오른쪽, 아래]`이고 단위는 픽셀이다.
    pub bounds: [f32; 4],
}
// ANCHOR_END: detection

// ANCHOR: decode
/// YOLOX의 출력을 상자들로 바꾼다.
///
/// 출력은 칸마다 `4 + 1 + classes`개의 값이다. 상자의 중심과 크기 넷, 물체가
/// 있을 점수 하나, 종류마다의 점수다. 칸은 사진을 `stride` 픽셀씩 나눈
/// 격자의 한 자리이고, 격자가 여러 개다(`strides`마다 하나. 촘촘한 격자가
/// 작은 물체를, 성긴 격자가 큰 물체를 맡는다). 출력에는 격자들이 차례로,
/// 격자 안에서는 위에서 아래로, 왼쪽에서 오른쪽으로 놓여 있다.
///
/// 모델이 내는 중심은 "자기 칸의 왼쪽 위에서 얼마나 떨어졌는가"를 칸의
/// 크기로 잰 것이고, 크기는 로그다. 칸의 위치를 더하고 `stride`를 곱해
/// 픽셀로 바꾼다.
///
/// 점수(물체가 있을 점수 × 종류의 점수)가 `threshold`를 넘는 것만
/// 돌려준다. 한 칸이 여러 종류로 나올 수 있다.
pub fn decode_yolox(
    output: &[f32],
    side: usize,
    strides: &[usize],
    classes: usize,
    threshold: f32,
) -> Vec<Detection> {
    let width = 5 + classes;
    let mut detections = Vec::new();
    let mut cells = output.chunks_exact(width);
    for &stride in strides {
        let grid = side / stride;
        for row in 0..grid {
            for column in 0..grid {
                let Some(cell) = cells.next() else {
                    return detections;
                };
                let scale = stride as f32;
                let center = [
                    (cell[0] + column as f32) * scale,
                    (cell[1] + row as f32) * scale,
                ];
                let size = [cell[2].exp() * scale, cell[3].exp() * scale];
                let bounds = [
                    center[0] - size[0] / 2.0,
                    center[1] - size[1] / 2.0,
                    center[0] + size[0] / 2.0,
                    center[1] + size[1] / 2.0,
                ];
                for (class, &score) in cell[5..].iter().enumerate() {
                    let score = cell[4] * score;
                    if score > threshold {
                        detections.push(Detection {
                            class,
                            score,
                            bounds,
                        });
                    }
                }
            }
        }
    }
    detections
}
// ANCHOR_END: decode

// ANCHOR: iou
/// 두 상자가 겹친 넓이를 둘을 합친 넓이로 나눈 값(intersection over
/// union). 똑같으면 1, 겹치지 않으면 0이다.
pub fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let area =
        |bounds: &[f32; 4]| (bounds[2] - bounds[0]).max(0.0) * (bounds[3] - bounds[1]).max(0.0);
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let shared = width * height;
    let total = area(a) + area(b) - shared;
    if total > 0.0 { shared / total } else { 0.0 }
}
// ANCHOR_END: iou

// ANCHOR: nms
/// 비최대 억제(non-maximum suppression). 같은 물체를 가리키는 상자들 가운데
/// 점수가 가장 높은 것만 남긴다.
///
/// 점수가 높은 것부터 본다. 이미 남기기로 한 같은 종류의 상자와
/// `threshold`보다 많이 겹치면 버리고, 아니면 남긴다. 결과는 점수가 높은
/// 순서다. 점수가 같으면 먼저 들어온 것이 앞선다.
pub fn non_max_suppression(mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in detections {
        let duplicate = kept.iter().any(|winner| {
            winner.class == candidate.class
                && overlap(&winner.bounds, &candidate.bounds) > threshold
        });
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}
// ANCHOR_END: nms

/// COCO 데이터의 물체 80가지의 이름. YOLOX가 쓰는 순서다.
pub const COCO: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(class: usize, score: f32, bounds: [f32; 4]) -> Detection {
        Detection {
            class,
            score,
            bounds,
        }
    }

    #[test]
    fn overlap_is_a_ratio_of_areas() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(overlap(&a, &a), 1.0);
        assert_eq!(overlap(&a, &[10.0, 0.0, 20.0, 10.0]), 0.0);
        assert_eq!(overlap(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0);
        // 절반이 겹친다: 50 / (100 + 100 − 50).
        assert!((overlap(&a, &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
        // 안에 들어 있다: 25 / 100.
        assert_eq!(overlap(&a, &[0.0, 0.0, 5.0, 5.0]), 0.25);
        // 넓이가 없는 상자.
        assert_eq!(overlap(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn suppression_keeps_the_best_of_each_cluster() {
        let boxes = vec![
            detection(0, 0.6, [1.0, 1.0, 11.0, 11.0]),
            detection(0, 0.9, [0.0, 0.0, 10.0, 10.0]),
            // 같은 자리지만 다른 종류다. 남는다.
            detection(1, 0.5, [0.0, 0.0, 10.0, 10.0]),
            // 멀리 떨어진 같은 종류. 남는다.
            detection(0, 0.4, [50.0, 50.0, 60.0, 60.0]),
            // 조금만 겹친다. 남는다.
            detection(0, 0.3, [8.0, 8.0, 18.0, 18.0]),
        ];
        let kept = non_max_suppression(boxes, 0.45);
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, [0.9, 0.5, 0.4, 0.3]);
        assert!(non_max_suppression(Vec::new(), 0.5).is_empty());
    }

    #[test]
    fn decoding_places_boxes_on_the_image() {
        // 8 × 8 사진, 스트라이드 4와 8. 칸이 4 + 1개이고 종류는 둘이다.
        let mut output = vec![0.0_f32; 5 * 7];
        // 첫 격자의 (줄 1, 칸 0): 중심이 칸의 한가운데, 크기는 칸의 두 배.
        output[2 * 7..3 * 7].copy_from_slice(&[
            0.5,
            0.5,
            2.0_f32.ln(),
            2.0_f32.ln(),
            0.8,
            0.1,
            0.9,
        ]);
        // 둘째 격자의 하나뿐인 칸: 두 종류가 모두 문턱을 넘는다.
        output[4 * 7..].copy_from_slice(&[0.25, 0.75, 0.0, 0.0, 1.0, 0.6, 0.7]);
        let found = decode_yolox(&output, 8, &[4, 8], 2, 0.5);
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].class, 1);
        assert!((found[0].score - 0.72).abs() < 1e-6);
        // 중심 (2, 6), 크기 8 × 8.
        assert_eq!(found[0].bounds, [-2.0, 2.0, 6.0, 10.0]);
        // 중심 (2, 6), 크기 8 × 8인 상자가 종류마다 하나씩.
        assert_eq!((found[1].class, found[2].class), (0, 1));
        assert_eq!(found[1].bounds, [-2.0, 2.0, 6.0, 10.0]);
        // 출력이 모자라도 죽지 않는다.
        assert_eq!(decode_yolox(&output[..10], 8, &[4, 8], 2, 0.5).len(), 0);
    }
}
