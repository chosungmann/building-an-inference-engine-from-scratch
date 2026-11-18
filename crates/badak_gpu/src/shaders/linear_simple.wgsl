// 선형 층 y = x Wᵀ. 스레드 하나가 결과의 값 하나를 맡는다.
//
// x는 (count, cols), w는 (rows, cols), y는 (count, rows)이고 모두 행 우선으로
// 놓여 있다.

struct Shape {
    count: u32,
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // 가로로는 가중치의 행, 세로로는 입력 벡터.
    let row = id.x;
    let vector = id.y;
    if (row >= shape.rows || vector >= shape.count) {
        return;
    }
    // 2장의 내적 그대로다. 하나씩 곱해서 차례로 더한다.
    var sum = 0.0;
    for (var k = 0u; k < shape.cols; k++) {
        sum += x[vector * shape.cols + k] * w[row * shape.cols + k];
    }
    y[vector * shape.rows + row] = sum;
}
