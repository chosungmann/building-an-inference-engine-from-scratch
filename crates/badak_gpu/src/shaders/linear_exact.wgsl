// 선형 층 y = x Wᵀ. 17장의 CPU 커널과 같은 순서로 계산한다.
//
// 합 여덟 개를 두고(vec4 둘), k번째 값의 곱을 k % 8번째 합에 더한다. 곱하고
// 더하기를 fma로 한다. 끝에 17.3절의 순서대로 모은다. GPU가 fma를 정말 한
// 번의 연산으로 하고 덧셈을 올바르게 반올림한다면 CPU와 같은 비트가 나온다.
// cols는 8의 배수여야 한다.

struct Shape {
    count: u32,
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let vector = id.y;
    if (row >= shape.rows || vector >= shape.count) {
        return;
    }
    let quads = shape.cols / 4u;
    let x_start = vector * quads;
    let w_start = row * quads;
    // low는 0~3번째 합, high는 4~7번째 합이다.
    var low = vec4<f32>(0.0);
    var high = vec4<f32>(0.0);
    for (var k = 0u; k < quads; k += 2u) {
        low = fma(x[x_start + k], w[w_start + k], low);
        high = fma(x[x_start + k + 1u], w[w_start + k + 1u], high);
    }
    // 17.3절의 reduce: 넷씩 떨어진 것끼리 더하고, (0 + 2) + (1 + 3).
    let four = low + high;
    y[vector * shape.rows + row] = (four.x + four.z) + (four.y + four.w);
}
