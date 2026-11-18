// 선형 층 y = x Wᵀ. 값을 넷씩 묶어 읽고, 합을 네 줄기로 나눠 쌓는다.
//
// 17장에서 CPU의 커널에 한 것과 같다. cols는 16의 배수여야 한다.

struct Shape {
    count: u32,
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
// 같은 메모리를 f32 넷의 묶음으로 읽는다.
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
    var sums = array<vec4<f32>, 4>(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    for (var k = 0u; k < quads; k += 4u) {
        sums[0] += x[x_start + k] * w[w_start + k];
        sums[1] += x[x_start + k + 1u] * w[w_start + k + 1u];
        sums[2] += x[x_start + k + 2u] * w[w_start + k + 2u];
        sums[3] += x[x_start + k + 3u] * w[w_start + k + 3u];
    }
    let total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    y[vector * shape.rows + row] = (total.x + total.y) + (total.z + total.w);
}
