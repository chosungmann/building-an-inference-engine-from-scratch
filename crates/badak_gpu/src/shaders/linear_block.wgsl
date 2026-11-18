// 선형 층 y = x Wᵀ. 스레드 하나가 결과의 값 열여섯(벡터 넷 × 행 넷)을 맡는다.
//
// 입력 벡터 넷과 가중치의 행 넷을 한 번씩만 읽어 열여섯 내적에 함께 쓴다.
// 읽는 양이 값 하나당 4분의 1로 준다. count, rows, cols가 모두 4의 배수여야
// 한다.
//
// 반복문과 배열을 쓰지 않고 열여섯 줄을 풀어 적었다. 합을 배열에 두고
// 반복문으로 돌면 절반도 안 되는 속도가 나온다(23.5절).

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
    let row = id.x * 4u;
    let vector = id.y * 4u;
    if (row >= shape.rows || vector >= shape.count) {
        return;
    }
    let quads = shape.cols / 4u;
    // s<v><r>은 벡터 v와 행 r의 내적이다.
    var s00 = 0.0; var s01 = 0.0; var s02 = 0.0; var s03 = 0.0;
    var s10 = 0.0; var s11 = 0.0; var s12 = 0.0; var s13 = 0.0;
    var s20 = 0.0; var s21 = 0.0; var s22 = 0.0; var s23 = 0.0;
    var s30 = 0.0; var s31 = 0.0; var s32 = 0.0; var s33 = 0.0;
    for (var k = 0u; k < quads; k++) {
        let w0 = w[(row + 0u) * quads + k];
        let w1 = w[(row + 1u) * quads + k];
        let w2 = w[(row + 2u) * quads + k];
        let w3 = w[(row + 3u) * quads + k];
        let x0 = x[(vector + 0u) * quads + k];
        s00 += dot(x0, w0);
        s01 += dot(x0, w1);
        s02 += dot(x0, w2);
        s03 += dot(x0, w3);
        let x1 = x[(vector + 1u) * quads + k];
        s10 += dot(x1, w0);
        s11 += dot(x1, w1);
        s12 += dot(x1, w2);
        s13 += dot(x1, w3);
        let x2 = x[(vector + 2u) * quads + k];
        s20 += dot(x2, w0);
        s21 += dot(x2, w1);
        s22 += dot(x2, w2);
        s23 += dot(x2, w3);
        let x3 = x[(vector + 3u) * quads + k];
        s30 += dot(x3, w0);
        s31 += dot(x3, w1);
        s32 += dot(x3, w2);
        s33 += dot(x3, w3);
    }
    let y0 = (vector + 0u) * shape.rows + row;
    y[y0 + 0u] = s00;
    y[y0 + 1u] = s01;
    y[y0 + 2u] = s02;
    y[y0 + 3u] = s03;
    let y1 = (vector + 1u) * shape.rows + row;
    y[y1 + 0u] = s10;
    y[y1 + 1u] = s11;
    y[y1 + 2u] = s12;
    y[y1 + 3u] = s13;
    let y2 = (vector + 2u) * shape.rows + row;
    y[y2 + 0u] = s20;
    y[y2 + 1u] = s21;
    y[y2 + 2u] = s22;
    y[y2 + 3u] = s23;
    let y3 = (vector + 3u) * shape.rows + row;
    y[y3 + 0u] = s30;
    y[y3 + 1u] = s31;
    y[y3 + 2u] = s32;
    y[y3 + 3u] = s33;
}
