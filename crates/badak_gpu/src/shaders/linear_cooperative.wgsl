// 선형 층 y = x Wᵀ. 작업 그룹 하나(스레드 32개)가 힘을 합쳐 8 × 8 행렬의
// 곱을 한 번에 계산한다. 그룹 하나가 결과의 값 1,024개(벡터 32 × 행 32)를
// 맡는다.
//
// WebGPU의 표준에는 없는 기능이다. wgpu가 실험 기능으로 내놓은 것으로,
// Apple의 GPU와 일부 Vulkan 장치에서만 쓸 수 있다. count와 rows는 32의 배수,
// cols는 8의 배수여야 한다.
enable wgpu_cooperative_matrix;

struct Shape {
    count: u32,
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// 그룹 하나가 SIMD 그룹 하나와 같아야 한다. 이 책의 GPU에서는 32다.
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x * 32u;
    let vector = group.y * 32u;
    // c<i><j>는 벡터 8개(i번째 묶음)와 행 8개(j번째 묶음)의 내적 64개다.
    var c00: coop_mat8x8<f32, C>;
    var c01: coop_mat8x8<f32, C>;
    var c02: coop_mat8x8<f32, C>;
    var c03: coop_mat8x8<f32, C>;
    var c10: coop_mat8x8<f32, C>;
    var c11: coop_mat8x8<f32, C>;
    var c12: coop_mat8x8<f32, C>;
    var c13: coop_mat8x8<f32, C>;
    var c20: coop_mat8x8<f32, C>;
    var c21: coop_mat8x8<f32, C>;
    var c22: coop_mat8x8<f32, C>;
    var c23: coop_mat8x8<f32, C>;
    var c30: coop_mat8x8<f32, C>;
    var c31: coop_mat8x8<f32, C>;
    var c32: coop_mat8x8<f32, C>;
    var c33: coop_mat8x8<f32, C>;
    for (var k = 0u; k < shape.cols; k += 8u) {
        // a<i>: 벡터 8개의 값 8개씩. 행이 벡터이고 열이 k다.
        let a0 = coopLoadT<coop_mat8x8<f32, A>>(&x[(vector + 0u) * shape.cols + k], shape.cols);
        let a1 = coopLoadT<coop_mat8x8<f32, A>>(&x[(vector + 8u) * shape.cols + k], shape.cols);
        let a2 = coopLoadT<coop_mat8x8<f32, A>>(&x[(vector + 16u) * shape.cols + k], shape.cols);
        let a3 = coopLoadT<coop_mat8x8<f32, A>>(&x[(vector + 24u) * shape.cols + k], shape.cols);
        // b<j>: 가중치의 행 8개의 값 8개씩. 열이 가중치의 행이고 행이 k다.
        let b0 = coopLoad<coop_mat8x8<f32, B>>(&w[(row + 0u) * shape.cols + k], shape.cols);
        let b1 = coopLoad<coop_mat8x8<f32, B>>(&w[(row + 8u) * shape.cols + k], shape.cols);
        let b2 = coopLoad<coop_mat8x8<f32, B>>(&w[(row + 16u) * shape.cols + k], shape.cols);
        let b3 = coopLoad<coop_mat8x8<f32, B>>(&w[(row + 24u) * shape.cols + k], shape.cols);
        // c += a × b. 곱셈 512번을 한 번의 연산으로 한다.
        c00 = coopMultiplyAdd(a0, b0, c00);
        c01 = coopMultiplyAdd(a0, b1, c01);
        c02 = coopMultiplyAdd(a0, b2, c02);
        c03 = coopMultiplyAdd(a0, b3, c03);
        c10 = coopMultiplyAdd(a1, b0, c10);
        c11 = coopMultiplyAdd(a1, b1, c11);
        c12 = coopMultiplyAdd(a1, b2, c12);
        c13 = coopMultiplyAdd(a1, b3, c13);
        c20 = coopMultiplyAdd(a2, b0, c20);
        c21 = coopMultiplyAdd(a2, b1, c21);
        c22 = coopMultiplyAdd(a2, b2, c22);
        c23 = coopMultiplyAdd(a2, b3, c23);
        c30 = coopMultiplyAdd(a3, b0, c30);
        c31 = coopMultiplyAdd(a3, b1, c31);
        c32 = coopMultiplyAdd(a3, b2, c32);
        c33 = coopMultiplyAdd(a3, b3, c33);
    }
    coopStoreT(c00, &y[(vector + 0u) * shape.rows + row + 0u], shape.rows);
    coopStoreT(c01, &y[(vector + 0u) * shape.rows + row + 8u], shape.rows);
    coopStoreT(c02, &y[(vector + 0u) * shape.rows + row + 16u], shape.rows);
    coopStoreT(c03, &y[(vector + 0u) * shape.rows + row + 24u], shape.rows);
    coopStoreT(c10, &y[(vector + 8u) * shape.rows + row + 0u], shape.rows);
    coopStoreT(c11, &y[(vector + 8u) * shape.rows + row + 8u], shape.rows);
    coopStoreT(c12, &y[(vector + 8u) * shape.rows + row + 16u], shape.rows);
    coopStoreT(c13, &y[(vector + 8u) * shape.rows + row + 24u], shape.rows);
    coopStoreT(c20, &y[(vector + 16u) * shape.rows + row + 0u], shape.rows);
    coopStoreT(c21, &y[(vector + 16u) * shape.rows + row + 8u], shape.rows);
    coopStoreT(c22, &y[(vector + 16u) * shape.rows + row + 16u], shape.rows);
    coopStoreT(c23, &y[(vector + 16u) * shape.rows + row + 24u], shape.rows);
    coopStoreT(c30, &y[(vector + 24u) * shape.rows + row + 0u], shape.rows);
    coopStoreT(c31, &y[(vector + 24u) * shape.rows + row + 8u], shape.rows);
    coopStoreT(c32, &y[(vector + 24u) * shape.rows + row + 16u], shape.rows);
    coopStoreT(c33, &y[(vector + 24u) * shape.rows + row + 24u], shape.rows);
}
