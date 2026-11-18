// MLP의 가운데: gate = silu(gate) × up. 결과를 gate의 자리에 덮어쓴다.
//
// 선형 층 둘의 결과를 CPU로 가져가지 않고 GPU에서 곧바로 다음 층의 입력으로
// 만든다.

struct Shape {
    len: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;

// silu.wgsl과 같은 방법으로 그룹을 여러 줄에 놓는다.
const ROW: u32 = 32768u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * ROW + id.x;
    if (i >= shape.len) {
        return;
    }
    let value = gate[i];
    gate[i] = value / (1.0 + exp(-value)) * up[i];
}
