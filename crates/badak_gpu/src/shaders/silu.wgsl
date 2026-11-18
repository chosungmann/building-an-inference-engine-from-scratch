// 원소마다 SiLU를 적용한다: y = x / (1 + exp(-x)).

// 버퍼 둘. 숫자는 묶는 자리(binding)다. x는 읽기만 하고 y에는 쓴다.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

// 그룹을 한 줄에 최대 32,768개까지 놓는다(한 축에 놓을 수 있는 그룹의 수에
// 한도가 있다). 한 줄이 맡는 원소의 수다.
const ROW: u32 = 32768u * 256u;

// 작업 그룹 하나가 스레드 256개다. 이 함수가 스레드마다 한 번씩 실행된다.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // id는 전체에서 이 스레드가 어디에 있는지다. 그것이 맡을 원소의 번호다.
    let i = id.y * ROW + id.x;
    // 원소의 수가 256의 배수가 아니면 마지막 그룹에는 할 일이 없는 스레드가
    // 있다.
    if (i >= arrayLength(&x)) {
        return;
    }
    y[i] = x[i] / (1.0 + exp(-x[i]));
}
