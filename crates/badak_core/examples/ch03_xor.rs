//! 3장: 손으로 가중치를 정한 MLP로 XOR을 계산한다.
//!
//! 실행: `cargo run --release -p badak_core --example ch03_xor`

use badak_core::nn::{Activation, Linear, Mlp, xor_mlp};

fn main() {
    let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let hidden = Linear::new(vec![1.0, 1.0, 1.0, 1.0], vec![0.0, -1.0], 2, 2);

    println!("== 활성화 함수가 있을 때 ==");
    println!(
        "{:>4} {:>4} | {:>6} {:>6} | {:>5}",
        "x1", "x2", "h1", "h2", "출력"
    );
    let model = xor_mlp();
    for x in inputs {
        let mut h = hidden.forward(&x);
        Activation::Relu.apply_in_place(&mut h);
        let y = model.forward(&x);
        println!(
            "{:>4} {:>4} | {:>6} {:>6} | {:>5}",
            x[0], x[1], h[0], h[1], y[0]
        );
    }

    println!();
    println!("== 같은 가중치에서 활성화 함수를 빼면 ==");
    println!("{:>4} {:>4} | {:>5}", "x1", "x2", "출력");
    let output = Linear::new(vec![1.0, -2.0], vec![0.0], 2, 1);
    let linear_only = Mlp::new(vec![hidden.clone(), output], Activation::Identity);
    for x in inputs {
        println!(
            "{:>4} {:>4} | {:>5}",
            x[0],
            x[1],
            linear_only.forward(&x)[0]
        );
    }

    println!();
    println!("파라미터 수: {}", model.parameter_count());
}
