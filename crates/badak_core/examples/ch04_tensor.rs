//! 4장: 텐서의 메타데이터가 뷰 연산에 따라 어떻게 바뀌는지 본다.
//!
//! 실행: `cargo run --release -p badak_core --example ch04_tensor`

use badak_core::{Result, Tensor};

fn describe(label: &str, tensor: &Tensor, original: &Tensor) -> Result<()> {
    println!(
        "{label:<22} 형상 {:<10} 스트라이드 {:<10} 시작 {:<3} 연속 {:<6} 공유 {:<6} {:?}",
        format!("{:?}", tensor.shape()),
        format!("{:?}", tensor.strides()),
        tensor.offset(),
        tensor.is_contiguous(),
        tensor.shares_storage_with(original),
        tensor.to_vec::<f32>()?
    );
    Ok(())
}

fn main() -> Result<()> {
    let t = Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4])?;
    describe("원본", &t, &t)?;
    describe("reshape(&[2, 6])", &t.reshape(&[2, 6])?, &t)?;
    describe("transpose(0, 1)", &t.transpose(0, 1)?, &t)?;
    describe("slice(0, 1..3)", &t.slice(0, 1..3)?, &t)?;
    describe("slice(1, 1..3)", &t.slice(1, 1..3)?, &t)?;
    describe(
        "transpose 후 contiguous",
        &t.transpose(0, 1)?.contiguous(),
        &t,
    )?;

    println!();
    let column = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0], &[3, 1])?;
    describe("열 벡터 (3, 1)", &column, &column)?;
    describe(
        "broadcast_to(&[3, 4])",
        &column.broadcast_to(&[3, 4])?,
        &column,
    )?;
    Ok(())
}
