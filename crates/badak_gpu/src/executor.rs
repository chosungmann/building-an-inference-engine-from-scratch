//! 미래의 값(`Future`)이 준비될 때까지 기다리는 가장 작은 실행기.
//!
//! `wgpu`의 함수 몇 개는 `async`다. 브라우저에서는 GPU를 여는 일이 정말로
//! 나중에 끝나기 때문이다. 브라우저 밖에서는 부르는 즉시 끝나므로, 준비됐는지
//! 물어보기를 되풀이하는 것으로 충분하다. 외부 실행기를 쓰지 않는다.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

// ANCHOR: block_on
/// `future`가 값을 낼 때까지 이 스레드에서 기다린다.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    // 깨워 줄 필요가 없으므로 아무 일도 하지 않는 깨우개를 준다.
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}
// ANCHOR_END: block_on

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_the_value_of_a_ready_future() {
        assert_eq!(block_on(async { 6 * 7 }), 42);
    }

    #[test]
    fn keeps_polling_until_the_future_is_ready() {
        struct Later(u32);
        impl Future for Later {
            type Output = &'static str;
            fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                self.0 += 1;
                if self.0 < 3 {
                    Poll::Pending
                } else {
                    Poll::Ready("done")
                }
            }
        }
        assert_eq!(block_on(Later(0)), "done");
    }
}
