//! 연산의 이름에서 커널을 찾는 표.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::graph::Op;
use crate::kernels::{self, Kernel};
use crate::{conv, spatial};

// ANCHOR: registry
/// 마디의 연산을 받아 그것을 계산할 커널을 만드는 함수.
pub type Factory = fn(&Op) -> Result<Box<dyn Kernel>>;

/// 연산자 레지스트리. 연산의 이름마다 커널을 만드는 함수를 적어 둔 표다.
///
/// 해석기는 어떤 연산이 있는지 모른다. 그래프의 마디마다 이 표에서 이름을
/// 찾을 뿐이다. 새 연산을 더하는 일은 표에 한 줄을 더하는 일이고, 해석기는
/// 고치지 않는다.
#[derive(Clone, Default)]
pub struct Registry {
    factories: BTreeMap<String, Factory>,
}

impl Registry {
    /// 아무 연산도 모르는 레지스트리.
    pub fn empty() -> Self {
        Self::default()
    }

    /// 이 크레이트가 가진 커널을 모두 담은 레지스트리.
    pub fn builtin() -> Self {
        let mut registry = Self::empty();
        registry.register("FULLY_CONNECTED", kernels::fully_connected);
        registry.register("ADD", kernels::add);
        registry.register("MUL", kernels::mul);
        registry.register("RELU", kernels::relu);
        registry.register("SOFTMAX", kernels::softmax);
        // 27장: 합성곱 모델의 연산들.
        registry.register("CONV_2D", conv::conv_2d);
        registry.register("DEPTHWISE_CONV_2D", conv::depthwise_conv_2d);
        registry.register("MAX_POOL_2D", spatial::max_pool_2d);
        registry.register("AVERAGE_POOL_2D", spatial::average_pool_2d);
        registry.register("PAD", spatial::pad);
        registry.register("MEAN", spatial::mean);
        registry.register("TRANSPOSE", spatial::transpose);
        registry
    }

    /// 연산 하나를 등록한다. 같은 이름이 이미 있으면 바꾸고, 있던 것을
    /// 돌려준다.
    pub fn register(&mut self, name: &str, factory: Factory) -> Option<Factory> {
        self.factories.insert(name.to_string(), factory)
    }

    /// 마디의 연산을 계산할 커널을 만든다.
    pub fn resolve(&self, op: &Op) -> Result<Box<dyn Kernel>> {
        let factory = self.factories.get(op.name()).ok_or_else(|| {
            Error::Unsupported(format!(
                "연산 {} (등록된 연산: {})",
                op.name(),
                self.names().collect::<Vec<_>>().join(", ")
            ))
        })?;
        factory(op)
    }

    /// 등록된 연산의 이름들.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }
}
// ANCHOR_END: registry

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Activation;

    #[test]
    fn resolves_registered_operators_only() {
        let registry = Registry::builtin();
        assert!(registry.resolve(&Op::Relu).is_ok());
        let unknown = Op::Custom {
            name: "BUILTIN_3".into(),
        };
        let Err(Error::Unsupported(message)) = registry.resolve(&unknown).map(|_| ()) else {
            panic!("모르는 연산은 오류여야 한다");
        };
        assert!(message.contains("BUILTIN_3") && message.contains("SOFTMAX"));
        assert!(Registry::empty().resolve(&Op::Relu).is_err());
    }

    #[test]
    fn registering_again_replaces_the_kernel() {
        let mut registry = Registry::builtin();
        let add = Op::Add {
            activation: Activation::None,
        };
        // ADD를 곱셈으로 바꿔치기한다.
        let previous = registry.register("ADD", |_| {
            kernels::mul(&Op::Mul {
                activation: Activation::None,
            })
        });
        assert!(previous.is_some());
        let kernel = registry.resolve(&add).unwrap();
        let mut y = [0.0_f32];
        let shape: &[usize] = &[1];
        let view = |data| kernels::View { shape, data };
        kernel
            .run(
                &[view(&[3.0]), view(&[4.0])],
                &mut [kernels::ViewMut {
                    shape,
                    data: &mut y,
                }],
            )
            .unwrap();
        assert_eq!(y, [12.0]);
    }
}
