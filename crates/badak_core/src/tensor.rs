//! 텐서: 형상이 붙은 숫자 배열 (4장).
//!
//! [`Tensor`]는 두 부분으로 이루어진다. 숫자가 실제로 들어 있는 저장소와,
//! 그 저장소를 어떻게 읽을지 적어 둔 메타데이터(형상, 스트라이드,
//! 시작 위치)다. 전치, 자르기, 형상 바꾸기 같은 연산은 숫자를 옮기지 않고
//! 메타데이터만 바꾼 새 텐서를 돌려준다. 이런 텐서를 뷰라고 한다.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::error::{Error, Result};

// ANCHOR: dtype
/// 텐서 원소의 자료형.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32비트 부동소수점. 가중치와 중간 계산의 기본 자료형.
    F32,
    /// 32비트 부호 있는 정수. 토큰 번호와 위치에 쓴다.
    I32,
    /// 8비트 부호 없는 정수. 사진의 픽셀 값에 쓴다.
    U8,
}

impl DType {
    /// 원소 하나가 차지하는 바이트 수.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::U8 => 1,
        }
    }
}
// ANCHOR_END: dtype

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U8 => "u8",
        })
    }
}

/// 저장소와 원소 타입을 잇는 내부 장치. 크레이트 밖에서는 보이지 않는다.
mod storage {
    use std::ops::Range;
    use std::sync::Arc;

    use super::DType;

    // ANCHOR: storage
    /// 숫자가 실제로 들어 있는 곳. 자료형마다 벡터가 하나씩 있다.
    pub enum Storage {
        F32(Vec<f32>),
        I32(Vec<i32>),
        U8(Vec<u8>),
        // ANCHOR_END: storage
        // ANCHOR: storage_borrowed
        /// 다른 누군가가 가진 바이트 덩어리의 한 구간을 빌려 쓴다 (6장).
        ///
        /// 메모리에 비춘 모델 파일이 대표적이다. `owner`를 붙들고 있는
        /// 동안 그 메모리는 살아 있다.
        Borrowed {
            owner: ByteOwner,
            range: Range<usize>,
            dtype: DType,
        },
        // ANCHOR_END: storage_borrowed
    }

    /// 빌려 쓸 수 있는 바이트 덩어리의 주인.
    pub type ByteOwner = Arc<dyn AsRef<[u8]> + Send + Sync>;

    impl std::fmt::Debug for Storage {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::F32(data) => write!(f, "F32({}개)", data.len()),
                Self::I32(data) => write!(f, "I32({}개)", data.len()),
                Self::U8(data) => write!(f, "U8({}개)", data.len()),
                Self::Borrowed { range, dtype, .. } => {
                    write!(f, "Borrowed({dtype}, 바이트 {range:?})")
                }
            }
        }
    }

    // ANCHOR: cast_bytes
    /// 바이트 슬라이스를 `T`의 슬라이스로 다시 읽는다. 복사하지 않는다.
    ///
    /// 시작 주소가 `T`의 정렬에 맞지 않거나 길이가 `T` 크기의 배수가
    /// 아니면 `None`을 돌려준다.
    pub fn cast_bytes<T: Sealed>(bytes: &[u8]) -> Option<&[T]> {
        // SAFETY: `Sealed`는 f32, i32, u8에만 구현되어 있다. 이 타입들은
        // 어떤 비트 무늬든 유효한 값이므로, 정렬이 맞는 가운데 부분을
        // `T`로 읽어도 정의되지 않은 동작이 생기지 않는다. `align_to`가
        // 가운데 부분의 정렬과 길이를 보장한다.
        let (prefix, values, suffix) = unsafe { bytes.align_to::<T>() };
        (prefix.is_empty() && suffix.is_empty()).then_some(values)
    }
    // ANCHOR_END: cast_bytes

    /// [`Element`](super::Element)를 이 크레이트 안에서만 구현할 수 있게 막는다.
    pub trait Sealed: Sized {
        fn wrap(data: Vec<Self>) -> Storage;
        fn unwrap(storage: &Storage) -> Option<&[Self]>;
        /// 리틀 엔디언 바이트를 값으로 복사해 읽는다.
        fn from_le_bytes(bytes: &[u8]) -> Vec<Self>;
    }

    macro_rules! impl_sealed {
        ($type:ty, $variant:ident) => {
            impl Sealed for $type {
                fn wrap(data: Vec<Self>) -> Storage {
                    Storage::$variant(data)
                }

                fn unwrap(storage: &Storage) -> Option<&[Self]> {
                    match storage {
                        Storage::$variant(data) => Some(data),
                        Storage::Borrowed {
                            owner,
                            range,
                            dtype: DType::$variant,
                        } => {
                            // 빌린 구간이 여전히 유효하고 정렬이 맞는지 읽을
                            // 때마다 확인한다. 만들 때 이미 확인했으므로
                            // 정상적인 주인이라면 항상 성공한다.
                            let bytes = (**owner).as_ref().get(range.clone())?;
                            cast_bytes(bytes)
                        }
                        _ => None,
                    }
                }

                fn from_le_bytes(bytes: &[u8]) -> Vec<Self> {
                    bytes
                        .chunks_exact(size_of::<Self>())
                        .map(|chunk| Self::from_le_bytes(chunk.try_into().expect("길이가 맞다")))
                        .collect()
                }
            }
        };
    }

    impl_sealed!(f32, F32);
    impl_sealed!(i32, I32);
    impl_sealed!(u8, U8);
}

use storage::{ByteOwner, Storage};

// ANCHOR: element
/// 텐서의 원소가 될 수 있는 Rust 타입.
///
/// `f32`, `i32`, `u8`에만 구현되어 있다. 텐서의 자료형은 실행 중에
/// 정해지지만, 이 트레이트 덕분에 `tensor.to_vec::<f32>()`처럼 타입을
/// 지정해 꺼낼 수 있다.
pub trait Element: Copy + fmt::Debug + storage::Sealed + 'static {
    /// 이 타입에 대응하는 자료형.
    const DTYPE: DType;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
}
// ANCHOR_END: element

// ANCHOR: tensor
/// 형상이 붙은 숫자 배열.
///
/// 복제(`clone`)는 저장소를 복사하지 않고 공유하므로 비용이 거의 없다.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// 숫자가 들어 있는 저장소. 여러 텐서가 함께 가리킬 수 있다.
    storage: Arc<Storage>,
    dtype: DType,
    /// 각 축의 길이.
    shape: Vec<usize>,
    /// 각 축에서 한 칸 움직일 때 저장소에서 건너뛰는 원소 수.
    strides: Vec<usize>,
    /// 첫 원소가 저장소의 몇 번째에 있는지.
    offset: usize,
}
// ANCHOR_END: tensor

// ANCHOR: contiguous_strides
/// 행 우선으로 빈틈없이 놓인 텐서의 스트라이드.
///
/// 마지막 축의 스트라이드는 1이고, 앞 축으로 갈수록 뒤 축들의 길이를
/// 곱한 값이 된다. 형상 `(2, 3, 4)`의 스트라이드는 `(12, 4, 1)`이다.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}
// ANCHOR_END: contiguous_strides

// ANCHOR: broadcast_shapes
/// 두 형상을 브로드캐스팅했을 때의 결과 형상.
///
/// 두 형상을 오른쪽 끝에 맞춰 놓고 축마다 비교한다. 길이가 같거나 한쪽이
/// 1이면 맞는 것이고, 결과는 둘 중 큰 쪽이다. 차원 수가 모자란 쪽은 앞에
/// 1이 있는 것으로 본다.
pub fn broadcast_shapes(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let ndim = left.len().max(right.len());
    let mut shape = vec![0; ndim];
    for i in 0..ndim {
        // 오른쪽 끝에서부터 i번째 축. 없으면 길이 1로 본다.
        let l = if i < left.len() {
            left[left.len() - 1 - i]
        } else {
            1
        };
        let r = if i < right.len() {
            right[right.len() - 1 - i]
        } else {
            1
        };
        shape[ndim - 1 - i] = match (l, r) {
            (l, r) if l == r => l,
            (1, r) => r,
            (l, 1) => l,
            _ => {
                return Err(Error::Shape {
                    operation: "브로드캐스팅",
                    left: left.to_vec(),
                    right: right.to_vec(),
                });
            }
        };
    }
    Ok(shape)
}
// ANCHOR_END: broadcast_shapes

impl Tensor {
    // ANCHOR: from_vec
    /// 벡터와 형상으로 텐서를 만든다. 벡터는 행 우선 순서로 읽는다.
    ///
    /// 벡터의 길이가 형상의 원소 수와 다르면 오류를 돌려준다.
    pub fn from_vec<T: Element>(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(Error::ElementCount {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            storage: Arc::new(T::wrap(data)),
            dtype: T::DTYPE,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        })
    }
    // ANCHOR_END: from_vec

    /// 0으로 채운 텐서를 만든다.
    pub fn zeros(dtype: DType, shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        let storage = match dtype {
            DType::F32 => Storage::F32(vec![0.0; len]),
            DType::I32 => Storage::I32(vec![0; len]),
            DType::U8 => Storage::U8(vec![0; len]),
        };
        Self {
            storage: Arc::new(storage),
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        }
    }

    // ANCHOR: from_bytes
    /// 다른 곳에 있는 바이트 구간을 텐서로 읽는다. 값은 리틀 엔디언이어야
    /// 한다.
    ///
    /// 구간의 시작 주소가 자료형의 정렬에 맞으면 복사하지 않고 그 메모리를
    /// 빌려 쓴다. 맞지 않으면 값을 새 저장소로 복사한다. 어느 쪽이었는지는
    /// [`is_borrowed`](Self::is_borrowed)로 알 수 있다.
    pub fn from_bytes(
        owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
        range: Range<usize>,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Self> {
        let expected: usize = shape.iter().product();
        let bytes = (*owner).as_ref().get(range.clone());
        let actual = bytes.map_or(0, |bytes| bytes.len() / dtype.size_in_bytes());
        let Some(bytes) = bytes.filter(|bytes| bytes.len() == expected * dtype.size_in_bytes())
        else {
            return Err(Error::ElementCount { expected, actual });
        };

        fn build<T: Element>(owner: &ByteOwner, range: Range<usize>, bytes: &[u8]) -> Storage {
            match storage::cast_bytes::<T>(bytes) {
                Some(_) => Storage::Borrowed {
                    owner: Arc::clone(owner),
                    range,
                    dtype: T::DTYPE,
                },
                None => T::wrap(T::from_le_bytes(bytes)),
            }
        }
        let storage = match dtype {
            DType::F32 => build::<f32>(&owner, range, bytes),
            DType::I32 => build::<i32>(&owner, range, bytes),
            DType::U8 => build::<u8>(&owner, range, bytes),
        };
        Ok(Self {
            storage: Arc::new(storage),
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        })
    }

    /// 저장소가 다른 곳의 메모리를 빌려 쓰고 있는지.
    pub fn is_borrowed(&self) -> bool {
        matches!(*self.storage, Storage::Borrowed { .. })
    }
    // ANCHOR_END: from_bytes

    /// 각 축의 길이.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 각 축의 스트라이드. 단위는 바이트가 아니라 원소 수다.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// 첫 원소가 저장소에서 시작하는 위치.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 원소의 자료형.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 차원 수. 스칼라는 0, 벡터는 1, 행렬은 2다.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// 원소의 개수. 형상의 수를 모두 곱한 값이다.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// 원소가 하나도 없는지.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 이 텐서의 원소들이 차지하는 바이트 수.
    pub fn size_in_bytes(&self) -> usize {
        self.len() * self.dtype.size_in_bytes()
    }

    /// 두 텐서가 같은 저장소를 가리키는지. 뷰인지 복사본인지 확인할 때 쓴다.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// 저장소 전체를 타입이 붙은 슬라이스로 꺼낸다.
    fn storage_slice<T: Element>(&self) -> Result<&[T]> {
        T::unwrap(&self.storage).ok_or(Error::DType {
            expected: T::DTYPE,
            actual: self.dtype,
        })
    }

    fn check_axis(&self, axis: usize) -> Result<()> {
        if axis < self.ndim() {
            Ok(())
        } else {
            Err(Error::Axis {
                axis,
                ndim: self.ndim(),
            })
        }
    }

    /// 같은 저장소를 가리키되 메타데이터만 다른 텐서를 만든다.
    fn view(&self, shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape,
            strides,
            offset,
        }
    }

    // ANCHOR: get
    /// 위치 하나의 원소를 읽는다. `index`에는 축마다 위치를 하나씩 준다.
    pub fn get<T: Element>(&self, index: &[usize]) -> Result<T> {
        if index.len() != self.ndim() {
            return Err(Error::Axis {
                axis: index.len(),
                ndim: self.ndim(),
            });
        }
        let mut position = self.offset;
        for (axis, (&i, &len)) in index.iter().zip(&self.shape).enumerate() {
            if i >= len {
                return Err(Error::OutOfBounds {
                    axis,
                    index: i,
                    len,
                });
            }
            position += i * self.strides[axis];
        }
        Ok(self.storage_slice::<T>()?[position])
    }
    // ANCHOR_END: get

    // ANCHOR: is_contiguous
    /// 원소들이 저장소에 행 우선 순서로 빈틈없이 놓여 있는지.
    ///
    /// 그렇다면 저장소의 한 구간을 그대로 슬라이스로 쓸 수 있다.
    pub fn is_contiguous(&self) -> bool {
        // 길이가 1인 축은 움직일 일이 없으므로 스트라이드가 무엇이든 상관없다.
        let mut expected = 1;
        for (&len, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if len != 1 && stride != expected {
                return false;
            }
            expected *= len;
        }
        true
    }
    // ANCHOR_END: is_contiguous

    // ANCHOR: as_slice
    /// 원소 전체를 슬라이스로 빌린다. 복사하지 않는다.
    ///
    /// 연속이 아닌 뷰에는 쓸 수 없다. 먼저 [`contiguous`](Self::contiguous)를
    /// 호출해야 한다.
    pub fn as_slice<T: Element>(&self) -> Result<&[T]> {
        if !self.is_contiguous() {
            return Err(Error::NotContiguous);
        }
        Ok(&self.storage_slice::<T>()?[self.offset..self.offset + self.len()])
    }
    // ANCHOR_END: as_slice

    // ANCHOR: iter
    /// 원소를 논리적 순서(행 우선)로 하나씩 내놓는다.
    ///
    /// 연속이 아닌 뷰에서도 동작한다. 저장소의 실제 배치와 무관하게 "이
    /// 텐서를 행 우선으로 읽으면 나오는 순서"를 따른다.
    pub fn iter<T: Element>(&self) -> Result<Iter<'_, T>> {
        Ok(Iter {
            data: self.storage_slice()?,
            shape: &self.shape,
            strides: &self.strides,
            index: vec![0; self.ndim()],
            position: self.offset,
            remaining: self.len(),
        })
    }

    /// 원소를 논리적 순서로 복사해 새 벡터로 돌려준다.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        match self.as_slice::<T>() {
            Ok(slice) => Ok(slice.to_vec()),
            Err(Error::NotContiguous) => Ok(self.iter::<T>()?.collect()),
            Err(error) => Err(error),
        }
    }
    // ANCHOR_END: iter

    // ANCHOR: contiguous
    /// 연속인 텐서를 돌려준다.
    ///
    /// 이미 연속이면 저장소를 공유하는 복제본을, 아니면 원소를 논리적
    /// 순서로 새 저장소에 복사한 텐서를 돌려준다.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        fn copy<T: Element>(tensor: &Tensor) -> Tensor {
            let data: Vec<T> = tensor.iter().expect("자료형을 확인하고 호출한다").collect();
            Tensor::from_vec(data, &tensor.shape).expect("원소 수는 형상과 같다")
        }
        match self.dtype {
            DType::F32 => copy::<f32>(self),
            DType::I32 => copy::<i32>(self),
            DType::U8 => copy::<u8>(self),
        }
    }
    // ANCHOR_END: contiguous

    // ANCHOR: reshape
    /// 같은 원소를 다른 형상으로 읽는 뷰를 만든다.
    ///
    /// 원소 수가 같아야 하고, 텐서가 연속이어야 한다.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if expected != self.len() {
            return Err(Error::Shape {
                operation: "reshape",
                left: self.shape.clone(),
                right: shape.to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(Error::NotContiguous);
        }
        Ok(self.view(shape.to_vec(), contiguous_strides(shape), self.offset))
    }
    // ANCHOR_END: reshape

    // ANCHOR: permute
    /// 축의 순서를 바꾼 뷰를 만든다. `axes[i]`는 결과의 `i`번째 축이 될
    /// 원래 축의 번호다.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let mut seen = vec![false; self.ndim()];
        let is_permutation = axes.len() == self.ndim()
            && axes
                .iter()
                .all(|&axis| axis < self.ndim() && !std::mem::replace(&mut seen[axis], true));
        if !is_permutation {
            return Err(Error::Shape {
                operation: "permute",
                left: self.shape.clone(),
                right: axes.to_vec(),
            });
        }
        let shape = axes.iter().map(|&axis| self.shape[axis]).collect();
        let strides = axes.iter().map(|&axis| self.strides[axis]).collect();
        Ok(self.view(shape, strides, self.offset))
    }

    /// 두 축을 맞바꾼 뷰를 만든다. 행렬에서는 전치다.
    pub fn transpose(&self, first: usize, second: usize) -> Result<Self> {
        self.check_axis(first)?;
        self.check_axis(second)?;
        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        axes.swap(first, second);
        self.permute(&axes)
    }
    // ANCHOR_END: permute

    // ANCHOR: slice
    /// 한 축에서 `range` 구간만 남긴 뷰를 만든다.
    pub fn slice(&self, axis: usize, range: Range<usize>) -> Result<Self> {
        self.check_axis(axis)?;
        let len = self.shape[axis];
        if range.start > range.end || range.end > len {
            return Err(Error::OutOfBounds {
                axis,
                index: range.end.max(range.start),
                len,
            });
        }
        let mut shape = self.shape.clone();
        shape[axis] = range.end - range.start;
        let offset = self.offset + range.start * self.strides[axis];
        Ok(self.view(shape, self.strides.clone(), offset))
    }
    // ANCHOR_END: slice

    // ANCHOR: squeeze
    /// 길이 1인 축을 `axis` 자리에 끼워 넣은 뷰를 만든다.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(Error::Axis {
                axis,
                ndim: self.ndim() + 1,
            });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.insert(axis, 1);
        // 길이 1인 축은 움직이지 않으므로 스트라이드 값은 쓰이지 않는다.
        strides.insert(axis, 1);
        Ok(self.view(shape, strides, self.offset))
    }

    /// 길이 1인 축 `axis`를 없앤 뷰를 만든다.
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        self.check_axis(axis)?;
        if self.shape[axis] != 1 {
            return Err(Error::Shape {
                operation: "squeeze",
                left: self.shape.clone(),
                right: vec![axis],
            });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.remove(axis);
        strides.remove(axis);
        Ok(self.view(shape, strides, self.offset))
    }
    // ANCHOR_END: squeeze

    // ANCHOR: broadcast_to
    /// 이 텐서를 더 큰 형상으로 늘려 읽는 뷰를 만든다.
    ///
    /// 길이 1인 축을 원하는 길이로 늘리고, 모자란 차원은 앞에 붙인다.
    /// 늘어난 축의 스트라이드를 0으로 두므로 숫자는 복사되지 않는다. 그
    /// 축에서는 아무리 움직여도 같은 원소를 다시 읽게 된다.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let error = || Error::Shape {
            operation: "broadcast_to",
            left: self.shape.clone(),
            right: shape.to_vec(),
        };
        if shape.len() < self.ndim() {
            return Err(error());
        }
        let added = shape.len() - self.ndim();
        let mut strides = vec![0; shape.len()];
        for (axis, &target) in shape.iter().enumerate().skip(added) {
            let source = self.shape[axis - added];
            strides[axis] = match source {
                len if len == target => self.strides[axis - added],
                1 => 0,
                _ => return Err(error()),
            };
        }
        Ok(self.view(shape.to_vec(), strides, self.offset))
    }
    // ANCHOR_END: broadcast_to
}

// ANCHOR: iter_struct
/// [`Tensor::iter`]가 돌려주는 반복자.
#[derive(Debug)]
pub struct Iter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    /// 지금 읽을 원소의 축별 위치.
    index: Vec<usize>,
    /// 지금 읽을 원소가 저장소의 몇 번째에 있는지.
    position: usize,
    remaining: usize,
}

impl<T: Element> Iterator for Iter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        let value = self.data[self.position];
        self.remaining -= 1;
        // 자동차 주행 거리계처럼 마지막 축부터 한 칸 올린다. 끝에 닿은
        // 축은 0으로 되돌리고 앞 축을 올린다.
        for axis in (0..self.shape.len()).rev() {
            self.index[axis] += 1;
            self.position += self.strides[axis];
            if self.index[axis] < self.shape[axis] {
                break;
            }
            self.position -= self.strides[axis] * self.shape[axis];
            self.index[axis] = 0;
        }
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Element> ExactSizeIterator for Iter<'_, T> {}
// ANCHOR_END: iter_struct

#[cfg(test)]
mod tests {
    use super::*;

    /// 0, 1, 2, ... 로 채운 f32 텐서.
    fn arange(shape: &[usize]) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor::from_vec((0..len).map(|i| i as f32).collect(), shape).unwrap()
    }

    #[test]
    fn from_vec_checks_element_count() {
        assert_eq!(
            Tensor::from_vec(vec![1.0_f32; 5], &[2, 3]).unwrap_err(),
            Error::ElementCount {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn metadata_of_a_fresh_tensor() {
        let t = arange(&[2, 3, 4]);
        assert_eq!(t.shape(), [2, 3, 4]);
        assert_eq!(t.strides(), [12, 4, 1]);
        assert_eq!((t.ndim(), t.len(), t.size_in_bytes()), (3, 24, 96));
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_contiguous());
    }

    #[test]
    fn scalars_and_empty_tensors() {
        let scalar = Tensor::from_vec(vec![7_i32], &[]).unwrap();
        assert_eq!((scalar.ndim(), scalar.len()), (0, 1));
        assert_eq!(scalar.get::<i32>(&[]).unwrap(), 7);
        assert_eq!(scalar.to_vec::<i32>().unwrap(), [7]);

        let empty = Tensor::zeros(DType::U8, &[3, 0, 2]);
        assert!(empty.is_empty());
        assert_eq!(empty.to_vec::<u8>().unwrap(), []);
    }

    #[test]
    fn get_follows_the_stride_formula() {
        let t = arange(&[2, 3, 4]);
        // 1 × 12 + 2 × 4 + 3 × 1 = 23
        assert_eq!(t.get::<f32>(&[1, 2, 3]).unwrap(), 23.0);
        assert_eq!(
            t.get::<f32>(&[1, 3, 0]).unwrap_err(),
            Error::OutOfBounds {
                axis: 1,
                index: 3,
                len: 3
            }
        );
    }

    #[test]
    fn typed_access_rejects_the_wrong_dtype() {
        let t = Tensor::from_vec(vec![1_u8, 2, 3], &[3]).unwrap();
        assert_eq!(
            t.to_vec::<f32>().unwrap_err(),
            Error::DType {
                expected: DType::F32,
                actual: DType::U8
            }
        );
    }

    #[test]
    fn transpose_is_a_view_with_swapped_strides() {
        let t = arange(&[2, 3]);
        let transposed = t.transpose(0, 1).unwrap();
        assert!(transposed.shares_storage_with(&t));
        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.strides(), [1, 3]);
        assert!(!transposed.is_contiguous());
        // [0 1 2]ᵀ   [0 3]
        // [3 4 5]  = [1 4]
        //            [2 5]
        assert_eq!(
            transposed.to_vec::<f32>().unwrap(),
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn transpose_agrees_with_the_copying_version_from_chapter_2() {
        let t = arange(&[3, 4]);
        let view = t.transpose(0, 1).unwrap().to_vec::<f32>().unwrap();
        let copied = crate::linalg::transpose(t.as_slice().unwrap(), 3, 4);
        assert_eq!(view, copied);
    }

    #[test]
    fn permute_reorders_axes() {
        let t = arange(&[2, 3, 4]);
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), [4, 2, 3]);
        assert_eq!(p.strides(), [1, 12, 4]);
        assert_eq!(
            p.get::<f32>(&[3, 1, 2]).unwrap(),
            t.get::<f32>(&[1, 2, 3]).unwrap()
        );
        assert!(t.permute(&[0, 0, 1]).is_err());
        assert!(t.permute(&[0, 1]).is_err());
    }

    #[test]
    fn slice_moves_the_offset() {
        let t = arange(&[3, 4]);
        let rows = t.slice(0, 1..3).unwrap();
        assert!(rows.shares_storage_with(&t));
        assert_eq!((rows.shape(), rows.offset()), (&[2, 4][..], 4));
        assert!(rows.is_contiguous());
        assert_eq!(
            rows.as_slice::<f32>().unwrap(),
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );

        let columns = t.slice(1, 1..3).unwrap();
        assert!(!columns.is_contiguous());
        assert_eq!(
            columns.to_vec::<f32>().unwrap(),
            [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
        );
        assert!(t.slice(1, 2..5).is_err());
    }

    #[test]
    fn reshape_is_a_view_but_needs_contiguous_data() {
        let t = arange(&[2, 6]);
        let r = t.reshape(&[3, 4]).unwrap();
        assert!(r.shares_storage_with(&t));
        assert_eq!(r.get::<f32>(&[2, 1]).unwrap(), 9.0);
        assert!(t.reshape(&[5, 2]).is_err());

        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.reshape(&[12]).unwrap_err(), Error::NotContiguous);
        let copied = transposed.contiguous();
        assert!(!copied.shares_storage_with(&t));
        assert_eq!(
            copied.reshape(&[12]).unwrap().as_slice::<f32>().unwrap()[..4],
            [0.0, 6.0, 1.0, 7.0]
        );
    }

    #[test]
    fn contiguous_does_not_copy_when_not_needed() {
        let t = arange(&[2, 3]);
        assert!(t.contiguous().shares_storage_with(&t));
    }

    #[test]
    fn squeeze_and_unsqueeze_change_only_the_shape() {
        let t = arange(&[3, 4]);
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), [1, 3, 4]);
        assert!(u.is_contiguous());
        assert_eq!(u.squeeze(0).unwrap().shape(), [3, 4]);
        assert_eq!(t.unsqueeze(2).unwrap().shape(), [3, 4, 1]);
        assert!(t.squeeze(0).is_err());
    }

    #[test]
    fn broadcast_shapes_follows_the_rule() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]).unwrap(), [3, 4]);
        assert_eq!(broadcast_shapes(&[4], &[2, 3, 4]).unwrap(), [2, 3, 4]);
        assert_eq!(broadcast_shapes(&[], &[5]).unwrap(), [5]);
        assert_eq!(broadcast_shapes(&[2, 1, 4], &[3, 1]).unwrap(), [2, 3, 4]);
        assert!(broadcast_shapes(&[3], &[4]).is_err());
    }

    #[test]
    fn broadcast_to_repeats_without_copying() {
        let column = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0], &[3, 1]).unwrap();
        let b = column.broadcast_to(&[3, 4]).unwrap();
        assert!(b.shares_storage_with(&column));
        assert_eq!(b.strides(), [1, 0]);
        assert_eq!(
            b.to_vec::<f32>().unwrap(),
            [
                10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0
            ]
        );

        let row = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).unwrap();
        let b = row.broadcast_to(&[3, 2]).unwrap();
        assert_eq!(b.strides(), [0, 1]);
        assert_eq!(b.to_vec::<f32>().unwrap(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert!(row.broadcast_to(&[2, 3]).is_err());
    }

    #[test]
    fn views_compose() {
        // 자르고, 전치하고, 늘린 뒤에도 원래 저장소 하나만 쓴다.
        let t = arange(&[4, 5]);
        let v = t.slice(0, 1..3).unwrap().transpose(0, 1).unwrap();
        assert_eq!(v.shape(), [5, 2]);
        assert_eq!(v.get::<f32>(&[4, 1]).unwrap(), 14.0);
        assert!(v.shares_storage_with(&t));
    }

    #[test]
    fn iterator_reports_its_length() {
        let t = arange(&[2, 3]).transpose(0, 1).unwrap();
        let iter = t.iter::<f32>().unwrap();
        assert_eq!(iter.len(), 6);
    }

    #[test]
    fn from_bytes_borrows_aligned_memory() {
        // Vec<f32>의 메모리를 바이트로 본 것이므로 f32 정렬이 보장된다.
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(AlignedBytes::new(&bytes));

        let t = Tensor::from_bytes(Arc::clone(&owner), 0..24, DType::F32, &[2, 3]).unwrap();
        assert!(t.is_borrowed());
        assert_eq!(t.as_slice::<f32>().unwrap(), values);
        assert_eq!(t.transpose(0, 1).unwrap().get::<f32>(&[2, 1]).unwrap(), 6.0);

        // 주인을 놓아도 텐서가 메모리를 붙들고 있다.
        drop(owner);
        assert_eq!(t.get::<f32>(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn from_bytes_copies_misaligned_memory() {
        let values = [1.0_f32, -2.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // 한 바이트 밀어서 일부러 정렬을 깨뜨린다.
        let mut shifted = vec![0_u8];
        shifted.extend(&bytes);
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(AlignedBytes::new(&shifted));

        let t = Tensor::from_bytes(owner, 1..9, DType::F32, &[2]).unwrap();
        assert!(!t.is_borrowed());
        assert_eq!(t.as_slice::<f32>().unwrap(), values);
    }

    #[test]
    fn from_bytes_checks_the_range() {
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(vec![0_u8; 8]);
        // 형상은 원소 3개를 요구하는데 구간에는 2개뿐이다.
        assert_eq!(
            Tensor::from_bytes(Arc::clone(&owner), 0..8, DType::F32, &[3]).unwrap_err(),
            Error::ElementCount {
                expected: 3,
                actual: 2
            }
        );
        // 구간이 주인의 메모리를 벗어난다.
        assert!(Tensor::from_bytes(owner, 4..12, DType::F32, &[2]).is_err());
    }

    #[test]
    fn typed_access_to_borrowed_storage_checks_the_dtype() {
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(vec![1_u8, 2, 3, 4]);
        let t = Tensor::from_bytes(owner, 0..4, DType::U8, &[4]).unwrap();
        assert!(t.is_borrowed());
        assert_eq!(t.to_vec::<u8>().unwrap(), [1, 2, 3, 4]);
        assert!(t.to_vec::<f32>().is_err());
    }

    /// 8바이트 정렬이 보장되는 바이트 덩어리. 테스트에서 정렬을 통제하려고 쓴다.
    struct AlignedBytes {
        words: Vec<u64>,
        len: usize,
    }

    impl AlignedBytes {
        fn new(bytes: &[u8]) -> Self {
            let mut words = vec![0_u64; bytes.len().div_ceil(8)];
            for (word, chunk) in words.iter_mut().zip(bytes.chunks(8)) {
                let mut buffer = [0_u8; 8];
                buffer[..chunk.len()].copy_from_slice(chunk);
                *word = u64::from_ne_bytes(buffer);
            }
            Self {
                words,
                len: bytes.len(),
            }
        }
    }

    impl AsRef<[u8]> for AlignedBytes {
        fn as_ref(&self) -> &[u8] {
            // SAFETY: u64의 메모리를 u8로 읽는 것은 언제나 유효하고, 길이는
            // 할당한 범위 안이다.
            let (_, bytes, _) = unsafe { self.words.align_to::<u8>() };
            &bytes[..self.len]
        }
    }
}
