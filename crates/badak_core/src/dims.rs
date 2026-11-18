//! 축마다의 수를 담는 짧은 목록 (22장).
//!
//! 텐서의 형상과 스트라이드는 수 서너 개짜리 목록이다. `Vec<usize>`에 담으면
//! 텐서를 하나 만들 때마다 이 작은 목록 둘을 위해 힙에서 자리를 두 번
//! 얻는다. [`Dims`]는 수가 [`INLINE`]개 이하이면 자기 안에 담고, 그보다
//! 많을 때만 힙을 쓴다.

use std::fmt;
use std::ops::{Deref, DerefMut};

/// 힙을 쓰지 않고 담을 수 있는 수의 개수.
pub const INLINE: usize = 4;

// ANCHOR: dims
/// 축마다의 수를 담는 짧은 목록. 형상, 스트라이드, 축별 위치에 쓴다.
///
/// `&[usize]`로 읽을 수 있어서 쓰는 쪽에서는 `Vec<usize>`와 거의 같다.
#[derive(Clone)]
pub struct Dims(Repr);

#[derive(Clone)]
enum Repr {
    /// 수가 적을 때. 앞의 `len`개만 뜻이 있다.
    Inline { len: u8, values: [usize; INLINE] },
    /// 수가 많을 때.
    Heap(Vec<usize>),
}

impl Deref for Dims {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        match &self.0 {
            Repr::Inline { len, values } => &values[..usize::from(*len)],
            Repr::Heap(values) => values,
        }
    }
}

impl From<&[usize]> for Dims {
    fn from(values: &[usize]) -> Self {
        if values.len() <= INLINE {
            let mut inline = [0; INLINE];
            inline[..values.len()].copy_from_slice(values);
            Self(Repr::Inline {
                len: values.len() as u8,
                values: inline,
            })
        } else {
            Self(Repr::Heap(values.to_vec()))
        }
    }
}
// ANCHOR_END: dims

impl Dims {
    /// `value`를 `len`개 담은 목록.
    pub fn filled(value: usize, len: usize) -> Self {
        if len <= INLINE {
            Self(Repr::Inline {
                len: len as u8,
                values: [value; INLINE],
            })
        } else {
            Self(Repr::Heap(vec![value; len]))
        }
    }

    /// 끝에 수 하나를 붙인다.
    pub fn push(&mut self, value: usize) {
        self.insert(self.len(), value);
    }

    /// `index` 자리에 수 하나를 끼워 넣는다. 뒤의 수들은 한 칸씩 밀린다.
    pub fn insert(&mut self, index: usize, value: usize) {
        match &mut self.0 {
            Repr::Inline { len, values } if usize::from(*len) < INLINE => {
                let count = usize::from(*len);
                assert!(index <= count, "끼워 넣을 자리가 목록 밖입니다");
                values.copy_within(index..count, index + 1);
                values[index] = value;
                *len += 1;
            }
            // 안에 담을 자리가 다 찼다. 힙으로 옮긴다.
            Repr::Inline { .. } => {
                let mut spilled = self.to_vec();
                spilled.insert(index, value);
                self.0 = Repr::Heap(spilled);
            }
            Repr::Heap(values) => values.insert(index, value),
        }
    }

    /// `index` 자리의 수를 빼서 돌려준다. 뒤의 수들은 한 칸씩 당겨진다.
    pub fn remove(&mut self, index: usize) -> usize {
        match &mut self.0 {
            Repr::Inline { len, values } => {
                let count = usize::from(*len);
                assert!(index < count, "뺄 자리가 목록 밖입니다");
                let removed = values[index];
                values.copy_within(index + 1..count, index);
                *len -= 1;
                removed
            }
            Repr::Heap(values) => values.remove(index),
        }
    }

    /// 힙을 쓰지 않고 자기 안에 담고 있는지.
    pub fn is_inline(&self) -> bool {
        matches!(self.0, Repr::Inline { .. })
    }
}

impl DerefMut for Dims {
    fn deref_mut(&mut self) -> &mut [usize] {
        match &mut self.0 {
            Repr::Inline { len, values } => &mut values[..usize::from(*len)],
            Repr::Heap(values) => values,
        }
    }
}

impl<'a> IntoIterator for &'a Dims {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Default for Dims {
    fn default() -> Self {
        Self::filled(0, 0)
    }
}

impl FromIterator<usize> for Dims {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut dims = Self::default();
        for value in iter {
            dims.push(value);
        }
        dims
    }
}

impl<const N: usize> From<[usize; N]> for Dims {
    fn from(values: [usize; N]) -> Self {
        Self::from(&values[..])
    }
}

impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl PartialEq for Dims {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for Dims {}

impl PartialEq<[usize]> for Dims {
    fn eq(&self, other: &[usize]) -> bool {
        **self == *other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for Dims {
    fn eq(&self, other: &[usize; N]) -> bool {
        **self == other[..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_lists_stay_inline() {
        let dims = Dims::from(&[2, 3, 4][..]);
        assert!(dims.is_inline());
        assert_eq!(dims, [2, 3, 4]);
        assert_eq!(dims.len(), 3);
        assert_eq!(dims.iter().product::<usize>(), 24);
        assert!(Dims::default().is_empty());
        assert_eq!(Dims::filled(1, 4), [1, 1, 1, 1]);
        assert!(Dims::filled(1, INLINE).is_inline());
    }

    #[test]
    fn long_lists_go_to_the_heap() {
        let dims = Dims::from(&[1, 2, 3, 4, 5, 6][..]);
        assert!(!dims.is_inline());
        assert_eq!(dims, [1, 2, 3, 4, 5, 6]);
        assert_eq!(Dims::filled(7, 5), [7; 5]);
    }

    #[test]
    fn insert_and_remove_shift_the_rest() {
        let mut dims = Dims::from(&[3, 4][..]);
        dims.insert(0, 1);
        assert_eq!(dims, [1, 3, 4]);
        dims.insert(3, 9);
        assert_eq!(dims, [1, 3, 4, 9]);
        assert!(dims.is_inline());
        // 다섯 번째 수는 안에 담지 못한다.
        dims.insert(2, 7);
        assert_eq!(dims, [1, 3, 7, 4, 9]);
        assert!(!dims.is_inline());
        assert_eq!(dims.remove(0), 1);
        assert_eq!(dims, [3, 7, 4, 9]);

        let mut dims = Dims::from(&[5, 6, 7][..]);
        assert_eq!(dims.remove(1), 6);
        assert_eq!(dims, [5, 7]);
        assert_eq!(dims.remove(1), 7);
        assert_eq!(dims.remove(0), 5);
        assert!(dims.is_empty());
    }

    #[test]
    fn collects_and_mutates_like_a_slice() {
        let mut dims: Dims = (0..3).collect();
        dims.swap(0, 2);
        dims[1] = 10;
        assert_eq!(dims, [2, 10, 0]);
        let long: Dims = (0..6).collect();
        assert_eq!(long, [0, 1, 2, 3, 4, 5]);
        assert_eq!(format!("{dims:?}"), "[2, 10, 0]");
    }
}
