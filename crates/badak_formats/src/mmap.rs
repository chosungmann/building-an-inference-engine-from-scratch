//! 파일을 메모리에 비춘다(메모리 매핑).

use std::fs::File;
use std::path::Path;

use crate::error::Result;

// ANCHOR: mmap_struct
/// 읽기 전용으로 메모리에 비춘 파일.
///
/// 파일의 내용을 `&[u8]`처럼 읽을 수 있다. 실제로 읽는 부분만 운영체제가
/// 디스크에서 가져온다. 값이 사라질 때 매핑을 해제한다.
#[derive(Debug)]
pub struct Mmap {
    pointer: *const u8,
    len: usize,
    /// 파일의 한 구간만 비췄을 때, 실제 매핑이 `pointer`보다 몇 바이트
    /// 앞에서 시작하는지 (44장). 파일 전체를 비췄으면 0이다.
    lead: usize,
}

// SAFETY: 매핑된 메모리는 읽기 전용이고, 이 타입은 그 메모리를 바꾸는
// 방법을 제공하지 않는다. 여러 스레드가 동시에 읽어도 안전하다.
unsafe impl Send for Mmap {}
// SAFETY: 위와 같다.
unsafe impl Sync for Mmap {}
// ANCHOR_END: mmap_struct

impl Mmap {
    /// 파일을 열어 전체를 메모리에 비춘다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(std::io::Error::other)?;
        Ok(Self::map(&file, 0, len)?)
    }

    // ANCHOR: open_range
    /// 열려 있는 파일의 한 구간만 비춘다 (44장).
    ///
    /// Android의 앱 꾸러미처럼 큰 파일 안에 모델이 들어 있을 때 쓴다.
    /// `offset`이 페이지의 경계가 아니어도 된다. 구간이 파일을 벗어나면
    /// 오류다.
    pub fn open_range(file: &File, offset: u64, len: usize) -> Result<Self> {
        let size = file.metadata()?.len();
        let end = offset.checked_add(len as u64);
        if end.is_none_or(|end| end > size) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("구간 {offset}+{len}이 파일의 크기 {size}를 벗어납니다"),
            )
            .into());
        }
        Ok(Self::map(file, offset, len)?)
    }
    // ANCHOR_END: open_range

    /// 비춘 메모리의 길이. 파일의 크기와 같다.
    pub fn len(&self) -> usize {
        self.len
    }

    /// 빈 파일인지.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `pointer`는 `map`이 성공해서 얻은 길이 `len`의 읽기 가능한
        // 매핑을 가리키고, 이 매핑은 `drop`에서만 해제된다. 돌려주는
        // 슬라이스의 수명은 `&self`에 묶인다.
        unsafe { std::slice::from_raw_parts(self.pointer, self.len) }
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    use super::Mmap;

    // ANCHOR: mmap_ffi
    // 운영체제의 C 라이브러리 함수를 직접 선언한다. 표준 라이브러리만 쓴다는
    // 원칙 때문에 libc 크레이트를 쓰지 않는다. 아래 상수의 값은 macOS와
    // Linux에서 같다.
    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;
    const MAP_FAILED: *mut c_void = !0 as *mut c_void;
    /// 매핑을 시작할 수 있는 자리의 간격. 쓰이는 모든 페이지 크기의 배수다.
    const ALIGNMENT: u64 = 64 * 1024;

    unsafe extern "C" {
        fn mmap(
            address: *mut c_void,
            length: usize,
            protection: c_int,
            flags: c_int,
            file_descriptor: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(address: *mut c_void, length: usize) -> c_int;
    }
    // ANCHOR_END: mmap_ffi

    // ANCHOR: mmap_map
    impl Mmap {
        pub(super) fn map(file: &File, offset: u64, len: usize) -> io::Result<Self> {
            // 길이 0인 매핑은 허용되지 않으므로 빈 파일은 따로 다룬다.
            if len == 0 {
                return Ok(Self {
                    pointer: std::ptr::null(),
                    len: 0,
                    lead: 0,
                });
            }
            // 매핑은 페이지의 경계에서만 시작할 수 있다. 페이지의 크기는
            // 기기마다 다르므로(4KB, 16KB, 64KB) 어느 것으로도 나누어지는
            // 64KB의 경계로 내려서 시작하고, 그만큼 뒤를 가리킨다.
            let lead = (offset % ALIGNMENT) as usize;
            let start = i64::try_from(offset - lead as u64).map_err(io::Error::other)?;
            // SAFETY: 주소를 운영체제가 고르도록 null을 넘기고, 유효한 파일
            // 디스크립터와 0이 아닌 길이를 준다. 실패하면 MAP_FAILED가
            // 돌아오며 바로 아래에서 확인한다. 파일 디스크립터는 호출이
            // 끝난 뒤 닫혀도 매핑은 유지된다.
            let pointer = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    lead + len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    start,
                )
            };
            if pointer == MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                // SAFETY: 매핑의 길이가 `lead + len`이므로 `lead`만큼 뒤는
                // 같은 매핑 안이다.
                pointer: unsafe { pointer.cast::<u8>().add(lead) },
                len,
                lead,
            })
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            if self.len > 0 {
                // SAFETY: `pointer`에서 `lead`만큼 앞이 `mmap`이 돌려준
                // 주소이고 길이는 `lead + len`이었다. 한 번만 해제한다.
                unsafe {
                    munmap(
                        self.pointer.sub(self.lead).cast_mut().cast(),
                        self.lead + self.len,
                    )
                };
            }
        }
    }
    // ANCHOR_END: mmap_map
}

#[cfg(not(unix))]
mod sys {
    use std::fs::File;
    use std::io::{self, Read, Seek, SeekFrom};

    use super::Mmap;

    // 메모리 매핑을 구현하지 않은 운영체제에서는 파일 전체를 읽어 들인다.
    // 동작은 같고, 읽지 않을 부분까지 미리 읽는다는 점만 다르다.
    impl Mmap {
        pub(super) fn map(mut file: &File, offset: u64, len: usize) -> io::Result<Self> {
            let mut bytes = vec![0_u8; len];
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut bytes)?;
            let bytes = bytes.into_boxed_slice();
            let len = bytes.len();
            Ok(Self {
                pointer: Box::into_raw(bytes).cast(),
                len,
                lead: 0,
            })
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: `pointer`와 `len`은 `Box<[u8]>`에서 얻은 그대로다.
            unsafe {
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    self.pointer.cast_mut(),
                    self.len,
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_file_and_reads_it_back() {
        let path = std::env::temp_dir().join(format!("badak_mmap_test_{}", std::process::id()));
        let content: Vec<u8> = (0..10_000_u32).flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(&path, &content).unwrap();

        let map = Mmap::open(&path).unwrap();
        assert_eq!(map.len(), content.len());
        assert_eq!(map.as_ref(), content);
        drop(map);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn empty_files_and_missing_files() {
        let path = std::env::temp_dir().join(format!("badak_mmap_empty_{}", std::process::id()));
        std::fs::write(&path, []).unwrap();
        let map = Mmap::open(&path).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.as_ref(), [0_u8; 0]);
        std::fs::remove_file(&path).unwrap();

        assert!(Mmap::open(&path).is_err());
    }

    #[test]
    fn maps_a_range_that_starts_anywhere() {
        let path = std::env::temp_dir().join(format!("badak_mmap_range_{}", std::process::id()));
        let content: Vec<u8> = (0..300_000_u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &content).unwrap();
        let file = File::open(&path).unwrap();
        for (offset, len) in [
            (0, 10),
            (1, 4096),
            (65_536, 100),
            (70_001, 200_000),
            (299_999, 1),
        ] {
            let map = Mmap::open_range(&file, offset as u64, len).unwrap();
            assert_eq!(
                map.as_ref(),
                &content[offset..offset + len],
                "{offset}+{len}"
            );
        }
        assert!(Mmap::open_range(&file, 299_999, 2).is_err());
        assert!(Mmap::open_range(&file, u64::MAX, 2).is_err());
        assert!(Mmap::open_range(&file, 5, 0).unwrap().is_empty());
        std::fs::remove_file(&path).unwrap();
    }
}
