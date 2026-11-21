//! 하드웨어 카운터: 프로세서가 세어 주는 명령과 사이클의 수.

// ANCHOR: counters
/// 이 프로세스가 지금까지 실행한 명령의 수와 쓴 사이클의 수.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// 실행을 마친 기계어 명령의 수.
    pub instructions: u64,
    /// 프로세서의 클록이 뛴 횟수. 이 프로세스가 CPU를 쓰는 동안만 센다.
    pub cycles: u64,
}

impl Counters {
    /// 두 시점 사이에 늘어난 만큼.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            instructions: self.instructions - earlier.instructions,
            cycles: self.cycles - earlier.cycles,
        }
    }

    /// 사이클 하나에 실행한 명령의 수(IPC).
    pub fn instructions_per_cycle(self) -> f64 {
        self.instructions as f64 / self.cycles as f64
    }
}
// ANCHOR_END: counters

// ANCHOR: macos
#[cfg(target_os = "macos")]
mod system {
    use super::Counters;

    /// `<sys/resource.h>`의 `struct rusage_info_v4`.
    ///
    /// 16바이트의 식별자 뒤에 64비트 수 35개가 이어진다. 우리가 읽는 것은
    /// 30번째(`ri_instructions`)와 31번째(`ri_cycles`)다.
    #[repr(C)]
    struct RusageInfoV4 {
        uuid: [u8; 16],
        fields: [u64; 35],
    }

    const RUSAGE_INFO_V4: i32 = 4;
    const INSTRUCTIONS: usize = 29;
    const CYCLES: usize = 30;

    // `<libproc.h>`에 선언된 함수. macOS의 시스템 라이브러리에 들어 있다.
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
    }

    pub fn read() -> Option<Counters> {
        let mut info = RusageInfoV4 {
            uuid: [0; 16],
            fields: [0; 35],
        };
        let pid = std::process::id() as i32;
        // SAFETY: `info`는 `rusage_info_v4`와 크기와 레이아웃이 같고, 호출하는
        // 동안 살아 있다. 함수는 그 크기만큼만 쓴다.
        let status = unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V4, &raw mut info) };
        (status == 0 && info.fields[CYCLES] > 0).then(|| Counters {
            instructions: info.fields[INSTRUCTIONS],
            cycles: info.fields[CYCLES],
        })
    }
}
// ANCHOR_END: macos

#[cfg(not(target_os = "macos"))]
mod system {
    use super::Counters;

    /// 다른 운영체제에서는 아직 읽지 않는다.
    pub fn read() -> Option<Counters> {
        None
    }
}

impl Counters {
    /// 지금의 값을 읽는다. 읽을 수 없는 환경이면 `None`이다.
    ///
    /// 프로세스 전체의 값이다. 다른 스레드가 한 일도 함께 들어 있다.
    pub fn now() -> Option<Self> {
        system::read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difference_and_ratio() {
        let before = Counters {
            instructions: 100,
            cycles: 50,
        };
        let after = Counters {
            instructions: 700,
            cycles: 250,
        };
        let spent = after.since(before);
        assert_eq!((spent.instructions, spent.cycles), (600, 200));
        assert_eq!(spent.instructions_per_cycle(), 3.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn counters_grow_with_work() {
        let Some(before) = Counters::now() else {
            // 가상 기계처럼 카운터를 내주지 않는 환경도 있다.
            return;
        };
        let mut sum = 0_u64;
        for i in 0..1_000_000_u64 {
            sum = std::hint::black_box(sum.wrapping_add(i));
        }
        std::hint::black_box(sum);
        let spent = Counters::now().unwrap().since(before);
        // 반복 한 번에 명령이 적어도 하나는 든다.
        assert!(spent.instructions > 1_000_000, "{spent:?}");
        assert!(spent.cycles > 0);
    }
}
