//! SME(Scalable Matrix Extension)의 외적 명령으로 짠 선형 층 (31장).
//!
//! NEON의 곱하고 더하기 명령 하나는 `f32` 넷을 계산한다. SME의 `FMOPA`는
//! 벡터 둘의 **외적**을 통째로 계산해 행렬 모양의 레지스터 `ZA`에 더한다.
//! 벡터가 `f32` 열여섯 개짜리이면 명령 하나가 곱셈 256번이다.
//!
//! `y[i][r] = Σₖ x[i][k] × w[r][k]`를 `k`의 순서로 풀어 쓰면, `k`마다
//! "입력 벡터들의 `k`번째 값을 모은 벡터"와 "가중치 행들의 `k`번째 값을
//! 모은 벡터"의 외적을 더하는 일이다. 그래서 두 행렬을 모두 **열을 따라
//! 읽을 수 있게** 다시 늘어놓아야 한다(packing).
//!
//! Rust의 표준 라이브러리에는 SME 명령을 내는 함수가 없다. 인라인
//! 어셈블리로 직접 적는다.

use std::arch::asm;
use std::sync::OnceLock;

/// 이 프로세서와 운영체제가 SME를 쓸 수 있게 해 주는지.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(detect)
}

#[cfg(all(target_os = "macos", not(miri)))]
fn detect() -> bool {
    // macOS는 프로세서의 기능을 `sysctl`로 알려 준다. 표준 라이브러리가
    // 이미 이 함수가 든 시스템 라이브러리에 연결되어 있으므로 선언만 하면
    // 부를 수 있다.
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> std::ffi::c_int;
    }
    let mut value: u32 = 0;
    let mut len = size_of::<u32>();
    // SAFETY: 이름은 0으로 끝나는 문자열이고, 값을 받을 자리와 그 크기를
    // 맞게 넘긴다.
    let status = unsafe {
        sysctlbyname(
            c"hw.optional.arm.FEAT_SME".as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    status == 0 && value == 1
}

#[cfg(not(all(target_os = "macos", not(miri))))]
fn detect() -> bool {
    // 다른 운영체제에서는 확인하는 방법을 시험해 보지 못했다. 쓰지 않는다.
    false
}

// ANCHOR: clobbers
/// SME 명령을 쓰는 어셈블리 조각. 망가뜨리는 레지스터를 모두 적어 둔다.
///
/// 스트리밍 모드(SME 명령을 쓸 수 있는 상태)에 들어가고 나올 때 벡터
/// 레지스터와 술어 레지스터의 내용이 **모두 지워진다**. 컴파일러가 그
/// 레지스터에 살려 둔 값이 있으면 사라지므로, 전부 망가뜨린다고 알린다.
macro_rules! streaming {
    ($($body:tt)*) => {
        asm!(
            // 어셈블러에게 SME 명령을 받아들이라고 알린다.
            ".arch_extension sme",
            $($body)*
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _,
            out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _,
            out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            out("ffr") _,
            options(nostack),
        )
    };
}
// ANCHOR_END: clobbers

/// 스트리밍 모드에서 벡터 하나에 들어가는 `f32`의 수. 프로세서마다 다르다.
/// SME를 쓸 수 없으면 0이다.
pub fn lanes() -> usize {
    static LANES: OnceLock<usize> = OnceLock::new();
    *LANES.get_or_init(|| {
        if !available() {
            return 0;
        }
        let lanes: u64;
        // SAFETY: SME가 있음을 확인했다. 모드에 들어가 길이만 읽고 나온다.
        unsafe {
            streaming!(
                "smstart sm",
                "cntw {lanes}",
                "smstop sm",
                lanes = out(reg) lanes,
            );
        }
        lanes as usize
    })
}

// ANCHOR: packed
/// 열을 따라 읽을 수 있게 다시 늘어놓은 가중치.
///
/// 행 `lanes`개가 한 묶음이다. 묶음마다 `k`의 순서로, "그 묶음의 행들의
/// `k`번째 값" `lanes`개가 이어져 있다. 남는 자리는 0이다.
#[derive(Debug, Clone)]
pub struct Packed {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    lanes: usize,
}

impl Packed {
    /// `(rows, cols)` 행렬을 다시 늘어놓는다. SME를 쓸 수 없으면 `None`이다.
    pub fn new(w: &[f32], rows: usize, cols: usize) -> Option<Self> {
        let lanes = lanes();
        if lanes == 0 || w.len() != rows * cols {
            return None;
        }
        // 커널이 묶음을 넷씩 읽으므로 넷의 배수가 되게 0으로 채운다.
        let groups = rows.div_ceil(lanes).next_multiple_of(4);
        let mut data = vec![0.0; groups * cols * lanes];
        for (row, values) in w.chunks_exact(cols.max(1)).enumerate() {
            let base = (row / lanes) * cols * lanes + row % lanes;
            for (k, &value) in values.iter().enumerate() {
                data[base + k * lanes] = value;
            }
        }
        Some(Self {
            data,
            rows,
            cols,
            lanes,
        })
    }

    /// 행의 수.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 열의 수.
    pub fn cols(&self) -> usize {
        self.cols
    }
}
// ANCHOR_END: packed

// ANCHOR: pack_inputs
/// 입력 벡터 `count`개(`lanes`개 이하)를 열을 따라 읽을 수 있게 늘어놓는다.
/// `packed[k × lanes + i] = x[i × cols + k]`. 남는 자리는 0이다.
pub fn pack_inputs(x: &[f32], count: usize, cols: usize, lanes: usize, packed: &mut [f32]) {
    assert!(count <= lanes && x.len() >= count * cols && packed.len() >= cols * lanes);
    if count < lanes {
        packed.fill(0.0);
    }
    // 벡터 넷, 값 넷씩 묶어 4 × 4 행렬을 뒤집는다(전치). 값 하나씩 옮기는
    // 것보다 네 배쯤 적게 읽고 쓴다.
    let mut i = 0;
    while i + 4 <= count {
        let rows: [&[f32]; 4] = std::array::from_fn(|r| &x[(i + r) * cols..][..cols]);
        let mut k = 0;
        #[cfg(not(miri))]
        while k + 4 <= cols {
            use std::arch::aarch64::{
                vld1q_f32, vreinterpretq_f32_f64, vreinterpretq_f64_f32, vst1q_f32, vtrn1q_f32,
                vtrn1q_f64, vtrn2q_f32, vtrn2q_f64,
            };
            // SAFETY: `k + 4 <= cols`이므로 읽는 곳은 각 벡터 안이고, 쓰는
            // 곳 `(k + 3) × lanes + i + 3`은 `packed`(길이 `cols × lanes`)
            // 안이다. NEON은 64비트 ARM에 늘 있다.
            unsafe {
                let [a, b, c, d] = rows.map(|row| vld1q_f32(row.as_ptr().add(k)));
                // (a0 b0 a2 b2), (a1 b1 a3 b3), (c0 d0 c2 d2), (c1 d1 c3 d3)
                let (ab0, ab1) = (vtrn1q_f32(a, b), vtrn2q_f32(a, b));
                let (cd0, cd1) = (vtrn1q_f32(c, d), vtrn2q_f32(c, d));
                let wide = |v| vreinterpretq_f64_f32(v);
                let columns = [
                    vtrn1q_f64(wide(ab0), wide(cd0)),
                    vtrn1q_f64(wide(ab1), wide(cd1)),
                    vtrn2q_f64(wide(ab0), wide(cd0)),
                    vtrn2q_f64(wide(ab1), wide(cd1)),
                ];
                for (offset, column) in columns.into_iter().enumerate() {
                    let target = packed.as_mut_ptr().add((k + offset) * lanes + i);
                    vst1q_f32(target, vreinterpretq_f32_f64(column));
                }
            }
            k += 4;
        }
        for k in k..cols {
            for (r, row) in rows.iter().enumerate() {
                packed[k * lanes + i + r] = row[k];
            }
        }
        i += 4;
    }
    for i in i..count {
        for (k, &value) in x[i * cols..][..cols].iter().enumerate() {
            packed[k * lanes + i] = value;
        }
    }
}
// ANCHOR_END: pack_inputs

// ANCHOR: block
/// 늘어놓은 입력과 늘어놓은 가중치의 곱.
///
/// `ZA`를 `f32`의 정사각 타일 넷(`za0`~`za3`)으로 쓴다. 타일 하나가 입력
/// 벡터 `lanes`개 × 가중치 행 `lanes`개의 결과다. 입력 묶음마다, 가중치
/// 묶음 넷마다: `k`의 순서로 입력의 열 하나와 가중치의 열 넷을 읽어 외적
/// 넷을 더하고, 끝나면 타일의 가로줄(입력 벡터 하나의 출력)을 `y`에 적는다.
///
/// 스트리밍 모드에는 한 번만 들어간다. 들어가고 나오는 데 드는 시간이
/// 작지 않아서, 묶음마다 드나들면 작은 층에서 그 시간이 계산보다 커진다.
///
/// # Safety
///
/// SME를 쓸 수 있어야 한다. `inputs`는 입력 묶음(`lanes`개씩)마다
/// `cols × lanes`개, `weights`는 가중치 묶음 넷마다 `4 × cols × lanes`개를
/// 읽을 수 있어야 한다. `y`는 `count × rows`개를 쓸 수 있어야 한다. `count`,
/// `rows`, `cols`는 1 이상이어야 한다.
unsafe fn multiply(
    inputs: *const f32,
    weights: *const f32,
    cols: usize,
    y: *mut f32,
    rows: usize,
    count: usize,
) {
    // SAFETY: 부르는 쪽이 지킨다. 이 안에서는 다른 함수를 부르지 않고,
    // 스택을 쓰지 않고, 나가기 전에 스트리밍 모드와 ZA를 끈다.
    unsafe {
        streaming!(
            // 스트리밍 모드에 들어가고 ZA를 켠다.
            "smstart",
            "ptrue p0.s",
            "cntw x9",                           // lanes
            "mul x10, {cols}, x9",               // 묶음 하나의 값의 수
            // 입력 묶음마다.
            "2:",
            "cmp {count}, x9",
            "csel x6, {count}, x9, lo",          // 이 묶음의 벡터 수
            "mov x15, {w}",
            "mov x17, {rows}",                   // 남은 출력 열
            "mov x7, {y}",
            // 가중치 묶음 넷마다.
            "3:",
            "zero {{za}}",
            "mov x8, {x}",
            "mov x16, {cols}",
            "mov x4, x15",
            "add x11, x15, x10, lsl #2",
            "add x14, x11, x10, lsl #2",
            "add x13, x14, x10, lsl #2",
            "4:",
            "ld1w z0.s, p0/z, [x8]",
            "ld1w z1.s, p0/z, [x4]",
            "ld1w z2.s, p0/z, [x11]",
            "ld1w z3.s, p0/z, [x14]",
            "ld1w z4.s, p0/z, [x13]",
            // 외적을 타일에 더한다. za[i][j] += z0[i] × zN[j].
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
            // 포인터를 벡터 하나만큼 옮긴다.
            "addvl x8, x8, #1",
            "addvl x4, x4, #1",
            "addvl x11, x11, #1",
            "addvl x14, x14, #1",
            "addvl x13, x13, #1",
            "subs x16, x16, #1",
            "b.ne 4b",
            // 타일마다, 남은 열만큼만 적게 하는 술어를 만든다.
            "mov x5, #0",
            "whilelt p1.s, x5, x17",
            "add x5, x5, x9",
            "whilelt p2.s, x5, x17",
            "add x5, x5, x9",
            "whilelt p3.s, x5, x17",
            "add x5, x5, x9",
            "whilelt p4.s, x5, x17",
            "lsl x11, x9, #1",
            "add x13, x11, x9",
            "mov w12, #0",
            "mov x3, x7",
            "5:",
            // 타일의 w12번째 가로줄을 적는다.
            "st1w {{za0h.s[w12, 0]}}, p1, [x3]",
            "st1w {{za1h.s[w12, 0]}}, p2, [x3, x9, lsl #2]",
            "st1w {{za2h.s[w12, 0]}}, p3, [x3, x11, lsl #2]",
            "st1w {{za3h.s[w12, 0]}}, p4, [x3, x13, lsl #2]",
            "add x3, x3, {rows}, lsl #2",
            "add w12, w12, #1",
            "cmp x12, x6",
            "b.lo 5b",
            // 다음 가중치 묶음 넷.
            "add x15, x15, x10, lsl #4",
            "add x7, x7, x9, lsl #4",
            "lsl x5, x9, #2",
            "subs x17, x17, x5",
            "b.hi 3b",
            // 다음 입력 묶음.
            "add {x}, {x}, x10, lsl #2",
            "mul x5, x9, {rows}",
            "add {y}, {y}, x5, lsl #2",
            "subs {count}, {count}, x9",
            "b.hi 2b",
            "smstop",
            x = inout(reg) inputs => _,
            w = in(reg) weights,
            cols = in(reg) cols,
            y = inout(reg) y => _,
            rows = in(reg) rows,
            count = inout(reg) count => _,
            out("x3") _, out("x4") _, out("x5") _, out("x6") _, out("x7") _, out("x8") _,
            out("x9") _, out("x10") _, out("x11") _, out("x12") _, out("x13") _, out("x14") _,
            out("x15") _, out("x16") _, out("x17") _,
        );
    }
}
// ANCHOR_END: block

// ANCHOR: linear
/// 선형 층의 몸통. `x`는 `(count, cols)`, 결과 `y`는 `(count, rows)`다.
///
/// 더하는 순서가 NEON 커널과 다르다(`k`의 순서대로 하나씩 더한다). 그래서
/// 결과가 [`super::linear`]와 비트까지 같지는 않다.
///
/// # Panics
///
/// 크기가 맞지 않으면 공황에 빠진다.
pub fn linear(x: &[f32], count: usize, weights: &Packed, y: &mut [f32]) {
    let (rows, cols, lanes) = (weights.rows, weights.cols, weights.lanes);
    assert_eq!(x.len(), count * cols, "입력의 크기가 맞지 않습니다");
    assert_eq!(y.len(), count * rows, "출력의 크기가 맞지 않습니다");
    if cols == 0 {
        y.fill(0.0);
        return;
    }
    if count == 0 || rows == 0 {
        return;
    }
    let group = cols * lanes;
    // 입력을 모두 늘어놓은 다음에 스트리밍 모드에 들어간다. 늘어놓는 데
    // 쓰는 NEON 명령은 그 모드에서 쓸 수 없다.
    crate::arena::with_scratch(count.div_ceil(lanes) * group, |packed| {
        for (block, first) in (0..count).step_by(lanes).enumerate() {
            let vectors = lanes.min(count - first);
            let source = &x[first * cols..][..vectors * cols];
            pack_inputs(
                source,
                vectors,
                cols,
                lanes,
                &mut packed[block * group..][..group],
            );
        }
        linear_packed(packed, count, weights, y);
    });
}

/// 이미 늘어놓은 입력으로 계산한다. `inputs`는 벡터 `lanes`개의 묶음마다
/// `cols × lanes`개이고, 묶음 안은 [`pack_inputs`]의 순서다. 합성곱처럼
/// 입력을 제 손으로 늘어놓는 쪽이 부른다.
///
/// # Panics
///
/// 크기가 맞지 않으면 공황에 빠진다.
pub fn linear_packed(inputs: &[f32], count: usize, weights: &Packed, y: &mut [f32]) {
    let (rows, cols, lanes) = (weights.rows, weights.cols, weights.lanes);
    assert_eq!(
        inputs.len(),
        count.div_ceil(lanes) * cols * lanes,
        "입력의 크기가 맞지 않습니다"
    );
    assert_eq!(y.len(), count * rows, "출력의 크기가 맞지 않습니다");
    if count == 0 || rows == 0 || cols == 0 {
        y.fill(0.0);
        return;
    }
    // SAFETY: SME가 있다(`Packed`가 만들어졌다). 입력은 묶음마다
    // `cols × lanes`개이고, 가중치는 묶음 넷의 배수만큼 0으로 채워 두었다
    // (`Packed::new`). `y`는 `count × rows`개다.
    unsafe {
        multiply(
            inputs.as_ptr(),
            weights.data.as_ptr(),
            cols,
            y.as_mut_ptr(),
            rows,
            count,
        );
    }
}
// ANCHOR_END: linear

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + seed * 101) % 29) as f32 * 0.07 - 1.0)
            .collect()
    }

    #[test]
    fn matches_the_definition_for_every_remainder() {
        if !available() {
            eprintln!("건너뜀: 이 기계에는 SME가 없습니다");
            return;
        }
        let lanes = lanes();
        assert!(lanes >= 4 && lanes.is_power_of_two());
        // 묶음의 경계 앞뒤의 크기를 고루 넣는다.
        let sizes = [
            1,
            2,
            3,
            5,
            lanes - 1,
            lanes,
            lanes + 1,
            4 * lanes,
            4 * lanes + 3,
            5 * lanes,
        ];
        for &count in &sizes {
            for &rows in &sizes {
                for cols in [1, 3, 4, 7, 19] {
                    let (x, w) = (values(count * cols, 1), values(rows * cols, 2));
                    let packed = Packed::new(&w, rows, cols).unwrap();
                    // 덮어쓰지 않은 자리가 있으면 드러나게 한다.
                    let mut y = vec![f32::NAN; count * rows + 1];
                    y[count * rows] = 7.0;
                    linear(&x, count, &packed, &mut y[..count * rows]);
                    assert_eq!(y[count * rows], 7.0, "출력 밖에 썼다");
                    for i in 0..count {
                        for r in 0..rows {
                            let exact: f64 = (0..cols)
                                .map(|k| f64::from(x[i * cols + k]) * f64::from(w[r * cols + k]))
                                .sum();
                            let got = f64::from(y[i * rows + r]);
                            assert!(
                                (got - exact).abs() < 1e-4,
                                "{count} × {rows} × {cols}의 ({i}, {r}): {got} 대 {exact}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_shapes_are_handled() {
        if !available() {
            return;
        }
        let packed = Packed::new(&[], 3, 0).unwrap();
        let mut y = [1.0; 6];
        linear(&[], 2, &packed, &mut y);
        assert_eq!(y, [0.0; 6]);
        assert!(Packed::new(&[1.0], 2, 2).is_none());
    }

    #[test]
    fn neighbouring_code_keeps_its_registers() {
        if !available() {
            return;
        }
        // 스트리밍 모드에 들어갔다 나오면 벡터 레지스터가 지워진다. 컴파일러가
        // 그것을 알고 값을 지켜 주는지 본다.
        let before: Vec<f64> = (0..16).map(|i| f64::from(i) * 1.5 + 0.25).collect();
        let sum: f64 = before.iter().sum();
        let scaled: Vec<f64> = before.iter().map(|v| v * sum).collect();
        let (x, w) = (values(64, 3), values(64, 4));
        let packed = Packed::new(&w, 8, 8).unwrap();
        let mut y = [0.0; 64];
        linear(&x, 8, &packed, &mut y);
        let after: Vec<f64> = before.iter().map(|v| v * sum).collect();
        assert_eq!(scaled, after);
        assert_eq!(std::hint::black_box(sum), before.iter().sum::<f64>());
    }
}
