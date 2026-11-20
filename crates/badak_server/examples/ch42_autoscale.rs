//! 42장: 서버를 늘리는 데 걸리는 시간이 기다림을 어떻게 바꾸는지 따져 본다.
//!
//! 실행: `cargo run --release -p badak_server --example ch42_autoscale`
//!
//! **측정이 아니라 모의실험이다.** 요청이 몰리는 때를 지어내고, 서버가
//! 요청 하나를 처리하는 시간과 서버 하나를 새로 띄우는 시간을 숫자로
//! 넣어, 줄이 어떻게 자라고 줄어드는지를 계산한다. 넣은 숫자 가운데 이
//! 책이 잰 것과 지어낸 것을 출력에 구분해 적는다.

/// 요청 하나를 처리하는 시간(초). 39장에서 잰 것: Llama 3.2 1B가 64토큰을
/// 쓰는 데 0.72초.
const SERVICE: f64 = 0.72;
/// 평소에 오는 요청(초당)과 몰릴 때의 요청(초당). 지어낸 것이다.
const CALM: f64 = 1.0;
const RUSH: f64 = 5.0;
/// 평소, 몰림, 다시 평소의 길이(초).
const PHASES: [f64; 3] = [60.0, 120.0, 120.0];
/// 서버 하나가 감당하는 것보다 줄이 이만큼(요청 수) 길면 하나를 더 띄운다.
const THRESHOLD: usize = 4;
/// 줄을 살피는 간격(초).
const CHECK: f64 = 1.0;

/// 씨앗이 같으면 늘 같은 수를 내는 난수.
struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 요청들이 도착하는 시각. 서로 상관없이, 구간마다 정해진 빠르기로.
fn arrivals() -> Vec<f64> {
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let (mut times, mut start) = (Vec::new(), 0.0);
    for (length, rate) in PHASES.iter().zip([CALM, RUSH, CALM]) {
        let mut at = start;
        loop {
            at += -(1.0 - random.uniform()).ln() / rate;
            if at >= start + length {
                break;
            }
            times.push(at);
        }
        start += length;
    }
    times
}

/// 서버를 새로 띄우는 데 `startup`초가 걸릴 때, 요청마다 기다린 시간(초)과
/// 끝에 떠 있는 서버의 수.
fn simulate(arrivals: &[f64], startup: f64) -> (Vec<f64>, usize) {
    // 서버마다 "다음 요청을 받을 수 있는 시각".
    let mut free_at = vec![0.0_f64];
    // 띄우고 있는 서버들이 준비되는 시각.
    let mut starting: Vec<f64> = Vec::new();
    let mut waits = Vec::with_capacity(arrivals.len());
    let mut next_check = 0.0;
    for &arrival in arrivals {
        // 그사이에 준비된 서버를 더한다.
        starting.retain(|&ready| {
            if ready <= arrival {
                free_at.push(ready);
            }
            ready > arrival
        });
        // 줄을 살핀다. 줄 = 아직 시작하지 못한 일의 양을 요청 수로 센 것.
        while next_check <= arrival {
            let backlog: f64 = free_at
                .iter()
                .map(|&free| (free - next_check).max(0.0) / SERVICE)
                .sum();
            let capacity = free_at.len() + starting.len();
            if startup.is_finite() && backlog > (THRESHOLD * capacity) as f64 {
                starting.push(next_check + startup);
            }
            next_check += CHECK;
        }
        // 가장 먼저 비는 서버가 이 요청을 맡는다.
        let (index, &free) = free_at
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("서버가 하나는 있다");
        let begin = free.max(arrival);
        waits.push(begin - arrival);
        free_at[index] = begin + SERVICE;
    }
    (waits, free_at.len() + starting.len())
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn main() {
    let arrivals = arrivals();
    println!(
        "요청 {}개. 평소 초당 {CALM}개로 {}초, 몰릴 때 초당 {RUSH}개로 {}초, 다시 평소로 {}초 (지어낸 것)",
        arrivals.len(),
        PHASES[0],
        PHASES[1],
        PHASES[2]
    );
    println!(
        "서버 하나가 요청 하나에 {SERVICE}초 (39장에서 잰 것). 서버 하나는 초당 {:.1}개를 감당한다",
        1.0 / SERVICE
    );
    println!(
        "줄이 서버마다 {THRESHOLD}개를 넘으면 하나를 더 띄운다. {CHECK}초마다 살핀다 (지어낸 것)\n"
    );
    println!("새 서버가 준비되기까지        기다림: 가운데   95%     가장 김   끝에 뜬 서버");
    let cases = [
        ("0.2초 (이 장에서 잰 것, Llama 3.2 1B)", 0.2),
        ("2초   (이 장에서 본 것, 파일을 디스크에서)", 2.0),
        ("30초  (지어낸 것)", 30.0),
        ("120초 (지어낸 것)", 120.0),
        ("늘리지 않음", f64::INFINITY),
    ];
    for (name, startup) in cases {
        let (waits, servers) = simulate(&arrivals, startup);
        println!(
            "{name:<34} {:>8.1}초 {:>7.1}초 {:>7.1}초   {servers}대",
            quantile(&waits, 0.5),
            quantile(&waits, 0.95),
            quantile(&waits, 1.0),
        );
    }
}
