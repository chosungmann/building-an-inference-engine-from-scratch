//! 측정한 환경을 적는다.

use std::process::Command;

/// 명령을 실행해 출력의 첫 줄을 얻는다. 실패하면 `None`.
fn run(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?.trim();
    (output.status.success() && !line.is_empty()).then(|| line.to_string())
}

/// macOS의 전력 모드. `pmset -g`의 출력에서 찾는다.
fn power_mode() -> Option<String> {
    let output = Command::new("pmset").arg("-g").output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().find(|line| line.contains("powermode"))?;
    let mode = match line.split_whitespace().last()? {
        "0" => "보통",
        "1" => "저전력",
        "2" => "고성능",
        _ => return None,
    };
    Some(mode.to_string())
}

// ANCHOR: environment
/// 측정 결과와 함께 남길 환경 정보를 (항목, 값)의 목록으로 모은다.
///
/// 알아낼 수 없는 항목은 뺀다. 같은 숫자를 다시 얻으려는 사람이 무엇을
/// 맞춰야 하는지 알려 주는 것이 목적이다.
pub fn describe() -> Vec<(&'static str, String)> {
    let mut items = Vec::new();
    let mut add = |name, value: Option<String>| {
        if let Some(value) = value {
            items.push((name, value));
        }
    };
    if cfg!(target_os = "macos") {
        add(
            "프로세서",
            run("sysctl", &["-n", "machdep.cpu.brand_string"]),
        );
        add(
            "성능 코어",
            run("sysctl", &["-n", "hw.perflevel0.physicalcpu"]),
        );
        add(
            "효율 코어",
            run("sysctl", &["-n", "hw.perflevel1.physicalcpu"]),
        );
        let memory =
            run("sysctl", &["-n", "hw.memsize"]).and_then(|bytes| bytes.parse::<u64>().ok());
        add("메모리", memory.map(|bytes| format!("{}GB", bytes >> 30)));
        add(
            "운영체제",
            run("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}")),
        );
        // 배터리로 돌 때, 특히 저전력 모드에서는 운영체제가 속도를 낮출 수 있다.
        add("전원", run("pmset", &["-g", "batt"]));
        add("전력 모드", power_mode());
    } else {
        add("프로세서", run("uname", &["-m"]));
        add("운영체제", run("uname", &["-sr"]));
    }
    add("Rust", run("rustc", &["--version"]));
    add(
        "빌드",
        Some(
            if cfg!(debug_assertions) {
                "debug (최적화 없음)"
            } else {
                "release"
            }
            .to_string(),
        ),
    );
    items
}
// ANCHOR_END: environment

/// 환경 정보를 한 줄에 하나씩 출력한다.
pub fn print() {
    for (name, value) in describe() {
        println!("{name}: {value}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_reports_the_build_kind() {
        let items = describe();
        assert!(items.iter().any(|(name, _)| *name == "빌드"));
        assert!(items.iter().all(|(_, value)| !value.is_empty()));
    }
}
