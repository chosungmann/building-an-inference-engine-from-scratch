//! 음성 구간 검출.
//!
//! 긴 녹음에서 사람이 말하는 구간을 찾는다. Whisper는 한 번에 30초만
//! 받는다. 긴 소리를 30초씩 칼같이 자르면 말하는 도중에 끊긴다. 말이 없는
//! 곳을 찾아 거기서 자른다.
//!
//! 여기의 방법은 가장 단순한 것이다. 짧은 토막마다 소리의 크기를 재고,
//! 조용한 토막들보다 뚜렷이 큰 토막을 말소리로 본다. 배경이 시끄러우면
//! 통하지 않는다. 실전에서는 작은 신경망을 쓴다.

use std::ops::Range;

// ANCHOR: settings
/// 검출의 설정.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// 크기를 재는 토막의 길이(초).
    pub frame: f64,
    /// 조용한 토막보다 이만큼(데시벨) 커야 말소리로 본다.
    pub margin: f64,
    /// 가장 큰 토막보다 이만큼(데시벨) 넘게 작으면 말소리로 보지 않는다.
    pub floor: f64,
    /// 이보다 짧은 침묵(초)은 말 사이의 숨으로 보고 잇는다.
    pub min_silence: f64,
    /// 이보다 짧은 소리(초)는 버린다.
    pub min_speech: f64,
    /// 찾은 구간의 앞뒤에 붙이는 여유(초).
    pub padding: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frame: 0.02,
            margin: 12.0,
            floor: 40.0,
            min_silence: 0.5,
            min_speech: 0.1,
            padding: 0.1,
        }
    }
}
// ANCHOR_END: settings

// ANCHOR: loudness
/// 토막마다의 크기(데시벨). 표본의 제곱의 평균에 로그를 씌운 것이다.
pub fn loudness(samples: &[f32], frame: usize) -> Vec<f64> {
    samples
        .chunks(frame.max(1))
        .map(|chunk| {
            let energy: f64 = chunk.iter().map(|&sample| f64::from(sample).powi(2)).sum();
            10.0 * (energy / chunk.len() as f64 + 1e-10).log10()
        })
        .collect()
}
// ANCHOR_END: loudness

// ANCHOR: segments
/// 말소리가 있는 구간들을 찾는다. 구간은 표본의 번호다.
pub fn segments(samples: &[f32], sample_rate: u32, settings: &Settings) -> Vec<Range<usize>> {
    let rate = f64::from(sample_rate);
    let frame = ((settings.frame * rate) as usize).max(1);
    let levels = loudness(samples, frame);
    if levels.is_empty() {
        return Vec::new();
    }
    // 조용한 토막의 크기: 아래에서 10분의 1 자리의 값.
    let mut sorted = levels.clone();
    sorted.sort_by(f64::total_cmp);
    let quiet = sorted[sorted.len() / 10];
    let loudest = sorted[sorted.len() - 1];
    // 처음부터 끝까지 말소리이면 "조용한 토막"도 말소리다. 임계값이 가장 큰
    // 토막에 너무 가까워지지 않게 6데시벨 아래에서 막는다.
    let threshold = (quiet + settings.margin)
        .min(loudest - 6.0)
        .max(loudest - settings.floor);

    // 임계값을 넘는 토막들이 이어진 곳을 구간으로 모은다.
    let mut found: Vec<Range<usize>> = Vec::new();
    let mut start = None;
    for (index, &level) in levels.iter().chain(&[f64::NEG_INFINITY]).enumerate() {
        match (level > threshold, start) {
            (true, None) => start = Some(index),
            (false, Some(first)) => {
                found.push(first * frame..(index * frame).min(samples.len()));
                start = None;
            }
            _ => {}
        }
    }
    // 짧은 침묵을 사이에 둔 구간들을 잇는다.
    let gap = (settings.min_silence * rate) as usize;
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in found {
        match merged.last_mut() {
            Some(previous) if range.start - previous.end < gap => previous.end = range.end,
            _ => merged.push(range),
        }
    }
    // 너무 짧은 것을 버리고, 앞뒤에 여유를 붙인다.
    let (shortest, padding) = (
        (settings.min_speech * rate) as usize,
        (settings.padding * rate) as usize,
    );
    merged
        .into_iter()
        .filter(|range| range.len() >= shortest)
        .map(|range| range.start.saturating_sub(padding)..(range.end + padding).min(samples.len()))
        .collect()
}
// ANCHOR_END: segments

// ANCHOR: windows
/// 구간들을 길이 `limit`(표본) 이하의 창으로 묶는다.
///
/// 앞에서부터 구간을 하나씩 더하다가 창이 `limit`을 넘게 되면 거기서 끊고
/// 새 창을 시작한다. 그러면 창의 경계는 언제나 침묵이다. 구간 하나가
/// `limit`보다 길면 어쩔 수 없이 그 안에서 자른다.
pub fn windows(segments: &[Range<usize>], limit: usize) -> Vec<Range<usize>> {
    let mut windows: Vec<Range<usize>> = Vec::new();
    for segment in segments {
        let mut start = segment.start;
        while segment.end - start > limit {
            windows.push(start..start + limit);
            start += limit;
        }
        match windows.last_mut() {
            Some(window) if segment.end - window.start <= limit && window.end <= start => {
                window.end = segment.end;
            }
            _ => windows.push(start..segment.end),
        }
    }
    windows
}
// ANCHOR_END: windows

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    /// 초 단위의 (길이, 소리인가)들로 소리를 짓는다. 침묵에도 작은 잡음을
    /// 깐다.
    fn recording(parts: &[(f64, bool)]) -> Vec<f32> {
        let mut samples = Vec::new();
        let mut state = 12345_u32;
        for &(seconds, speech) in parts {
            for t in 0..(seconds * f64::from(RATE)) as usize {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                let tone = (t as f32 * 0.2).sin() * 0.3;
                samples.push(if speech {
                    tone + noise * 0.01
                } else {
                    noise * 0.001
                });
            }
        }
        samples
    }

    fn seconds(range: &Range<usize>) -> (f64, f64) {
        (
            range.start as f64 / f64::from(RATE),
            range.end as f64 / f64::from(RATE),
        )
    }

    #[test]
    fn finds_speech_between_silences() {
        // 침묵 1초, 말 2초, 숨 0.2초, 말 1초, 침묵 2초, 딸깍 0.05초, 침묵 1초, 말 1초.
        let parts = [
            (1.0, false),
            (2.0, true),
            (0.2, false),
            (1.0, true),
            (2.0, false),
            (0.05, true),
            (1.0, false),
            (1.0, true),
        ];
        let found = segments(&recording(&parts), RATE, &Settings::default());
        // 짧은 숨은 이어지고, 딸깍 소리는 버려진다.
        assert_eq!(found.len(), 2, "{found:?}");
        let (start, end) = seconds(&found[0]);
        assert!(
            (start - 0.9).abs() < 0.05 && (end - 4.3).abs() < 0.05,
            "{start} ~ {end}"
        );
        let (start, end) = seconds(&found[1]);
        assert!(
            (start - 7.15).abs() < 0.05 && (end - 8.25).abs() < 0.05,
            "{start} ~ {end}"
        );
        assert!(segments(&[], RATE, &Settings::default()).is_empty());
        // 처음부터 끝까지 말이면 구간 하나다.
        let all = segments(&recording(&[(1.0, true)]), RATE, &Settings::default());
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], 0..16_000);
    }

    #[test]
    fn windows_break_at_silences() {
        let segments = [0..10, 20..30, 45..50, 60..100, 110..115];
        // 길이 35까지: 앞의 둘을 묶고, 긴 구간은 나누고, 나눈 나머지에 뒤의
        // 구간이 붙는다.
        assert_eq!(windows(&segments, 35), [0..30, 45..50, 60..95, 95..115]);
        let whole = windows(&segments, 1000);
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0], 0..115);
        assert!(windows(&[], 10).is_empty());
    }
}
