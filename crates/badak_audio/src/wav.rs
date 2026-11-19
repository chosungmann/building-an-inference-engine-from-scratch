//! WAV 파일 읽기.
//!
//! WAV는 압축하지 않은 소리를 담는 가장 단순한 형식이다. 파일은 이름이 붙은
//! 덩어리(chunk)들이고, `fmt ` 덩어리가 형식을, `data` 덩어리가 표본들을
//! 담는다. 표본은 일정한 간격으로 잰 공기의 압력이다.

use std::path::Path;

use crate::{Error, Result};

// ANCHOR: wav
/// 읽어 들인 소리.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav {
    /// 1초에 표본이 몇 개인가.
    pub sample_rate: u32,
    /// 표본들. −1과 1 사이의 값이다. 채널이 여럿이면 평균을 내어 하나로
    /// 만든 것이다.
    pub samples: Vec<f32>,
}
// ANCHOR_END: wav

fn invalid(message: impl Into<String>) -> Error {
    Error::Format(message.into())
}

impl Wav {
    /// 파일을 읽는다.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read(path)?)
    }

    // ANCHOR: parse
    /// 바이트를 해석한다. 16비트 정수 표본(PCM)만 받는다.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let number = |at: usize, len: usize| -> Result<u32> {
            let field = bytes
                .get(at..at + len)
                .ok_or_else(|| invalid("파일이 중간에 끝났습니다"))?;
            // 리틀 엔디언이다. 낮은 자리가 앞에 있다.
            Ok(field
                .iter()
                .rev()
                .fold(0, |value, &byte| value << 8 | u32::from(byte)))
        };
        if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
            return Err(invalid("RIFF/WAVE로 시작하지 않습니다"));
        }
        let (mut format, mut data) = (None, None);
        let mut at = 12;
        // 덩어리마다 이름 넷, 길이 넷, 내용. 모르는 덩어리는 건너뛴다.
        while at + 8 <= bytes.len() {
            let len = number(at + 4, 4)? as usize;
            let body = at + 8;
            let end = body.checked_add(len).filter(|&end| end <= bytes.len());
            match (&bytes[at..at + 4], end) {
                (b"fmt ", Some(_)) if len >= 16 => {
                    format = Some((
                        number(body, 2)?,
                        number(body + 2, 2)?,
                        number(body + 4, 4)?,
                        number(body + 14, 2)?,
                    ));
                }
                (b"data", Some(end)) => data = Some(&bytes[body..end]),
                // 길이가 파일을 넘는 `data`는 남은 만큼만 읽는다. 녹음하다
                // 끊긴 파일이 그렇다.
                (b"data", None) => data = Some(&bytes[body..]),
                (_, None) => break,
                _ => {}
            }
            // 덩어리는 짝수 바이트에서 시작한다.
            at = body + len + len % 2;
        }
        let (kind, channels, sample_rate, bits) =
            format.ok_or_else(|| invalid("fmt 덩어리가 없습니다"))?;
        let data = data.ok_or_else(|| invalid("data 덩어리가 없습니다"))?;
        if kind != 1 || bits != 16 {
            return Err(Error::Unsupported(format!(
                "형식 {kind}번, 표본 {bits}비트 (16비트 PCM만 읽습니다)"
            )));
        }
        if channels == 0 || sample_rate == 0 {
            return Err(invalid("채널 수나 표본 속도가 0입니다"));
        }
        let channels = channels as usize;
        let samples = data
            .chunks_exact(2 * channels)
            .map(|frame| {
                let (pairs, _) = frame.as_chunks::<2>();
                let sum: i32 = pairs
                    .iter()
                    .map(|&pair| i32::from(i16::from_le_bytes(pair)))
                    .sum();
                sum as f32 / (32768.0 * channels as f32)
            })
            .collect();
        Ok(Self {
            sample_rate,
            samples,
        })
    }
    // ANCHOR_END: parse

    /// 길이(초).
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    // ANCHOR: write
    /// WAV 파일의 바이트들로 바꾼다 (37장). 채널 하나, 16비트 정수 표본이다.
    /// −1과 1 밖의 값은 끝값으로 자른다.
    pub fn to_bytes(&self) -> Vec<u8> {
        let data = (self.samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        // 형식 덩어리: 길이 16, PCM(1), 채널 1, 표본 속도, 1초의 바이트 수,
        // 표본 하나의 바이트 수(2), 표본의 비트 수(16).
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(self.sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for &sample in &self.samples {
            let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// 파일로 쓴다.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes())?)
    }
    // ANCHOR_END: write
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_files_read_back_within_one_step() {
        let wav = Wav {
            sample_rate: 24_000,
            samples: vec![0.0, 0.5, -0.25, 1.5, -2.0],
        };
        let bytes = wav.to_bytes();
        assert_eq!(bytes.len(), 44 + 10);
        let back = Wav::parse(&bytes).unwrap();
        assert_eq!(back.sample_rate, 24_000);
        // 16비트로 적었으므로 한 단계(1/32768)쯤의 차이가 난다. 범위 밖은 잘린다.
        for (a, b) in back.samples.iter().zip([0.0, 0.5, -0.25, 1.0, -1.0]) {
            assert!((a - b).abs() < 1e-4, "{a} {b}");
        }
    }

    /// 표본들로 WAV 파일의 바이트를 짓는다.
    fn file(channels: u16, rate: u32, bits: u16, samples: &[i16], extra: bool) -> Vec<u8> {
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let mut bytes = b"RIFF\0\0\0\0WAVE".to_vec();
        if extra {
            // 모르는 덩어리. 길이가 홀수라 채우는 바이트가 하나 붙는다.
            bytes.extend(b"LIST\x03\0\0\0abc\0");
        }
        bytes.extend(b"fmt \x10\0\0\0\x01\0");
        bytes.extend(channels.to_le_bytes());
        bytes.extend(rate.to_le_bytes());
        bytes.extend((rate * u32::from(channels) * 2).to_le_bytes());
        bytes.extend((channels * 2).to_le_bytes());
        bytes.extend(bits.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn reads_mono_and_mixes_stereo() {
        let mono = Wav::parse(&file(1, 16000, 16, &[0, 16384, -32768, 32767], true)).unwrap();
        assert_eq!(mono.sample_rate, 16000);
        assert_eq!(mono.samples, [0.0, 0.5, -1.0, 32767.0 / 32768.0]);
        assert_eq!(mono.seconds(), 4.0 / 16000.0);
        // 두 채널의 평균.
        let stereo = Wav::parse(&file(2, 8000, 16, &[16384, 0, -16384, -16384], false)).unwrap();
        assert_eq!(stereo.samples, [0.25, -0.5]);
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        let good = file(1, 16000, 16, &[1, 2, 3], false);
        assert!(Wav::parse(b"RIFF").is_err());
        assert!(Wav::parse(&good[..20]).is_err());
        let mut other = good.clone();
        other[8..12].copy_from_slice(b"AVI ");
        assert!(Wav::parse(&other).is_err());
        assert!(matches!(
            Wav::parse(&file(1, 16000, 8, &[1], false)),
            Err(Error::Unsupported(_))
        ));
        assert!(Wav::parse(&file(0, 16000, 16, &[1], false)).is_err());
        // 어디서 잘라도 죽지 않는다.
        for len in 0..good.len() {
            let _ = Wav::parse(&good[..len]);
        }
        // data의 길이가 파일보다 길면 있는 만큼만 읽는다.
        let mut long = good.clone();
        let at = long.len() - 6 - 4;
        long[at..at + 4].copy_from_slice(&1000_u32.to_le_bytes());
        assert_eq!(Wav::parse(&long).unwrap().samples.len(), 3);
    }
}
