//! 기기에 내려받은 모델 파일을 두는 곳 (44장).
//!
//! 앱이 모델 파일을 새로 받았을 때 할 일은 셋이다. 받은 것이 보낸 그대로인지
//! 확인하고, 쓰던 모델을 망가뜨리지 않으면서 제자리에 놓고, 새 모델이 말썽을
//! 부리면 되돌린다. 파일을 **받는** 일은 여기에 없다. 그것은 앱이 운영체제의
//! 내려받기 기능으로 한다.
//!
//! 폴더는 이렇게 생겼다.
//!
//! ```text
//! <곳>/<이름>/<버전>/manifest.json   받은 명세 그대로
//! <곳>/<이름>/<버전>/<파일>          모델 파일
//! <곳>/<이름>/current              지금 쓰는 버전의 이름이 적힌 글 파일
//! <곳>/<이름>/previous             그 앞에 쓰던 버전 (있으면)
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::{json, sha256};

/// 이 엔진이 읽을 수 있는 모델 파일의 세대. 명세의 `requires`가 이보다 크면
/// 그 모델은 더 새로운 엔진을 위한 것이다.
pub const ENGINE_GENERATION: usize = 1;

// ANCHOR: manifest
/// 모델 파일 하나를 설명하는 명세. 서버가 모델 파일과 함께 내준다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// 모델의 이름. 폴더 이름으로 쓴다.
    pub name: String,
    /// 버전의 이름. 폴더 이름으로 쓴다.
    pub version: String,
    /// 모델 파일의 이름.
    pub file: String,
    /// 파일의 크기(바이트).
    pub size: u64,
    /// 파일의 SHA-256. 소문자 16진수 예순네 글자.
    pub sha256: String,
    /// 이 파일을 읽는 데 필요한 엔진의 세대.
    pub requires: usize,
}
// ANCHOR_END: manifest

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse {
        format: "모델 명세",
        offset: 0,
        message: message.into(),
    }
}

// ANCHOR: safe_name
/// 폴더나 파일의 이름으로 써도 되는 글인지 본다.
///
/// 명세는 네트워크에서 온다. `../../databases/app.db` 같은 이름을 그대로
/// 길에 붙이면 앱의 다른 파일을 덮어쓰게 된다. 글자, 숫자, 점, 줄표,
/// 밑줄만 받고, 점으로 시작하는 이름은 받지 않는다.
fn safe_name(what: &str, name: &str) -> Result<()> {
    let allowed = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_');
    if name.is_empty() || name.len() > 100 || name.starts_with('.') || !name.bytes().all(allowed) {
        return Err(invalid(format!("{what} {name:?}는 쓸 수 없는 이름입니다")));
    }
    Ok(())
}
// ANCHOR_END: safe_name

impl Manifest {
    /// JSON으로 적힌 명세를 읽는다.
    pub fn parse(text: &str) -> Result<Self> {
        let value = json::parse(text)?;
        let string = |key: &str| {
            value
                .get(key)
                .and_then(json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("{key:?}가 없거나 글이 아닙니다")))
        };
        let number = |key: &str| {
            value
                .get(key)
                .and_then(json::Value::as_usize)
                .ok_or_else(|| invalid(format!("{key:?}가 없거나 수가 아닙니다")))
        };
        let manifest = Self {
            name: string("name")?,
            version: string("version")?,
            file: string("file")?,
            size: number("size")? as u64,
            sha256: string("sha256")?,
            requires: number("requires")?,
        };
        safe_name("이름", &manifest.name)?;
        safe_name("버전", &manifest.version)?;
        safe_name("파일", &manifest.file)?;
        let hex = |byte: u8| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
        if manifest.sha256.len() != 64 || !manifest.sha256.bytes().all(hex) {
            return Err(invalid("sha256은 소문자 16진수 예순네 글자여야 합니다"));
        }
        Ok(manifest)
    }

    /// 명세를 JSON으로 적는다.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\": {}, \"version\": {}, \"file\": {}, \"size\": {}, \"sha256\": {}, \"requires\": {}}}\n",
            json::quote(&self.name),
            json::quote(&self.version),
            json::quote(&self.file),
            self.size,
            json::quote(&self.sha256),
            self.requires
        )
    }

    /// 파일을 읽어 그 파일의 명세를 만든다. 모델을 내주는 쪽이 쓴다.
    pub fn describe(name: &str, version: &str, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("파일 이름을 읽을 수 없습니다"))?;
        let (hash, size) = sha256::digest_reader(File::open(path)?)?;
        let manifest = Self {
            name: name.to_owned(),
            version: version.to_owned(),
            file: file.to_owned(),
            size,
            sha256: sha256::hex(&hash),
            requires: ENGINE_GENERATION,
        };
        Self::parse(&manifest.to_json())
    }
}

/// 받은 파일을 거절한 까닭이 "파일이 명세와 다르다"인지. 앱은 이때 다시
/// 받아야 한다. 다른 오류(디스크가 찼다 등)와 구분하려고 둔다.
pub fn is_mismatch(error: &Error) -> bool {
    matches!(error, Error::Mismatch(_))
}

fn mismatch(message: String) -> Error {
    Error::Mismatch(message)
}

/// 글 파일을 "다 쓰였거나 아예 없거나"가 되게 쓴다.
fn write_atomically(path: &Path, content: &str) -> Result<()> {
    let partial = path.with_extension("partial");
    let mut file = File::create(&partial)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()?;
    fs::rename(&partial, path)?;
    sync_directory(path.parent().unwrap_or(Path::new(".")))
}

/// 폴더에 생긴 이름의 변화를 디스크에 내린다. 지원하지 않는 운영체제에서는
/// 아무 일도 하지 않는다.
fn sync_directory(directory: &Path) -> Result<()> {
    if cfg!(unix) {
        File::open(directory)?.sync_all()?;
    }
    Ok(())
}

/// 모델 파일들을 두는 곳.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// `root` 아래를 쓰는 곳. 폴더는 처음 놓을 때 만든다.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn pointer(&self, name: &str, which: &str) -> Result<Option<String>> {
        safe_name("이름", name)?;
        match fs::read_to_string(self.root.join(name).join(which)) {
            Ok(version) => {
                let version = version.trim().to_owned();
                safe_name("버전", &version)?;
                Ok(Some(version))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// 지금 쓰는 버전의 이름.
    pub fn current_version(&self, name: &str) -> Result<Option<String>> {
        self.pointer(name, "current")
    }

    /// 지금 쓰는 버전의 명세와 모델 파일의 길. 놓인 것이 없으면 `None`이다.
    ///
    /// 파일의 크기만 다시 본다. 해시는 놓을 때 확인했고, 열 때마다 다시
    /// 계산하면 시작이 그만큼 늦어진다.
    pub fn current(&self, name: &str) -> Result<Option<(Manifest, PathBuf)>> {
        let Some(version) = self.current_version(name)? else {
            return Ok(None);
        };
        let directory = self.root.join(name).join(&version);
        let manifest = Manifest::parse(&fs::read_to_string(directory.join("manifest.json"))?)?;
        let path = directory.join(&manifest.file);
        let size = fs::metadata(&path)?.len();
        if size != manifest.size {
            return Err(mismatch(format!(
                "{}의 크기가 {size}바이트입니다 (명세는 {})",
                manifest.file, manifest.size
            )));
        }
        Ok(Some((manifest, path)))
    }

    // ANCHOR: install
    /// 받은 파일을 확인하고 제자리에 놓은 뒤, 지금 쓰는 버전으로 삼는다.
    ///
    /// `downloaded`는 받아 둔 파일이다. 성공하면 그 파일은 곳 안으로
    /// 옮겨진다(같은 저장 장치에 있어야 한다). 실패하면 쓰던 버전은 그대로다.
    pub fn install(&self, manifest: &Manifest, downloaded: impl AsRef<Path>) -> Result<PathBuf> {
        let downloaded = downloaded.as_ref();
        // 1. 이 엔진이 읽을 수 있는 파일인가.
        if manifest.requires > ENGINE_GENERATION {
            return Err(Error::Unsupported(format!(
                "{} {}버전은 엔진 {}세대가 필요합니다 (이 엔진은 {ENGINE_GENERATION}세대)",
                manifest.name, manifest.version, manifest.requires
            )));
        }
        // 2. 받은 것이 보낸 그대로인가. 크기를 먼저 본다. 끊긴 내려받기는
        //    여기에서 걸리고, 해시를 계산할 필요가 없다.
        let size = fs::metadata(downloaded)?.len();
        if size != manifest.size {
            return Err(mismatch(format!(
                "크기가 {size}바이트입니다 (명세는 {})",
                manifest.size
            )));
        }
        let (hash, _) = sha256::digest_reader(File::open(downloaded)?)?;
        if sha256::hex(&hash) != manifest.sha256 {
            return Err(mismatch(format!(
                "해시가 {}입니다 (명세는 {})",
                sha256::hex(&hash),
                manifest.sha256
            )));
        }
        // 3. 옆 폴더에 다 차려 놓고, 이름을 한 번에 바꾼다.
        let home = self.root.join(&manifest.name);
        let target = home.join(&manifest.version);
        let staging = home.join(format!("{}.partial", manifest.version));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        fs::rename(downloaded, staging.join(&manifest.file))?;
        File::open(staging.join(&manifest.file))?.sync_all()?;
        let mut note = File::create(staging.join("manifest.json"))?;
        note.write_all(manifest.to_json().as_bytes())?;
        note.sync_all()?;
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        fs::rename(&staging, &target)?;
        sync_directory(&home)?;
        // 4. 지금 쓰는 버전을 가리키는 글을 바꾼다. 여기까지 오기 전에 앱이
        //    죽으면 쓰던 버전이 그대로 쓰인다.
        let before = self.current_version(&manifest.name)?;
        if let Some(before) = before.filter(|before| *before != manifest.version) {
            write_atomically(&home.join("previous"), &before)?;
        }
        write_atomically(&home.join("current"), &manifest.version)?;
        Ok(target.join(&manifest.file))
    }
    // ANCHOR_END: install

    // ANCHOR: rollback
    /// 앞에 쓰던 버전으로 되돌린다. 되돌릴 버전이 없으면 `false`다.
    pub fn rollback(&self, name: &str) -> Result<bool> {
        let Some(previous) = self.pointer(name, "previous")? else {
            return Ok(false);
        };
        let home = self.root.join(name);
        if !home.join(&previous).join("manifest.json").exists() {
            return Ok(false);
        }
        write_atomically(&home.join("current"), &previous)?;
        fs::remove_file(home.join("previous"))?;
        sync_directory(&home)?;
        Ok(true)
    }
    // ANCHOR_END: rollback

    /// 지금 쓰는 버전과 앞에 쓰던 버전만 남기고 지운다. 하다 만 폴더도 지운다.
    /// 지운 폴더의 수를 돌려준다.
    pub fn prune(&self, name: &str) -> Result<usize> {
        let keep = [
            self.pointer(name, "current")?,
            self.pointer(name, "previous")?,
        ];
        let mut removed = 0;
        let home = self.root.join(name);
        if !home.exists() {
            return Ok(0);
        }
        for entry in fs::read_dir(&home)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let version = entry.file_name().to_string_lossy().into_owned();
            if !keep.iter().flatten().any(|kept| *kept == version) {
                fs::remove_dir_all(entry.path())?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("badak_store_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    /// 내용이 `seed`로 정해지는 파일을 만들고 그 명세를 돌려준다.
    fn download(directory: &Path, version: &str, seed: u8, len: usize) -> (Manifest, PathBuf) {
        let made = directory.join("made");
        fs::create_dir_all(&made).unwrap();
        let source = made.join("model.bin");
        let bytes: Vec<u8> = (0..len)
            .map(|index| (index as u8).wrapping_mul(seed))
            .collect();
        fs::write(&source, &bytes).unwrap();
        let manifest = Manifest::describe("toy", version, &source).unwrap();
        let arrived = directory.join(format!("arrived_{version}"));
        fs::rename(&source, &arrived).unwrap();
        (manifest, arrived)
    }

    #[test]
    fn installs_updates_and_rolls_back() {
        let directory = scratch("happy");
        let store = Store::new(directory.join("models"));
        assert!(store.current("toy").unwrap().is_none());

        let (first, arrived) = download(&directory, "1", 3, 5000);
        let path = store.install(&first, &arrived).unwrap();
        assert!(!arrived.exists(), "받은 파일은 옮겨진다");
        assert_eq!(
            store.current("toy").unwrap().unwrap(),
            (first.clone(), path.clone())
        );
        assert!(!store.rollback("toy").unwrap(), "되돌릴 버전이 없다");

        let (second, arrived) = download(&directory, "2", 5, 7000);
        store.install(&second, &arrived).unwrap();
        assert_eq!(store.current_version("toy").unwrap().as_deref(), Some("2"));
        assert!(path.exists(), "앞의 버전은 남아 있다");

        let (third, arrived) = download(&directory, "3", 7, 100);
        store.install(&third, &arrived).unwrap();
        assert_eq!(store.prune("toy").unwrap(), 1, "1버전만 지운다");
        assert!(!path.exists());

        assert!(store.rollback("toy").unwrap());
        assert_eq!(store.current("toy").unwrap().unwrap().0, second);
        assert!(!store.rollback("toy").unwrap(), "한 번만 되돌린다");
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_bad_download_leaves_the_old_model_in_place() {
        let directory = scratch("bad");
        let store = Store::new(directory.join("models"));
        let (first, arrived) = download(&directory, "1", 3, 5000);
        store.install(&first, &arrived).unwrap();

        // 끊긴 내려받기.
        let (second, arrived) = download(&directory, "2", 5, 7000);
        let bytes = fs::read(&arrived).unwrap();
        fs::write(&arrived, &bytes[..6999]).unwrap();
        let error = store.install(&second, &arrived).unwrap_err();
        assert!(is_mismatch(&error), "{error}");
        // 비트 하나가 뒤집힌 파일.
        let mut flipped = bytes.clone();
        flipped[1234] ^= 0x10;
        fs::write(&arrived, &flipped).unwrap();
        let error = store.install(&second, &arrived).unwrap_err();
        assert!(
            is_mismatch(&error) && error.to_string().contains("해시"),
            "{error}"
        );
        // 더 새로운 엔진을 위한 파일.
        fs::write(&arrived, &bytes).unwrap();
        let newer = Manifest {
            requires: ENGINE_GENERATION + 1,
            ..second.clone()
        };
        let error = store.install(&newer, &arrived).unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(!is_mismatch(&error));

        assert_eq!(store.current("toy").unwrap().unwrap().0, first);
        assert!(arrived.exists(), "거절한 파일은 건드리지 않는다");
        // 놓다가 앱이 죽은 자리(하다 만 폴더)가 있어도 다시 놓을 수 있다.
        fs::create_dir_all(directory.join("models/toy/2.partial/junk")).unwrap();
        store.install(&second, &arrived).unwrap();
        assert_eq!(store.current("toy").unwrap().unwrap().0, second);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn names_from_the_network_cannot_leave_the_store() {
        let good = "{\"name\": \"toy\", \"version\": \"1\", \"file\": \"m.bin\", \"size\": 1, \
                    \"sha256\": \"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\", \
                    \"requires\": 1}";
        assert!(Manifest::parse(good).is_ok());
        for (from, to) in [
            ("\"toy\"", "\"../toy\""),
            ("\"toy\"", "\"\""),
            ("\"1\"", "\"..\""),
            ("\"m.bin\"", "\"a/b\""),
            ("\"m.bin\"", "\".hidden\""),
            ("e3b0", "E3B0"),
            ("e3b0", "e3b"),
            ("\"size\": 1", "\"size\": -1"),
        ] {
            let bad = good.replacen(from, to, 1);
            assert_ne!(bad, good);
            assert!(Manifest::parse(&bad).is_err(), "{bad}");
        }
        let store = Store::new("unused");
        assert!(store.current("../x").is_err());
    }
}
