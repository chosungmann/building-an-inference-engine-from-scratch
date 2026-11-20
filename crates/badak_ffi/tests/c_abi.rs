//! 문의 약속을 밖에서 확인한다 (44장).

use std::ffi::{CString, c_char};
use std::path::PathBuf;
use std::process::Command;

use badak_ffi::*;
use badak_graph::models_dir;

/// ImageNet의 부류 번호: 컵.
const CUP: u32 = 968;

fn last_error() -> String {
    let mut buffer = vec![0_u8; 512];
    // SAFETY: 쓸 수 있는 512바이트다.
    let length = unsafe { badak_last_error(buffer.as_mut_ptr().cast(), buffer.len()) };
    String::from_utf8(buffer[..length.min(511)].to_vec()).unwrap()
}

fn model_path() -> CString {
    let path = models_dir().join("vision_tflite/mobilenetv2.tflite");
    CString::new(path.to_str().unwrap()).unwrap()
}

#[test]
fn classifies_through_the_c_functions() {
    let jpeg = std::fs::read(models_dir().join("images/cup.jpg")).unwrap();
    let mut classifier = std::ptr::null_mut();
    // SAFETY: 약속대로 부른다.
    unsafe {
        assert_eq!(
            badak_classifier_open(model_path().as_ptr(), &mut classifier),
            0,
            "{}",
            last_error()
        );
        let mut top = [Score::default(); 3];
        let mut written = 0;
        let status = badak_classifier_classify_jpeg(
            classifier,
            jpeg.as_ptr(),
            jpeg.len(),
            top.as_mut_ptr(),
            top.len(),
            &mut written,
        );
        assert_eq!(status, 0, "{}", last_error());
        assert_eq!((written, top[0].index), (3, CUP));
        assert!(top[0].probability > 0.9 && top[1].probability < top[0].probability);

        // 숫자로 바꾼 입력을 직접 넣어도 같은 부류가 나온다.
        let image = badak_image::Image::decode_jpeg(&jpeg).unwrap();
        let input = badak_image::Preprocess::IMAGENET.apply(&image);
        let mut scores = vec![0.0_f32; badak_classifier_output_len(classifier)];
        let status = badak_classifier_run(
            classifier,
            input.as_ptr(),
            input.len(),
            scores.as_mut_ptr(),
            scores.len(),
        );
        assert_eq!(status, 0, "{}", last_error());
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        assert_eq!(best as u32, CUP);
        // 길이가 맞지 않으면 오류다. 읽거나 쓰지 않는다.
        let status = badak_classifier_run(classifier, input.as_ptr(), 10, scores.as_mut_ptr(), 10);
        assert_eq!(status, Status::InvalidArgument as i32);
        badak_classifier_close(classifier);
    }
}

#[cfg(unix)]
#[test]
fn opens_a_model_inside_a_bigger_file() {
    use std::io::Write;
    use std::os::fd::AsRawFd;

    // 앱 꾸러미를 흉내 낸다. 앞에 다른 것이 있고, 그 뒤에 모델이 있고, 또
    // 다른 것이 있다. 모델이 시작하는 자리는 페이지의 경계가 아니다.
    let model = std::fs::read(models_dir().join("vision_tflite/mobilenetv2.tflite")).unwrap();
    let path = std::env::temp_dir().join(format!("badak_ffi_bundle_{}", std::process::id()));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&vec![0xAA; 12_345]).unwrap();
    file.write_all(&model).unwrap();
    file.write_all(&[0xBB; 999]).unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mut classifier = std::ptr::null_mut();
    // SAFETY: 약속대로 부른다.
    unsafe {
        let status = badak_classifier_open_fd(
            file.as_raw_fd(),
            12_345,
            model.len() as i64,
            &mut classifier,
        );
        assert_eq!(status, 0, "{}", last_error());
        assert_eq!(badak_classifier_output_len(classifier), 1000);
        badak_classifier_close(classifier);
        // 구간이 틀리면 모델로 읽히지 않는다.
        let status = badak_classifier_open_fd(file.as_raw_fd(), 0, 12_345, &mut classifier);
        assert_eq!(status, Status::Model as i32);
        let status = badak_classifier_open_fd(file.as_raw_fd(), 12_345, i64::MAX, &mut classifier);
        assert_eq!(status, Status::Io as i32);
        let status = badak_classifier_open_fd(file.as_raw_fd(), -1, 10, &mut classifier);
        assert_eq!(status, Status::InvalidArgument as i32);
    }
    // 디스크립터는 빌려 썼을 뿐이다. 아직 열려 있다.
    assert!(file.metadata().is_ok());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_panic_does_not_cross_the_boundary() {
    assert_eq!(badak_test_panic(), Status::Panic as i32);
    assert!(
        last_error().contains("시험으로 낸 패닉"),
        "{}",
        last_error()
    );
    // 오류의 설명은 자리가 모자라면 글자의 경계에서 잘린다.
    let mut small = [0x7F_u8; 8];
    // SAFETY: 쓸 수 있는 8바이트다.
    let length = unsafe { badak_last_error(small.as_mut_ptr().cast::<c_char>(), small.len()) };
    assert!(length > small.len());
    let end = small.iter().position(|&byte| byte == 0).unwrap();
    assert!(std::str::from_utf8(&small[..end]).is_ok());
}

#[test]
fn installs_a_model_through_the_c_functions() {
    use badak_formats::store::Manifest;

    let directory = std::env::temp_dir().join(format!("badak_ffi_store_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let arrived = directory.join("arrived.bin");
    std::fs::write(&arrived, b"model bytes").unwrap();
    let manifest = Manifest::describe("toy", "7", &arrived).unwrap();
    let c = |text: &str| CString::new(text).unwrap();
    let store = c(directory.join("models").to_str().unwrap());
    let mut buffer = vec![0_u8; 1024];
    let mut length = 99;
    // SAFETY: 약속대로 부른다.
    unsafe {
        let wrong = c(&manifest.to_json().replace("\"size\": 11", "\"size\": 12"));
        let status = badak_model_install(
            store.as_ptr(),
            wrong.as_ptr(),
            c(arrived.to_str().unwrap()).as_ptr(),
        );
        assert_eq!(status, Status::Mismatch as i32, "{}", last_error());
        let status = badak_model_current(
            store.as_ptr(),
            c("toy").as_ptr(),
            buffer.as_mut_ptr().cast(),
            1024,
            &mut length,
        );
        assert_eq!((status, length), (0, 0), "아직 놓인 것이 없다");

        let good = c(&manifest.to_json());
        let status = badak_model_install(
            store.as_ptr(),
            good.as_ptr(),
            c(arrived.to_str().unwrap()).as_ptr(),
        );
        assert_eq!(status, 0, "{}", last_error());
        let status = badak_model_current(
            store.as_ptr(),
            c("toy").as_ptr(),
            buffer.as_mut_ptr().cast(),
            1024,
            &mut length,
        );
        assert_eq!(status, 0, "{}", last_error());
        let path = PathBuf::from(std::str::from_utf8(&buffer[..length]).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), b"model bytes");
        assert!(path.ends_with("toy/7/arrived.bin"));
        let mut rolled = 5;
        assert_eq!(
            badak_model_rollback(store.as_ptr(), c("toy").as_ptr(), &mut rolled),
            0
        );
        assert_eq!(rolled, 0);
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// `badak.h`가 실제 함수들과 맞는지 C 컴파일러에게 묻는다. 헤더를 넣은 C
/// 프로그램을 만들어 이 크레이트의 동적 라이브러리와 묶고 돌린다.
#[cfg(unix)]
#[test]
fn the_header_matches_the_library() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `cargo test`는 동적 라이브러리를 만들지 않는다. 여기에서 만든다.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let built = Command::new(cargo)
        .args(["build", "--quiet", "-p", "badak_ffi"])
        .current_dir(&manifest)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    // 시험의 실행 파일에서 위로 올라가며 라이브러리가 놓인 폴더를 찾는다.
    let name = if cfg!(target_os = "macos") {
        "libbadak_ffi.dylib"
    } else {
        "libbadak_ffi.so"
    };
    let executable = std::env::current_exe().unwrap();
    let library = executable
        .ancestors()
        .find(|directory| directory.join(name).exists())
        .unwrap_or_else(|| panic!("{name}을 찾지 못했습니다"))
        .to_owned();
    let program = library.join(format!("badak_ffi_host_{}", std::process::id()));
    let compiled = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-o"])
        .arg(&program)
        .arg(manifest.join("tests/host.c"))
        .arg("-I")
        .arg(manifest.join("include"))
        .arg("-L")
        .arg(&library)
        .arg("-lbadak_ffi")
        .arg(format!("-Wl,-rpath,{}", library.display()))
        .output()
        .expect("C 컴파일러 cc가 있어야 합니다");
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&program)
        .arg(models_dir().join("vision_tflite/mobilenetv2.tflite"))
        .arg(models_dir().join("images/cup.jpg"))
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&program);
    let printed = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "{printed}{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let top: Vec<&str> = printed
        .lines()
        .filter(|line| line.starts_with("BADAK top "))
        .collect();
    assert!(
        top[0].starts_with(&format!("BADAK top {CUP} ")),
        "{printed}"
    );
    assert_eq!(top.len(), 5);
}
