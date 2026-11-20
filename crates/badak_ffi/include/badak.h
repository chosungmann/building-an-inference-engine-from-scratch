/* badak의 C 함수들 (44장).
 *
 * 이 파일은 손으로 적었다. crates/badak_ffi/src/lib.rs 의 함수들과 같아야
 * 하고, crates/badak_ffi/tests/c_abi.rs 가 C 컴파일러로 그것을 확인한다.
 *
 * 약속:
 *   - 값을 돌려주는 함수는 BADAK_OK(0)나 오류의 종류를 돌려준다. 설명은
 *     같은 스레드에서 badak_last_error 로 읽는다.
 *   - 분류기는 badak_classifier_close 로만 지운다. 한 분류기를 두 스레드가
 *     함께 쓰면 안 된다.
 *   - 글은 0으로 끝나는 UTF-8이다.
 */
#ifndef BADAK_H
#define BADAK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    BADAK_OK = 0,
    BADAK_INVALID_ARGUMENT = 1, /* 빈 포인터, 맞지 않는 길이 */
    BADAK_IO = 2,               /* 파일을 읽지 못했다 */
    BADAK_MODEL = 3,            /* 모델 파일이 올바르지 않거나 돌리지 못한다 */
    BADAK_IMAGE = 4,            /* 사진 파일을 풀지 못했다 */
    BADAK_MISMATCH = 5,         /* 받은 모델 파일이 명세와 다르다 */
    BADAK_PANIC = 6             /* 엔진의 버그 */
};

/* 속을 보여 주지 않는 타입. 포인터로만 다룬다. */
typedef struct badak_classifier badak_classifier;

typedef struct {
    uint32_t index;    /* 부류의 번호 */
    float probability; /* 확률 */
} badak_score;

const char *badak_version(void);
size_t badak_last_error(char *buffer, size_t capacity);
size_t badak_cpu_features(char *buffer, size_t capacity);
int32_t badak_set_threads(size_t threads);

int32_t badak_classifier_open(const char *path, badak_classifier **out);
int32_t badak_classifier_open_fd(int fd, int64_t offset, int64_t length,
                                 badak_classifier **out);
size_t badak_classifier_input_len(const badak_classifier *classifier);
size_t badak_classifier_output_len(const badak_classifier *classifier);
int32_t badak_classifier_run(badak_classifier *classifier,
                             const float *input, size_t input_len,
                             float *output, size_t output_len);
int32_t badak_classifier_classify_jpeg(badak_classifier *classifier,
                                       const uint8_t *bytes, size_t len,
                                       badak_score *top, size_t capacity,
                                       size_t *written);
void badak_classifier_close(badak_classifier *classifier);

int32_t badak_model_install(const char *directory, const char *manifest,
                            const char *downloaded);
int32_t badak_model_current(const char *directory, const char *name,
                            char *buffer, size_t capacity, size_t *length);
int32_t badak_model_rollback(const char *directory, const char *name,
                             int32_t *rolled);

#ifdef __cplusplus
}
#endif

#endif /* BADAK_H */
