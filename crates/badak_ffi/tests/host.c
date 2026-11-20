/* C에서 badak을 부른다 (44장).
 *
 * 쓰는 법: host <모델.tflite> <사진.jpg>
 *
 * 문의 약속들을 하나씩 확인한 뒤, 사진을 분류해 확률이 높은 다섯 부류를
 * 적는다. 확률은 비트 그대로 16진수로 적는다. 폰의 앱도 같은 줄을 적으므로
 * 두 출력을 글자 그대로 견줄 수 있다.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "badak.h"

#define CHECK(condition)                                              \
    do {                                                              \
        if (!(condition)) {                                           \
            fprintf(stderr, "%d번째 줄: %s\n", __LINE__, #condition); \
            return 1;                                                 \
        }                                                             \
    } while (0)

int main(int argc, char **argv) {
    char message[256];
    badak_classifier *classifier = NULL;
    CHECK(argc == 3);
    CHECK(sizeof(badak_score) == 8);
    CHECK(strlen(badak_version()) > 0);
    CHECK(badak_cpu_features(message, sizeof message) > 0);
    printf("BADAK cpu %s\n", message);

    /* 잘못된 값은 오류로 돌아온다. 죽지 않는다. */
    CHECK(badak_classifier_open(NULL, &classifier) == BADAK_INVALID_ARGUMENT);
    CHECK(badak_classifier_open(argv[1], NULL) == BADAK_INVALID_ARGUMENT);
    CHECK(badak_classifier_open("없는 파일", &classifier) == BADAK_IO);
    CHECK(classifier == NULL);
    CHECK(badak_last_error(message, sizeof message) > 0);
    CHECK(badak_classifier_open(argv[2], &classifier) == BADAK_MODEL);
    CHECK(badak_classifier_input_len(NULL) == 0);
    badak_classifier_close(NULL);

    CHECK(badak_classifier_open(argv[1], &classifier) == BADAK_OK);
    CHECK(badak_classifier_input_len(classifier) == 3 * 224 * 224);
    CHECK(badak_classifier_output_len(classifier) == 1000);

    FILE *file = fopen(argv[2], "rb");
    CHECK(file != NULL);
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    uint8_t *jpeg = malloc((size_t)size);
    CHECK(fread(jpeg, 1, (size_t)size, file) == (size_t)size);
    fclose(file);

    badak_score top[5];
    size_t written = 0;
    /* 사진이 아닌 바이트. */
    CHECK(badak_classifier_classify_jpeg(classifier, jpeg, 100, top, 5,
                                         &written) == BADAK_IMAGE);
    CHECK(badak_classifier_classify_jpeg(classifier, jpeg, (size_t)size, top, 5,
                                         &written) == BADAK_OK);
    CHECK(written == 5);
    for (size_t rank = 0; rank < written; rank++) {
        uint32_t bits;
        memcpy(&bits, &top[rank].probability, sizeof bits);
        printf("BADAK top %u %08x\n", top[rank].index, bits);
    }
    free(jpeg);
    badak_classifier_close(classifier);
    return 0;
}
