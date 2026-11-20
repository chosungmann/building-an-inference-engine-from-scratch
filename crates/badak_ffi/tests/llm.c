/* C에서 badak의 언어 모델을 부른다 (47장).
 *
 * 쓰는 법: llm <모델 폴더> <질문> [토큰 수] [지시문]
 *
 * 엔진이 글 조각마다 on_piece를 부른다. 조각을 모아, 끝나면 한 줄로 적는다.
 * 줄의 뒤쪽은 답의 바이트를 16진수로 적은 것이다. 폰의 앱도 같은 줄을
 * 적으므로 두 출력을 글자 그대로 견줄 수 있다.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "badak.h"

struct answer {
    char text[8192];
    size_t length;
    int pieces;
};

/* 0을 돌려주면 계속, 아니면 그만. user는 우리가 넘긴 포인터 그대로다. */
static int32_t on_piece(void *user, const char *piece, size_t length) {
    struct answer *answer = user;
    if (answer->length + length >= sizeof answer->text) {
        return 1;
    }
    memcpy(answer->text + answer->length, piece, length);
    answer->length += length;
    answer->pieces++;
    return 0;
}

int main(int argc, char **argv) {
    char message[256];
    if (argc < 3) {
        fprintf(stderr, "쓰는 법: llm <모델 폴더> <질문> [토큰 수] [지시문]\n");
        return 2;
    }
    size_t limit = argc > 3 ? (size_t)atoi(argv[3]) : 32;
    /* 지시문을 주지 않으면 NULL을 넘긴다. */
    const char *system = argc > 4 ? argv[4] : NULL;
    badak_llm *llm = NULL;
    if (badak_llm_open(argv[1], 0, &llm) != BADAK_OK) {
        badak_last_error(message, sizeof message);
        fprintf(stderr, "열지 못했습니다: %s\n", message);
        return 1;
    }
    struct answer answer = {0};
    size_t written = 0;
    if (badak_llm_chat(llm, system, argv[2], limit, on_piece, &answer, &written) != BADAK_OK) {
        badak_last_error(message, sizeof message);
        fprintf(stderr, "쓰지 못했습니다: %s\n", message);
        return 1;
    }
    printf("BADAK llm %zu ", written);
    for (size_t index = 0; index < answer.length; index++) {
        printf("%02x", (unsigned char)answer.text[index]);
    }
    printf("\n%.*s\n", (int)answer.length, answer.text);
    badak_llm_close(llm);
    return answer.pieces > 0 ? 0 : 1;
}
