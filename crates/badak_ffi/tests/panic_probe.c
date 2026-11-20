/* 엔진 안에서 패닉이 났을 때 부른 쪽에 무엇이 돌아오는지 본다 (44장). */
#include <stdio.h>

int badak_test_panic(void);

int main(void) {
    int status = badak_test_panic();
    printf("돌아온 값: %d\n", status);
    return 0;
}
