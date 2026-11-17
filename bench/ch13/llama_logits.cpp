// 13장: llama.cpp로 점수를 계산해 파일로 내보낸다.
//
// 사용법: llama_logits <model.gguf> <점수를 적을 파일> <이어 쓸 토큰 수> <default|exact> <토큰 번호>...
//
// 넷째 인자가 default이면 llama.cpp의 기본 설정을 그대로 쓴다. exact이면
// 플래시 어텐션을 끄고 KV 캐시를 f32로 둔다. 16비트 실수를 거치지 않는 설정이다.
//
// 토큰 번호들을 한꺼번에 넣어 마지막 토큰 다음의 점수를 파일에 f32로 적는다.
// 그다음 늘 점수가 가장 높은 토큰을 골라 글을 이어 쓰고, 뽑은 토큰 번호를
// 표준 출력에 적는다. badak의 `ch13_compare` 예제가 이 프로그램을 부른다.
//
// CPU만 쓰고 스레드는 하나만 쓴다. 속도가 아니라 값을 보려는 것이다.
// 빌드: sh scripts/build_ch13_tools.sh

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "사용법: %s <model.gguf> <출력 파일> <이어 쓸 토큰 수> <default|exact> <토큰 번호>...\n", argv[0]);
        return 1;
    }
    const int n_generate = atoi(argv[3]);
    const bool exact = strcmp(argv[4], "exact") == 0;
    std::vector<llama_token> tokens;
    for (int i = 5; i < argc; i++) {
        tokens.push_back(atoi(argv[i]));
    }

    // llama.cpp가 내는 안내 글을 끈다.
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        fprintf(stderr, "모델을 읽지 못했습니다: %s\n", argv[1]);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = (uint32_t) tokens.size() + n_generate + 8;
    ctx_params.n_batch         = ctx_params.n_ctx;
    ctx_params.n_threads       = 1;
    ctx_params.n_threads_batch = 1;
    if (exact) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.type_k          = GGML_TYPE_F32;
        ctx_params.type_v          = GGML_TYPE_F32;
    }
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 프롬프트 전체를 한 번에 넣는다. 점수는 마지막 토큰의 것만 나온다.
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) != 0) {
        fprintf(stderr, "llama_decode가 실패했습니다\n");
        return 1;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    FILE * out = fopen(argv[2], "wb");
    fwrite(logits, sizeof(float), n_vocab, out);
    fclose(out);

    // 늘 1위를 골라 이어 쓴다. 뽑은 토큰을 하나씩 넣는다.
    for (int i = 0; i < n_generate; i++) {
        llama_token next = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
        printf("%d ", next);
        if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
            fprintf(stderr, "llama_decode가 실패했습니다\n");
            return 1;
        }
        logits = llama_get_logits_ith(ctx, -1);
    }
    printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
