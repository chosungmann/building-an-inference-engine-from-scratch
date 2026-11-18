// 25장: 문맥을 채운 뒤 토큰을 하나씩 넣는 시간을 llama.cpp로 잰다.
//
// 사용법: llama_context <model.gguf> <스레드 수> <GPU에 올릴 층의 수> <가장 긴 문맥>
//
// badak의 `ch24_context` 예제와 같은 일을 한다. 문맥을 0, 256, 512, ...
// 토큰으로 채워 두고, 토큰을 20개 하나씩 넣는다. 처음 넷은 준비 운동이고
// 나머지 열여섯의 중앙값을 낸다. 문맥은 256토큰씩 끊어 채운다. 토큰은 아무
// 번호나 쓴다. 속도는 토큰이 무엇인지와 상관이 없다.
//
// llama.cpp의 기본 설정을 그대로 쓴다(플래시 어텐션은 자동, KV 캐시는 16비트).
// 빌드: sh scripts/build_ch25_tools.sh

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llama.h"

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "사용법: %s <model.gguf> <스레드 수> <GPU에 올릴 층의 수> <가장 긴 문맥>\n", argv[0]);
        return 1;
    }
    const int n_threads = atoi(argv[2]);
    const int n_gpu_layers = atoi(argv[3]);
    const int longest = atoi(argv[4]);
    const int n_new = 20;
    const int n_warm_up = 4;

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        fprintf(stderr, "모델을 읽지 못했습니다: %s\n", argv[1]);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = (uint32_t) longest + n_new + 8;
    ctx_params.n_batch         = 256;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 아무 토큰이나. 특수 토큰이 아닌 범위에서 고른다.
    std::vector<llama_token> tokens(longest + n_new);
    unsigned long long state = 25;
    for (auto & token : tokens) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        token = (llama_token) ((state >> 33) % (unsigned long long) (n_vocab - 1000));
    }

    printf("%10s %16s %14s %14s\n", "문맥", "채우는 시간 (초)", "토큰 하나 (ms)", "토큰/초");
    std::vector<int> contexts = {0};
    for (int context = 256; context <= longest; context *= 2) {
        contexts.push_back(context);
    }
    int filled = 0;
    double fill_seconds = 0.0;
    for (int context : contexts) {
        auto start = std::chrono::steady_clock::now();
        while (filled < context) {
            const int count = std::min(256, context - filled);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + filled, count)) != 0) {
                fprintf(stderr, "llama_decode가 실패했습니다\n");
                return 1;
            }
            filled += count;
        }
        // GPU에 보낸 일이 끝나기를 기다린다.
        llama_synchronize(ctx);
        fill_seconds += seconds_since(start);

        std::vector<double> times;
        for (int i = 0; i < n_new; i++) {
            start = std::chrono::steady_clock::now();
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + context + i, 1)) != 0) {
                fprintf(stderr, "llama_decode가 실패했습니다\n");
                return 1;
            }
            llama_synchronize(ctx);
            if (i >= n_warm_up) {
                times.push_back(seconds_since(start) * 1e3);
            }
        }
        std::sort(times.begin(), times.end());
        const double median = (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
        printf("%10d %16.2f %14.2f %14.1f\n", context, fill_seconds, median, 1e3 / median);
        fflush(stdout);
        // 방금 넣은 토큰들을 캐시에서 뺀다.
        llama_memory_seq_rm(llama_get_memory(ctx), 0, context, -1);
        filled = context;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
