"""7장: 같은 모델을 PyTorch로 실행하는 시간을 잰다.

실행: python reference/ch07_bench_pytorch.py

badak의 `ch07_bench` 예제와 같은 일을 같은 방법으로 잰다. 여러 번 실행해
가장 빠른 시간을 취한다.
"""

import time
import torch
from safetensors.numpy import load_file

from ch07_train_mnist import DATA, OUT, Mlp, preprocess, read_idx


def best_of(repeats, f):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    model = Mlp()
    weights = load_file(OUT / "model.safetensors")
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    model.eval()
    images = read_idx(DATA / "t10k-images-idx3-ubyte")
    x = torch.from_numpy(preprocess(images))
    count = len(x)
    singles = [x[i : i + 1] for i in range(1000)]

    def one_by_one():
        for single in singles:
            model(single)

    default_threads = torch.get_num_threads()
    for threads in (1, default_threads):
        torch.set_num_threads(threads)
        with torch.no_grad():
            batch = best_of(20, lambda: model(x))
            single = best_of(5, one_by_one)
        print(f"PyTorch {torch.__version__}, 스레드 {threads}개")
        print(f"{f'순전파, {count}장을 한 번에':<34} {batch * 1e3:>10.3f} ms")
        print(f"{'  1초에 처리하는 사진':<34} {count / batch:>10.1f} 장")
        print(f"{'순전파, 한 장씩 (장당)':<34} {single / 1000 * 1e6:>10.3f} µs")
        print()


if __name__ == "__main__":
    main()
