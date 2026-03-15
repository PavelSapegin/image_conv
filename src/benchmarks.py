import timeit
from PIL import Image
import numpy as np
from main import to_grayscale, conv


img = Image.open("./images/breaking_bad.jpg").convert("RGB")

kernels = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "box_blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    "gaussian_blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
}

NUM_RUNS = 3


def benchmark(func, number):
    total_time = timeit.timeit(func, number=number)
    return (total_time / number) * 1000


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    print(f"\nImage size: {img.size}")
    print(f"Number of runs: {NUM_RUNS}\n")

    print("to_grayscale:")
    t_lum = benchmark(lambda: to_grayscale(img, "lum"), NUM_RUNS)
    print(f"  lum: {t_lum:.3f} ms")

    t_avg = benchmark(lambda: to_grayscale(img, "avg"), NUM_RUNS)
    print(f"  avg: {t_avg:.3f} ms")

    print("\nconv (padding=True, optimize=True):")
    for name, kernel in kernels.items():
        t = benchmark(
            lambda k=kernel: conv(img, kernel=k, padding=True, optimize=True), NUM_RUNS
        )
        print(f"  {name:15s}: {t:.3f} ms")

    print("\nconv (padding=True, optimize=False):")
    for name, kernel in kernels.items():
        t = benchmark(
            lambda k=kernel: conv(img, kernel=k, padding=True, optimize=False), NUM_RUNS
        )
        print(f"  {name:15s}: {t:.3f} ms")

    print("\nconv (padding=False, optimize=True):")
    for name, kernel in kernels.items():
        t = benchmark(
            lambda k=kernel: conv(img, kernel=k, padding=False, optimize=True), NUM_RUNS
        )
        print(f"  {name:15s}: {t:.3f} ms")

    print("\nconv (padding=False, optimize=False):")
    for name, kernel in kernels.items():
        t = benchmark(
            lambda k=kernel: conv(img, kernel=k, padding=False, optimize=False),
            NUM_RUNS,
        )
        print(f"  {name:15s}: {t:.3f} ms")
