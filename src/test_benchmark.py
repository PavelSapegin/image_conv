import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter
from pytest_benchmark.fixture import BenchmarkFixture

from main import FILTERS, conv

KERNELS_TO_TEST = ["sharpen", "blur", "edge"]
PADDINGS_TO_TEST = ["zero", "edge", "reflect"]

CV2_BORDER_MODES = {
    "zero": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
}


# Generating random images for benchmarking
@pytest.fixture(
    params=[128, 512, 1024, 4096], ids=["128x128", "512x512", "1024x1024", "4096x4096"]
)
def image_data(request: pytest.FixtureRequest) -> Image.Image:
    size = request.param
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.mark.benchmark(group="convolution")
@pytest.mark.parametrize("kernel_name", KERNELS_TO_TEST)
@pytest.mark.parametrize("padding_mode", PADDINGS_TO_TEST)
@pytest.mark.parametrize("gray", [False, True])
def test_custom_conv(
    benchmark: BenchmarkFixture,
    image_data: Image.Image,
    kernel_name: str,
    padding_mode: str,
    gray: bool,
) -> None:

    kernel_np = FILTERS[kernel_name]

    benchmark(conv, image_data, kernel=kernel_np, padding_mode=padding_mode, gray=gray)


@pytest.mark.benchmark(group="convolution")
@pytest.mark.parametrize("kernel_name", KERNELS_TO_TEST)
@pytest.mark.parametrize("padding_mode", PADDINGS_TO_TEST)
@pytest.mark.parametrize("gray", [False, True])
def test_opencv_conv(
    benchmark: BenchmarkFixture,
    image_data: Image.Image,
    kernel_name: str,
    padding_mode: str,
    gray: bool,
) -> None:

    kernel_np = FILTERS[kernel_name]
    cv2_border = CV2_BORDER_MODES[padding_mode]

    def run_cv2(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if gray:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        res_arr = cv2.filter2D(arr, -1, kernel_np, borderType=cv2_border)
        return Image.fromarray(res_arr)

    benchmark(run_cv2, image_data)


@pytest.mark.benchmark(group="convolution")
@pytest.mark.parametrize("kernel_name", KERNELS_TO_TEST)
@pytest.mark.parametrize("padding_mode", PADDINGS_TO_TEST)
@pytest.mark.parametrize("gray", [False, True])
def test_pillow_conv(
    benchmark: BenchmarkFixture,
    image_data: Image.Image,
    kernel_name: str,
    padding_mode: str,
    gray: bool,
) -> None:

    kernel_np = FILTERS[kernel_name]
    kernel_flat = kernel_np.flatten().tolist()
    kernel_size = (kernel_np.shape[1], kernel_np.shape[0])
    pillow_kernel = ImageFilter.Kernel(
        size=kernel_size,
        kernel=kernel_flat,
        scale=1,
        offset=0,
    )

    def run_pillow(img: Image.Image) -> Image.Image:
        if gray:
            img = img.convert("L")
        return img.filter(pillow_kernel)

    benchmark(run_pillow, image_data)
