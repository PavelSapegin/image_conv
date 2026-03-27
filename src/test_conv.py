import os

import numpy as np
import pytest
from PIL import Image

from main import conv, to_grayscale

# PATHS
# Dir with test_conv.py
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to origin test image
INPUT_IMAGE_PATH = os.path.join(TEST_DIR, "images", "corgy_test.png")

# Dir with golden files (.png format)
GOLDEN_DIR = os.path.join(TEST_DIR, "images", "golden_images")


@pytest.fixture
def input_image() -> Image.Image:
    """Fixture for loading test image"""
    if not os.path.exists(INPUT_IMAGE_PATH):
        pytest.fail(f"The file does't exist: {INPUT_IMAGE_PATH}")

    return Image.open(INPUT_IMAGE_PATH).convert("RGB")


def check_against_golden(result_img: Image.Image, golden_filename: str) -> None:
    """Load golden file and compare with result"""
    golden_path = os.path.join(GOLDEN_DIR, golden_filename)

    if not os.path.exists(golden_path):
        pytest.fail(f"Эталонный файл не найден: {golden_path}")

    golden_img = Image.open(golden_path).convert("L")

    result_arr = np.array(result_img)
    golden_arr = np.array(golden_img)

    np.testing.assert_array_equal(
        result_arr,
        golden_arr,
        err_msg=f"The result does not match the reference file {golden_filename}!",
    )


# Parameters (Golden file, kernel, padding)
KERNELS_TO_TEST = [
    ("identity_no_padding.png", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), False),
    (
        "edge_detect_padding.png",
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        True,
    ),
    (
        "blur_no_padding.png",
        np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]),
        False,
    ),
    ("sharpen_padding.png", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), True),
]


@pytest.mark.parametrize("golden_file, kernel, use_padding", KERNELS_TO_TEST)
def test_conv_with_various_kernels(
    input_image: Image.Image, golden_file: str, kernel: np.ndarray, use_padding: bool
) -> None:
    """
    Test function conv on some kernels (blur, sharpen, edge detection)
    comparing result with golden .png files.
    """
    result = conv(input_image, kernel=kernel, padding=use_padding)

    check_against_golden(result, golden_file)


def test_to_grayscale(input_image: Image.Image) -> None:
    """
    Test grayscale function
    """
    result = to_grayscale(input_image)
    check_against_golden(result, "grayscale_expected.png")
