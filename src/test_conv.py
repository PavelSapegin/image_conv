import os

import numpy as np
import pytest
from PIL import Image

from main import FILTERS, conv, to_grayscale

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
        pytest.fail(f"Golden file didn't found: {golden_path}")

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
    ("identity_no_padding.png", FILTERS["identity"], "valid"),
    (
        "edge_detect_padding.png",
        FILTERS["edge"],
        "zero",
    ),
    (
        "blur_no_padding.png",
        FILTERS["blur"],
        "valid",
    ),
    ("sharpen_padding.png", FILTERS["sharpen"], "zero"),
    (
        "blur_edge_padding.png",
        FILTERS["blur"],
        "edge",
    ),
    (
        "sharpen_reflect_padding.png",
        FILTERS["sharpen"],
        "reflect",
    ),
]


@pytest.mark.parametrize("golden_file, kernel, padding_mode", KERNELS_TO_TEST)
def test_conv_with_various_kernels(
    input_image: Image.Image, golden_file: str, kernel: np.ndarray, padding_mode: str
) -> None:
    """
    Test function conv on some kernels (blur, sharpen, edge detection)
    comparing result with golden .png files.
    """
    result = conv(input_image, kernel=kernel, padding_mode=padding_mode)

    check_against_golden(result, golden_file)


def test_to_grayscale(input_image: Image.Image) -> None:
    """
    Test grayscale function
    """
    result = to_grayscale(input_image)
    check_against_golden(result, "grayscale_expected.png")
