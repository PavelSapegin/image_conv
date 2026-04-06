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

    mode = "RGB" if result_img.mode == "RGB" else "L"
    golden_img = Image.open(golden_path).convert(mode)

    result_arr = np.array(result_img)
    golden_arr = np.array(golden_img)

    np.testing.assert_array_equal(
        result_arr,
        golden_arr,
        err_msg=f"The result does not match the reference file {golden_filename}!",
    )


# Parameters (Golden file, kernel, padding)
KERNELS_TO_TEST = [
    ("identity_no_padding_gray.png", FILTERS["identity"], "valid", True),
    ("edge_detect_padding_gray.png", FILTERS["edge"], "zero", True),
    ("blur_edge_padding_gray.png", FILTERS["blur"], "edge", True),
    ("sharpen_reflect_padding_gray.png", FILTERS["sharpen"], "reflect", True),
    
    ("identity_no_padding_color.png", FILTERS["identity"], "valid", False),
    ("edge_detect_padding_color.png", FILTERS["edge"], "zero", False),
    ("blur_edge_padding_color.png", FILTERS["blur"], "edge", False),
    ("sharpen_reflect_padding_color.png", FILTERS["sharpen"], "reflect", False),
]


@pytest.mark.parametrize("golden_file, kernel, padding_mode, gray", KERNELS_TO_TEST)
def test_conv_with_various_kernels(
    input_image: Image.Image,
    golden_file: str,
    kernel: np.ndarray,
    padding_mode: str,
    gray: bool,
) -> None:
    """
    Test function conv on some kernels (blur, sharpen, edge detection)
    comparing result with golden .png files.
    """
    result = conv(input_image, kernel=kernel, padding_mode=padding_mode, gray=gray)

    check_against_golden(result, golden_file)


def test_to_grayscale(input_image: Image.Image) -> None:
    """
    Test grayscale function
    """
    result = to_grayscale(input_image)
    check_against_golden(result, "grayscale_expected.png")
