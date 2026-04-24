import argparse
import sys

import numpy as np
from PIL import Image


def to_grayscale(img: Image.Image) -> Image.Image:
    img_arr = np.array(img)

    result = (
        0.299 * img_arr[::, ::, 0]
        + 0.587 * img_arr[::, ::, 1]
        + 0.114 * img_arr[::, ::, 2]
    )

    result = result.astype(np.uint8)
    result_img = Image.fromarray(result)
    return result_img


def apply_edge_padding(
    img_arr: np.ndarray, padded_img: np.ndarray, padding_size: int, w: int, h: int
) -> None:
    # Fill up and down edges
    for i in range(padding_size):
        padded_img[i, padding_size : padding_size + w] = img_arr[0, :]
        padded_img[padding_size + h + i, padding_size : padding_size + w] = img_arr[
            -1, :
        ]

    # Fill left and right edges
    for j in range(padding_size):
        padded_img[:, j] = padded_img[:, padding_size]
        padded_img[:, padding_size + w + j] = padded_img[:, padding_size + w - 1]


def apply_reflect_padding(
    img_arr: np.ndarray, padded_img: np.ndarray, padding_size: int, w: int, h: int
) -> None:
    # Mirror reflection up and down
    for i in range(padding_size):
        padded_img[padding_size - 1 - i, padding_size : padding_size + w] = img_arr[
            i + 1, :
        ]
        padded_img[padding_size + h + i, padding_size : padding_size + w] = img_arr[
            -2 - i, :
        ]

    # Mirror reflection left and right
    for j in range(padding_size):
        padded_img[:, padding_size - 1 - j] = padded_img[:, padding_size + 1 + j]
        padded_img[:, padding_size + w + j] = padded_img[:, padding_size + w - 2 - j]


def get_shapes(
    img_shape: tuple[int, ...], padding_size: int, padding_mode: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    h, w = img_shape[:2]
    is_rgb = len(img_shape) == 3

    if padding_mode == "valid":
        res_h, res_w = h - 2 * padding_size, w - 2 * padding_size
        pad_h, pad_w = h, w
    else:
        res_h, res_w = h, w
        pad_h, pad_w = h + 2 * padding_size, w + 2 * padding_size

    return (
        ((res_h, res_w, 3), (pad_h, pad_w, 3))
        if (is_rgb)
        else ((res_h, res_w), (pad_h, pad_w))
    )


def padding_convert(
    img_arr: np.ndarray, kernel: np.ndarray, padding_mode: str
) -> tuple[np.ndarray, np.ndarray]:

    padding_size = (kernel.shape[0] - 1) // 2
    h, w = img_arr.shape[:2]

    result_shape, padded_shape = get_shapes(img_arr.shape, padding_size, padding_mode)

    result = np.zeros(result_shape)

    if padding_mode == "valid":
        return (result, img_arr)

    padded_img = np.zeros(padded_shape)
    padded_img[padding_size : padding_size + h, padding_size : padding_size + w] = (
        img_arr
    )

    if padding_mode == "edge":
        apply_edge_padding(img_arr, padded_img, padding_size, w, h)

    elif padding_mode == "reflect":
        apply_reflect_padding(img_arr, padded_img, padding_size, w, h)

    return (result, padded_img)


def conv(
    img: Image.Image,
    kernel: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    padding_mode: str = "valid",
    gray: bool = False,
) -> Image.Image:
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel shape must be a square.")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel shape must be odd.")

    if gray:
        img = to_grayscale(img)
    img_arr = np.array(img)

    result, padded_img = padding_convert(img_arr, kernel, padding_mode)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            img_slice = padded_img[i : i + result.shape[0], j : j + result.shape[1]]

            result += img_slice * kernel[i, j]

    result = np.clip(result, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)
    return result_img


FILTERS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "blur": np.array(
        [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    ),
    "gaussian_blur": np.array(
        [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]]
    ),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Applying filters to an image using convolution.")

    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output file")

    parser.add_argument(
        "--filter",
        choices=FILTERS.keys(),
        default="identity",
        help="Name of using filter",
    )

    parser.add_argument(
        "--padding",
        choices=["valid", "zero", "edge", "reflect"],
        default="zero",
        help="Manual edge handling mode: 'valid', 'zero', 'edge', 'reflect'",
    )

    parser.add_argument(
        "--gray",
        action="store_true",
        help="Process image in grayscale mode (instead of RGB)",
    )
    args = parser.parse_args()

    try:
        img = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        print(f"Error: file {args.input} wasn't found.")
        sys.exit(1)

    selected_kernel = FILTERS[args.filter]

    result = conv(
        img, kernel=selected_kernel, padding_mode=args.padding, gray=args.gray
    )
    result.save(args.output)
