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


def padding_convert(
    img_arr: np.ndarray, kernel: np.ndarray, padding_mode: str
) -> tuple[np.ndarray, np.ndarray]:

    padding_size = (kernel.shape[0] - 1) // 2
    h, w = img_arr.shape
    if padding_mode == "valid":
        padded_img = img_arr
        result = np.zeros(
            (
                h - (kernel.shape[0] - 1),
                w - (kernel.shape[1] - 1),
            )
        )
    else:
        padded_img = np.zeros((h + 2 * padding_size, w + 2 * padding_size))
        padded_img[padding_size : padding_size + h, padding_size : padding_size + w] = (
            img_arr
        )

        if padding_mode == "zero":
            pass

        elif padding_mode == "edge":
            # Fill up and down edges
            for i in range(padding_size):
                padded_img[i, padding_size : padding_size + w] = img_arr[0, :]
                padded_img[padding_size + h + i, padding_size : padding_size + w] = (
                    img_arr[-1, :]
                )

            # Fill left and right edges
            for j in range(padding_size):
                padded_img[:, j] = padded_img[:, padding_size]
                padded_img[:, padding_size + w + j] = padded_img[
                    :, padding_size + w - 1
                ]

        elif padding_mode == "reflect":
            # Mirror reflection up and down
            for i in range(padding_size):
                padded_img[padding_size - 1 - i, padding_size : padding_size + w] = (
                    img_arr[i + 1, :]
                )
                padded_img[padding_size + h + i, padding_size : padding_size + w] = (
                    img_arr[-2 - i, :]
                )

            # Mirror reflection left and right
            for j in range(padding_size):
                padded_img[:, padding_size - 1 - j] = padded_img[
                    :, padding_size + 1 + j
                ]
                padded_img[:, padding_size + w + j] = padded_img[
                    :, padding_size + w - 2 - j
                ]

        result = np.zeros((h, w))

    return (result, padded_img)


def conv(
    img: Image.Image,
    kernel: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    padding_mode: str = "valid",
) -> Image.Image:
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel shape must be a square.")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel shape must be odd.")

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

    args = parser.parse_args()

    try:
        img = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        print(f"Error: file {args.input} wasn't found.")
        sys.exit(1)

    selected_kernel = FILTERS[args.filter]

    result = conv(img, kernel=selected_kernel, padding_mode=args.padding)
    result.save(args.output)
