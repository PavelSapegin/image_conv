import argparse

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


def conv(
    img: Image.Image,
    kernel: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    padding: bool = True,
) -> Image.Image:
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel shape must be a square.")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel shape must be odd.")

    img = to_grayscale(img)
    img_arr = np.array(img)

    if padding:
        padding_size = (kernel.shape[0] - 1) // 2

        zero_h = np.zeros((img_arr.shape[0], padding_size))
        zero_w = np.zeros((padding_size, img_arr.shape[1] + 2 * padding_size))
        padded_img = np.hstack((zero_h, img_arr, zero_h))
        padded_img = np.vstack((zero_w, padded_img, zero_w))
        result = np.zeros((img_arr.shape[0], img_arr.shape[1]))

    else:
        padded_img = img_arr
        result = np.zeros(
            (
                img_arr.shape[0] - (kernel.shape[0] - 1),
                img_arr.shape[1] - (kernel.shape[1] - 1),
            )
        )

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

    parser.add_argument("--padding", action="store_true", help="Use padding")

    args = parser.parse_args()

    try:
        img = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        print(f"Error: file {args.input} wasn't found.")

    selected_kernel = FILTERS[args.filter]

    result = conv(img, kernel=selected_kernel, padding=args.padding)
    result.save(args.output)
