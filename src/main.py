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
        padding = (kernel.shape[0] - 1) // 2

        zero_h = np.zeros((img_arr.shape[0], padding))
        zero_w = np.zeros((padding, img_arr.shape[1] + 2 * padding))
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
            slice = padded_img[i : i + result.shape[0], j : j + result.shape[1]]

            result += slice * kernel[i, j]

    result = np.clip(result, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)
    return result_img


if __name__ == "__main__":
    img = Image.open("./images/corgy0.jpg").convert("RGB")
    result = conv(img, padding=False)
    result.save("output.jpg")
