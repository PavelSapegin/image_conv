from PIL import Image
import numpy as np

img = Image.open("./breaking_bad.jpg").convert("RGB")


def to_grayscale_avr_mth(img: Image) -> Image:
    img_arr = np.array(img)

    result = np.mean(img_arr, axis=2).astype(np.uint8)

    result = Image.fromarray(result)
    return result


def to_grayscale_lum_mth(img: Image) -> Image:
    img_arr = np.array(img)

    result = (
        0.299 * img_arr[::, ::, 0]
        + 0.587 * img_arr[::, ::, 1]
        + 0.114 * img_arr[::, ::, 2]
    )
    result = result.astype(np.uint8)
    result = Image.fromarray(result)
    return result


def to_grayscale(img: Image, type: str = "lum") -> Image:
    if type == "lum":
        return to_grayscale_lum_mth(img)
    else:
        return to_grayscale_avr_mth(img)


def conv(
    img: Image,
    kernel: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    padding: bool = True,
) -> Image:
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
    result = Image.fromarray(result)
    return result


result = conv(img, padding=False)
result.save("output.jpg")
