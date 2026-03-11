from PIL import Image
import numpy as np

img = Image.open("./image.jpg").convert("RGB")


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


def conv(img: Image) -> Image:
    img = to_grayscale(img)
    img_arr = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    padding = (kernel.shape[0] - 1) // 2

    zero_h = np.zeros((img_arr.shape[0], padding))
    zero_w = np.zeros((padding, img_arr.shape[1] + 2 * padding))
    padded_img = np.hstack((zero_h, img_arr, zero_h))
    padded_img = np.vstack((zero_w, padded_img, zero_w))

    result = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = padded_img[i : i + kernel.shape[0], j : j + kernel.shape[0]]
            result[i][j] = np.sum(window * kernel)

    result = np.clip(result, 0, 255).astype(np.uint8)
    result = Image.fromarray(result)
    return result


result = conv(img)
result.save("output.jpg")
