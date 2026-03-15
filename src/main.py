from PIL import Image
import numpy as np


img = Image.open("./images/breaking_bad.jpg").convert("RGB")


def to_grayscale(img: Image, type: str = "lum") -> Image:
    img_arr = np.array(img)

    if type == "lum":  # Lum method
        result = (
            0.299 * img_arr[::, ::, 0]
            + 0.587 * img_arr[::, ::, 1]
            + 0.114 * img_arr[::, ::, 2]
        )
    else:  # Average method
        result = np.mean(img_arr, axis=2)

    result = result.astype(np.uint8)
    result = Image.fromarray(result)
    return result


def conv(
    img: Image,
    kernel: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    padding: bool = True,
    optimize: bool = True,
) -> Image:
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel shape must be a square.")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel shape must be odd.")

    # img = to_grayscale(img)
    img_arr = np.array(img, dtype=np.float32)
    kernel = kernel.astype(np.float32)

    if padding:
        padding = (kernel.shape[0] - 1) // 2

        zero_h = np.zeros((img_arr.shape[0], padding, 3))
        zero_w = np.zeros((padding, img_arr.shape[1] + 2 * padding, 3))
        padded_img = np.hstack((zero_h, img_arr, zero_h), dtype=np.float32)
        padded_img = np.vstack((zero_w, padded_img, zero_w), dtype=np.float32)

        result = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.float32)

    else:
        padded_img = img_arr
        result = np.zeros(
            (
                img_arr.shape[0] - (kernel.shape[0] - 1),
                img_arr.shape[1] - (kernel.shape[1] - 1),
                3,
            ),
            dtype=np.float32,
        )

    if optimize:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                slice = padded_img[i : i + result.shape[0], j : j + result.shape[1]]

                result += slice * kernel[i, j]
    else:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                window = padded_img[i : i + kernel.shape[0], j : j + kernel.shape[0]]
                result[i][j] = np.sum(window * kernel, axis=(0, 1))

    result = np.clip(result, 0, 255).astype(np.uint8)
    result = Image.fromarray(result)
    return result


if __name__ == "__main__":
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    result = conv(img, padding=False, kernel=kernel)
    result.save("output.jpg")
