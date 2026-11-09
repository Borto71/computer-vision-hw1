import cv2
import numpy as np


def psnr(ref, img):
    return cv2.PSNR(ref, img)


def mae(ref, img):
    return float(np.mean(np.abs(ref.astype(np.float32) - img.astype(np.float32))))


def ssim(ref, img):
    """Done per channel and then averaged, written by hand to avoid installing scikit-image"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)

    scores = []
    for c in range(ref.shape[2]):
        x = ref[:, :, c].astype(np.float64)
        y = img[:, :, c].astype(np.float64)
        mu_x, mu_y = blur(x), blur(y)
        var_x = blur(x * x) - mu_x ** 2
        var_y = blur(y * y) - mu_y ** 2
        cov = blur(x * y) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
        # crop the borders, the blur there is not reliable (same as skimage does)
        scores.append(ssim_map[5:-5, 5:-5].mean())
    return float(np.mean(scores))


def compute_all(ref, img):
    return {"psnr": psnr(ref, img), "ssim": ssim(ref, img), "mae": mae(ref, img)}
