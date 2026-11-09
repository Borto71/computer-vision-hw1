import cv2
import numpy as np


def load_image(path, max_side):
    """Read a color image and shrink it if the longest side is above max_side"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        # INTER_AREA is the one to use when going down in size
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def degrade(img, alpha, sigma, rng, illum_min=1.0):
    """Reduce contrast around the mean intensity, darken the left side and add Gaussian noise"""
    f = img.astype(np.float32)
    mu = f.mean()  # one mean for all the channels, so the colours don't shift
    low_contrast = alpha * (f - mu) + mu

    # horizontal ramp: dark on the left, normal on the right -> non uniform illumination
    w = img.shape[1]
    ramp = np.linspace(illum_min, 1.0, w, dtype=np.float32)[None, :, None]
    low_contrast = low_contrast * ramp

    noise = rng.normal(0.0, sigma, size=f.shape).astype(np.float32)

    return np.clip(low_contrast + noise, 0, 255).round().astype(np.uint8)
