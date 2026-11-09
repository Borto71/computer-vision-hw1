import cv2
import numpy as np

# Parameters of the methods
P_LOW, P_HIGH = 1, 99           # percentiles for the stretching (min/max are ruined by noise)
NLM_H = 10                      # NLM filter strength, roughly the noise sigma
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)


def stretch_channel(ch):
    """Linear stretch of a single channel between the P_LOW and P_HIGH percentiles"""
    lo, hi = np.percentile(ch, (P_LOW, P_HIGH))
    if hi <= lo:  # flat channel, nothing to stretch
        return ch.copy()
    out = (ch.astype(np.float32) - lo) / (hi - lo) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def stretch_rgb(img):
    """Stretch every channel on its own -> each channel gets a different gain, colours can shift"""
    return cv2.merge([stretch_channel(c) for c in cv2.split(img)])


def stretch_lab(img):
    """Stretch only the L channel in Lab, a and b are left alone so the colours stay the same"""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([stretch_channel(l), a, b]), cv2.COLOR_LAB2BGR)


def histeq_lab(img):
    """Global histogram equalization on L (Lab)"""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([cv2.equalizeHist(l), a, b]), cv2.COLOR_LAB2BGR)


def clahe_lab(img):
    """CLAHE on L: equalization done on tiles, so it adapts to the uneven light"""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def nlm(img):
    """Non-local means denoising (OpenCV colored version)"""
    return cv2.fastNlMeansDenoisingColored(img, None, NLM_H, NLM_H, 7, 21)


def nlm_clahe(img):
    """My 4th method: denoise first, otherwise CLAHE would boost the noise too"""
    return clahe_lab(nlm(img))


# nlm and clahe alone are there just to see what each step of nlm_clahe does
METHODS = {
    "baseline": lambda img: img.copy(),
    "stretch_rgb": stretch_rgb,
    "stretch_lab": stretch_lab,
    "histeq_lab": histeq_lab,
    "nlm": nlm,
    "clahe": clahe_lab,
    "nlm_clahe": nlm_clahe,
}
