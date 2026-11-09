import csv
from pathlib import Path

import cv2
import numpy as np

from data import load_image, degrade
from enhancement import METHODS
from metrics import compute_all
from visualization import comparison_figure, real_figure, histogram_figure

# Parameters set-up
SEED = 42
ALPHA = 0.5        # contrast factor, 1 = untouched, 0 = flat gray
SIGMA = 10.0       # std of the gaussian noise, in gray levels (0-255)
ILLUM_MIN = 0.7    # brightness on the left border of the ramp (1.0 = no ramp)
MAX_SIDE = 1024    # img1 is a 4032x3024 phone photo, way too big for NLM

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIG_DIR = ROOT / "img"
DEG_DIR = ROOT / "img_degraded"
RES_DIR = ROOT / "results"
ENH_DIR = RES_DIR / "enhanced"   # methods applied to the degraded images
REAL_DIR = RES_DIR / "real"      # methods applied to the real originals (no reference here)
FIG_DIR = RES_DIR / "figures"


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "ssim", "mae"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "psnr": f"{r['psnr']:.2f}", "ssim": f"{r['ssim']:.4f}",
                             "mae": f"{r['mae']:.2f}"})


def main():
    for d in (DEG_DIR, ENH_DIR, REAL_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    paths = sorted(ORIG_DIR.glob("*.jpg"))  # sorted -> same order on every machine
    if not paths:
        raise SystemExit(f"No images found in {ORIG_DIR}")

    rows = []
    for i, path in enumerate(paths):
        name = path.stem
        print(f"\n[{i + 1}/{len(paths)}] {name}")
        original = load_image(path, MAX_SIDE)

        # a seed for each image: adding/removing an image doesn't change the noise of the others
        rng = np.random.default_rng(SEED + i)
        degraded = degrade(original, ALPHA, SIGMA, rng, ILLUM_MIN)
        cv2.imwrite(str(DEG_DIR / f"{name}_degraded.png"), degraded)  # png, jpg would change the noise

        # 1) synthetic case: methods on the degraded image, compared with the original
        results, scores = {}, {}
        for method, fn in METHODS.items():
            out = fn(degraded)
            results[method] = out
            cv2.imwrite(str(ENH_DIR / f"{name}_{method}.png"), out)

            scores[method] = compute_all(original, out)
            rows.append({"image": name, "method": method, **scores[method]})
            print(f"  {method:<12} PSNR {scores[method]['psnr']:6.2f}  "
                  f"SSIM {scores[method]['ssim']:.3f}  MAE {scores[method]['mae']:6.2f}")

        comparison_figure(name, original, results, scores, FIG_DIR / f"{name}_comparison.png")
        histogram_figure(name, original, results, FIG_DIR / f"{name}_histogram.png")

        # 2) real case: same methods on the original, only for a visual check
        real_results = {}
        for method, fn in METHODS.items():
            if method == "baseline":
                real_results[method] = original
                continue
            real_results[method] = fn(original)
            cv2.imwrite(str(REAL_DIR / f"{name}_{method}.png"), real_results[method])
        real_figure(name, original, real_results, FIG_DIR / f"{name}_real.png")

    # mean over all the images, one row per method
    for method in METHODS:
        mine = [r for r in rows if r["method"] == method]
        rows.append({"image": "mean", "method": method,
                     **{k: float(np.mean([r[k] for r in mine])) for k in ("psnr", "ssim", "mae")}})

    save_csv(rows, RES_DIR / "metrics.csv")

    print("\nMean over all images:")
    for r in rows:
        if r["image"] == "mean":
            print(f"  {r['method']:<12} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.3f}  MAE {r['mae']:6.2f}")
    print(f"\nDone, everything is in {RES_DIR}")


if __name__ == "__main__":
    main()