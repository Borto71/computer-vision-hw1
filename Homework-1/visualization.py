import cv2
import matplotlib
matplotlib.use("Agg")  # no windows
import matplotlib.pyplot as plt


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def comparison_figure(name, original, results, scores, path):
    """Original + degraded + every method, 2 rows x 4. Metrics written in the titles"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    axes[0].imshow(rgb(original))
    axes[0].set_title("Original (reference)")

    for ax, (method, res) in zip(axes[1:], results.items()):
        p, s = scores[method]["psnr"], scores[method]["ssim"]
        label = "Degraded (baseline)" if method == "baseline" else method
        ax.imshow(rgb(res))
        ax.set_title(f"{label}\nPSNR {p:.2f} dB | SSIM {s:.3f}", fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{name} - synthetic degradation", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def real_figure(name, original, results, path):
    """Same grid but on the real image, no metrics because there is no clean reference"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    axes[0].imshow(rgb(original))
    axes[0].set_title("Real image (no processing)")
    shown = [(m, r) for m, r in results.items() if m != "baseline"]  # baseline = the image itself
    for ax, (method, res) in zip(axes[1:], shown):
        ax.imshow(rgb(res))
        ax.set_title(method)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{name} - methods on the real image", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def histogram_figure(name, original, results, path):
    """L histograms, useful to see what stretching vs equalization actually do"""
    panels = [("original", original)] + [(m, results[m]) for m in
                                         ("baseline", "stretch_lab", "histeq_lab", "nlm_clahe")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3), sharey=True)
    for ax, (label, img) in zip(axes, panels):
        l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
        ax.hist(l.ravel(), bins=256, range=(0, 255), color="gray")
        ax.set_title("degraded" if label == "baseline" else label)
        ax.set_xlim(0, 255)  # same x scale everywhere, otherwise they can't be compared
    axes[0].set_ylabel("pixels")
    fig.suptitle(f"{name} - L channel histogram")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
