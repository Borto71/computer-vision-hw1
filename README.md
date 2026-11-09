# Homework 1: Image Enhancement

Classical pipeline to enhance images affected by low contrast, non-uniform illumination and noise.
Seven variants are compared on synthetic degradations (PSNR, SSIM, MAE) and on the real originals (visual check only).
No learning-based method is used.

## Requirements

- Python 3.12.3
- numpy 2.4.6
- opencv-python 5.0.0.93
- matplotlib 3.11.2
- reportlab 5.1.0 (only to rebuild the PDF report)

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy==2.4.6 opencv-python==5.0.0.93 matplotlib==3.11.2 reportlab==5.1.0
```

## Run

From the repository root:

```bash
python Homework-1/main.py
```

This reads the images in `img/`, creates the degraded versions and writes all the results and metrics.

To rebuild the report after running the pipeline:

```bash
python Homework-1/report/build_report.py
```

## File structure

```
.
├── img/                  original images
├── img_degraded/         degraded versions (generated)
├── results/
│   ├── enhanced/         methods applied to the degraded images
│   ├── real/             methods applied to the real originals
│   ├── figures/          comparison figures and histograms
│   └── metrics.csv       PSNR, SSIM and MAE per image and method
└── Homework-1/
    ├── main.py           runs the complete pipeline
    ├── data.py           image loading and degradation
    ├── enhancement.py    enhancement methods
    ├── metrics.py        PSNR, SSIM and MAE
    ├── visualization.py  comparison figures and histograms
    └── report/           report source, figures and PDF
```

## Degradation parameters

Set at the top of `Homework-1/main.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `SEED` | 42 | base seed, image `i` (alphabetical order) uses `SEED + i` |
| `ALPHA` | 0.5 | contrast factor around the image mean |
| `SIGMA` | 10.0 | standard deviation of the Gaussian noise, in grey levels |
| `ILLUM_MIN` | 0.7 | brightness on the left border of the horizontal illumination ramp |
| `MAX_SIDE` | 1024 | longest side after resizing |

The original images are the reference for the metrics on the synthetic degradations. A clean reference may not be available in real-world scenarios.

## Methods

| Name | Description |
|---|---|
| `baseline` | degraded image, no processing |
| `stretch_rgb` | linear stretching (1st to 99th percentile) per channel |
| `stretch_lab` | linear stretching (1st to 99th percentile) on the L channel of CIE Lab |
| `histeq_lab` | histogram equalization on the L channel of CIE Lab |
| `nlm` | non-local means denoising |
| `clahe` | CLAHE on the L channel |
| `nlm_clahe` | non-local means denoising followed by CLAHE (fourth method) |

## Images

| File | Source | Author | License |
|---|---|---|---|
| `img1.jpg` | personal photo | Mattia Bortolaso | own work |
| `img2.jpg` | [Pexels](https://www.pexels.com/it-it/foto/citta-alba-tramonto-punto-di-riferimento-18927434/) | Oscar Portan | [Pexels License](https://www.pexels.com/license/) |
| `img3.jpg` | [Pexels](https://www.pexels.com/it-it/foto/strada-nebbiosa-con-alberi-spogli-in-un-autunno-nebbioso-34649347/) | Christina & Peter | [Pexels License](https://www.pexels.com/license/) |
| `img4.jpg` | [Pexels](https://www.pexels.com/it-it/foto/vivace-mercato-notturno-con-prodotti-freschi-34559646/) | Jahra Tasfia Reza | [Pexels License](https://www.pexels.com/license/) |
| `img5.jpg` | [Pexels](https://www.pexels.com/it-it/foto/luce-del-sole-30422461/) | Try Putro Utomo | [Pexels License](https://www.pexels.com/license/) |
