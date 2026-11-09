# Homework 1: Image Enhancement

Classical pipeline to enhance images affected by low contrast, non-uniform illumination and noise.
Seven variants are compared on synthetic degradations (PSNR, SSIM, MAE) and on the real originals (visual check only).
No learning-based method is used.

## Requirements

- Python 3.12.3
- numpy 2.4.6
- opencv-python 5.0.0.93
- matplotlib 3.11.2

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy==2.4.6 opencv-python==5.0.0.93 matplotlib==3.11.2
```

## Run

From the repository root:

```bash
python Homework-1/main.py
```

This reads the images in `img/`, creates the degraded versions and writes all the results and metrics.

## Images

| File | Source | Author | License |
|---|---|---|---|
| `img1.jpg` | personal photo | Mattia Bortolaso | own work |
| `img2.jpg` | [Pexels](https://www.pexels.com/it-it/foto/citta-alba-tramonto-punto-di-riferimento-18927434/) | Oscar Portan | [Pexels License](https://www.pexels.com/license/) |
| `img3.jpg` | [Pexels](https://www.pexels.com/it-it/foto/strada-nebbiosa-con-alberi-spogli-in-un-autunno-nebbioso-34649347/) | Christina & Peter | [Pexels License](https://www.pexels.com/license/) |
| `img4.jpg` | [Pexels](https://www.pexels.com/it-it/foto/vivace-mercato-notturno-con-prodotti-freschi-34559646/) | Jahra Tasfia Reza | [Pexels License](https://www.pexels.com/license/) |
| `img5.jpg` | [Pexels](https://www.pexels.com/it-it/foto/luce-del-sole-30422461/) | Try Putro Utomo | [Pexels License](https://www.pexels.com/license/) |
