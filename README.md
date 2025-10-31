# ️ OCR Playground: `ocr-playground.py`

## Overview

A Python script for **Optical Character Recognition (OCR)** using **Tesseract** and **OpenCV**. It features powerful, optional image preprocessing to maximize OCR accuracy.

##  Usage

The script requires the image path.

### 1. Default run (with preprocessing)

Applies enabled filters(see filter map), saves the `*_preprocessed.png` image, and runs OCR.

```bash
python ocr-playground.py image_path/image_name.png
```
### 2. Run without preprocessing

```bash
python ocr-playground.py image_path/image_name.png --no-preprocess
```
