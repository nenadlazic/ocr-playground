#!/usr/bin/env python3

"""
ocr-playground.py

A simple script to perform OCR on an image with optional preprocessing.  
Uses OpenCV to clean and prepare the image, and Tesseract OCR to extract text.

Features:
- Image preprocessing (grayscale, blur, threshold, morphology, etc.)
- OCR on original or preprocessed image
- Saves preprocessed image
- Optional flag to skip preprocessing (--no-preprocess)

Usage examples:
    python ocr-playground.py path/to/image.png
    python ocr-playground.py path/to/image.png --no-preprocess
"""

import cv2
import pytesseract
import sys
import os
import argparse
import numpy as np

# Default filters (all enabled by default)
DEFAULT_FILTERS = {
    "grayscale": True,             # Convert image to grayscale (essential)
    "resize": True,                # Upscale small images to improve OCR
    "contrast_enhance": True,      # Apply CLAHE contrast enhancement
    "gaussian_denoise": False,     # Gentle Gaussian blur to reduce noise
    "normalize_contrast": False,   # Linear normalization to balance overall contrast
    "deskew": False,               # Straighten rotated text
    "invert_colors": False,        # Invert colors if text is lighter than background
    "sharpen": False,              # Sharpen edges to enhance text
    "remove_snow": False,          # Remove tiny white specks/noise after binarization
}


# Prepare an image for OCR by applying standard preprocessing steps.
def preprocess(img, filters=None):
    """
    Preprocess an image for OCR with configurable filters.
    Logs each applied filter and relevant parameters.
    """
    if filters is None:
        filters = DEFAULT_FILTERS

    processed = img.copy()

    # 1. Grayscale
    if filters.get("grayscale"):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        print("[INFO] Applied grayscale filter")

    # 2. Resize
    if filters.get("resize"):
        h, w = processed.shape[:2]
        print(f"[INFO] Original image size: {w}x{h}")
        if w < 700:
            target_width = 900
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"[INFO] Upscaled image -> New size: {new_w}x{new_h}")
        else:
            print("[INFO] Skipping resize (sufficient resolution)")

    # 3. Contrast enhancement (CLAHE)
    if filters.get("contrast_enhance", False):
        processed = apply_auto_clahe(processed)
        print("[INFO] Applied CLAHE contrast enhancement")

    # 4. Gentle Gaussian denoise
        if filters.get("gaussian_denoise", False):
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            print("[INFO] Applied gentle Gaussian denoise")

    # 5. Normalize contrast
    if filters.get("normalize_contrast", False):
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        print("[INFO] Applied linear contrast normalization")
        
    # 6. Deskew
    if filters.get("deskew"):
        coords = np.column_stack(np.where(processed > 0))
        if coords.shape[0] > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = processed.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            processed = cv2.warpAffine(processed, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
            print(f"[INFO] Applied deskew filter -> Angle: {angle:.2f} degrees")

    # 7. Invert colors
    if filters.get("invert_colors", False):
        processed = cv2.bitwise_not(processed)
        print("[INFO] Applied invert colors filter")

    # 8. Sharpen
    if filters.get("sharpen", False):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)
        print("[INFO] Applied sharpening filter")

    # 9. Remove tiny white specks ("snow") safely
    if filters.get("remove_snow", False):
        # Ensure grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Remove light pixels likely to be noise
        snow_mask = processed >= 230
        processed[snow_mask] = 0

        # Gentle median blur to smooth remaining specks
        processed = cv2.medianBlur(processed, 3)

        print("[INFO] Removed light white specks (snow) while preserving text")


    return processed



def apply_auto_clahe(gray):
    """
    Apply adaptive CLAHE based on estimated contrast.
    """
    mean, stddev = cv2.meanStdDev(gray)
    contrast = stddev[0][0]
    print(f"[INFO] Estimated contrast stddev = {contrast:.2f}")

    if contrast < 30:
        clipLimit, gridSize = 3.0, (4, 4)
        print("[INFO] Low contrast detected → strong CLAHE")
    elif contrast < 70:
        clipLimit, gridSize = 2.0, (8, 8)
        print("[INFO] Medium contrast → standard CLAHE")
    else:
        clipLimit, gridSize = 1.5, (8, 8)
        print("[INFO] High contrast → mild CLAHE")

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=gridSize)
    return clahe.apply(gray)


# Perform OCR on an image.

# If do_preprocess=True, the image will first be preprocessed to improve text recognition.  
# The preprocessed image will be saved with a "_preprocessed" suffix in the same folder.
def ocr_image(image_path, do_preprocess=True, filters=None):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    if do_preprocess:
        processed_img = preprocess(img, filters)
        # Save preprocessed image
        base, ext = os.path.splitext(image_path)
        preprocessed_path = f"{base}_preprocessed{ext}"
        cv2.imwrite(preprocessed_path, processed_img)
        print(f"[INFO] Preprocessed image saved as: {preprocessed_path}")
    else:
        processed_img = img
        print("[INFO] Skipping preprocessing. Using original image for OCR.")

    text = pytesseract.image_to_string(processed_img)
    print("OCR Result:", text.strip() or "[No text detected]")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR an image with optional preprocessing filters")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-n", "--no-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--disable", nargs="*", default=[], help="List of filters to disable")

    args = parser.parse_args()

    # Prepare filters based on command-line arguments
    filters = DEFAULT_FILTERS.copy()
    for f in args.disable:
        if f in filters:
            filters[f] = False

    ocr_image(args.image_path, do_preprocess=not args.no_preprocess, filters=filters)
