"""
cleans and binarizes CAPTCHA images for better OCR performance
converts to grayscale, applies Otsu thresholding, removes noise, saves  results into data/processed/
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_TRAIN = "../data/train"
INPUT_TEST = "../data/test"
OUTPUT_TRAIN = "../data/processed/train"
OUTPUT_TEST = "../data/processed/test"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Enhance contrast (Histogram Equalization)
    gray = cv2.equalizeHist(gray)

    # 3. Adaptive thresholding (local binarization)
    # adaptiveThreshold: divides the image into small blocks, calculate average pixel brightness, subtract constant C from average, 
    # any pixel brighter than that threshold is set to white, otherwise black
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        15, 5
    )

    # 4. Morphological cleanup (open → erode → dilate)
    # 'open' removes thin noise lines that cross letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # light erosion breaks tiny bridges between characters
    binary = cv2.erode(binary, kernel, iterations=1)

    # light dilation restores character thickness after erosion
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 5. Ensure letters are white, background black
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    if white_pixels < black_pixels:
        binary = cv2.bitwise_not(binary)

    # 6. --- Remove long lines (horizontal/vertical/diagonal) WITHOUT rotating ---
    # detect straight lines and paint them out
    inv = cv2.bitwise_not(binary)                    # lines bright for edge detection
    edges = cv2.Canny(inv, 50, 150, apertureSize=3)
    # HoughLinesP finds straight segments at any angle
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=60,
        minLineLength=max(20, binary.shape[1]//10),  # long enough to be nuisance lines
        maxLineGap=10
    )
    if lines is not None:
        # draw over detected lines with background color (black in 'binary')
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(binary, (x1, y1), (x2, y2), color=0, thickness=2)

    # 7. Final light clean (close tiny gaps after line removal)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

def preprocess_and_save(input_dir, output_dir):
    """
    Processes all images in a folder and saves them.
    """
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        processed = preprocess_image(in_path)
        if processed is not None:
            cv2.imwrite(out_path, processed)

if __name__ == "__main__":
    print("=== Preprocessing train set ===")
    preprocess_and_save(INPUT_TRAIN, OUTPUT_TRAIN)

    print("\n=== Preprocessing test set ===")
    preprocess_and_save(INPUT_TEST, OUTPUT_TEST)

    print("\n✅ Done! Cleaned images saved in data/processed/")