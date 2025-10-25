import cv2, numpy as np

img = cv2.imread("data/processed/train/0a1gfi-0.png", cv2.IMREAD_GRAYSCALE)
print("Unique pixel values:", np.unique(img))
print("Mean pixel value:", np.mean(img))
