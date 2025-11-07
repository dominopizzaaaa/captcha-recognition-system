import cv2
import numpy as np


def kmeans(img, k):
  # img = cv2.imread("data/train/zdfvcb-0.png")
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  alpha = 1  # Increase contrast
  beta = 10     # No change in brightness

  # Apply the transformation
  img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


  # kernel = np.ones((3, 3), np.float32) / 9 # Example 5x5 averaging kernel
  # img = cv2.filter2D(img, -1, kernel) # -1 means output image depth is same as input
  # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])
  kernel = np.ones((3, 3), np.uint8)
  img = cv2.erode(img, kernel, iterations=1)
  # Apply the filter
  # high = cv2.filter2D(img, -1, kernel)
  high = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), ddepth=cv2.CV_16S, ksize=1)
  # img = cv2.dilate(img, kernel, iterations=1)

  img[high != 0] = 255
  kernel = np.ones((3, 3), np.uint8)
  # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  img = cv2.erode(img, kernel, iterations=1)
  pixel_vals = img.reshape((-1, 3))
  pixel_vals = np.float32(pixel_vals)

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)


  retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

  labels = labels.flatten().reshape((img.shape[0], img.shape[1]))

  return labels, centers

  # print(retval)

  # print(centers[0])
  # centers[0] = 255
  # centers[1] = 255
  # centers[2] = 255
  # centers[3] = 255
  # centers[4] = 255
  # centers[5] = 255

  # segmented_data = centers[labels.flatten()]

  # segmented_image = segmented_data.reshape((img.shape))



  # cv2.imwrite("segmented.png", segmented_image)
if __name__ == "__main__":
  img = cv2.imread("data/train/nhotstv-0.png")
  labels, centers = kmeans(img, 9)

  segmented_data = centers[labels]
  cv2.imwrite("segmented.png", segmented_data)

# kmeans("a")
# 