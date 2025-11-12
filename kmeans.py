import cv2
import numpy as np
import torch

def add_pos_channels(img, mode: str = "norm"):
    """
    img: numpy array (H, W, 3)
    mode: "norm" or "minus1to1"
    returns: numpy array (H, W, 5)
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Expected (H, W, 3)"
    H, W, C = img.shape

    if mode == "norm":
        xs = np.linspace(0, 1, W, dtype=np.float32)
        ys = np.linspace(0, 1, H, dtype=np.float32)
    elif mode == "minus1to1":
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        ys = np.linspace(-1, 1, H, dtype=np.float32)
    else:
        raise ValueError("mode must be 'norm' or 'minus1to1'")

    x_grid = np.tile(xs, (H, 1))
    y_grid = np.tile(ys[:, None], (1, W))

    white_mask = (img == 255).all(axis=2)
    x_grid[white_mask] = 0
    y_grid[white_mask] = 0

    out = np.dstack([img, x_grid, y_grid])
    return out


def kmeans(img, k):
  # img = cv2.imread("data/train/zdfvcb-0.png")
  img = np.array(img)
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
  # img = cv2.erode(img, kernel, iterations=1)
  # Apply the filter
  # high = cv2.filter2D(img, -1, kernel)
  high = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), ddepth=cv2.CV_16S, ksize=1)
  # img[high != 0] = 255
  # img = cv2.dilate(img, kernel, iterations=1)

  # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  kernel = np.ones((3, 3), np.uint8)
  # img = cv2.erode(img, kernel, iterations=1)
  img = add_pos_channels(img)
  pixel_vals = img.reshape((-1, 5))
  pixel_vals = np.float32(pixel_vals)

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)


  retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

  labels = labels.flatten().reshape((img.shape[0], img.shape[1]))

  centers = centers[:,:3]

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

def getSegmentedImages(xx, k):
  # x = xx.squeeze(0)
  x = xx
  # x = (np.transpose(x.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
  H, W, C = x.shape

  labels, centers = kmeans(x, k)
  x_coords = torch.linspace(0, W, W, device=x.device)
  x_coords = x_coords.view(1, W, 1).expand(H, W, 1)


  cols = []  
  for col in range(k):
    tmp_x = x_coords[labels == col]
    cols.append((tmp_x.mean(), col))

  cols.sort()
  ret = []
  for i, col_info in enumerate(cols):
    coord, col = col_info
    if col == labels[0][0]:
      continue

    centers_tmp = np.array([[255, 255, 255] for i in range(k)])
    centers_tmp[col] = centers[col]

    res = centers_tmp[labels]

    l = max(0, round(coord.item()) - 40)
    r = min(res.shape[1] - 1, l + 80)
    l = r - 80

    # l = 0
    # r = res.shape[1] - 1

    res = res[:,l : r]
    # print(res)
    # print(x)
    res = np.float32(res)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    res = np.uint8(res)

    th = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    ret.append(th)

  return ret


if __name__ == "__main__":
  img = cv2.imread("data/train/0gfm9dbw-0.png")
  H, W, C = img.shape
  img = img.reshape(C, H, W)
  img = torch.tensor(img)
  labels, centers = kmeans(img, 9)

  getSegmentedImages(img, 9)

  segmented_data = centers[labels]
  # segmented_data = segmented_data[:,:,:3]
  cv2.imwrite("segmented.png", segmented_data)

# kmeans("a")
# 