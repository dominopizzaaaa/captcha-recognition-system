import cv2
import numpy as np
import torch

def add_pos_channels(img: torch.Tensor, mode: str = "norm"):
  """
  img: (3, H, W)
  mode: "norm" for [0,1], "minus1to1" for [-1,1]
  returns: (5, H, W)
  """
  assert img.ndim == 3, "Expected (C, H, W)"
  H, W, C = img.shape
  assert C == 3, "Expected 3-channel input"

  if mode == "norm":
    xs = torch.linspace(0 ,8000, W, device=img.device)
    ys = torch.linspace(0, 0, H, device=img.device)
  elif mode == "minus1to1":
    xs = torch.linspace(-1, 1, W, device=img.device)
    ys = torch.linspace(-1, 1, H, device=img.device)
  else:
    raise ValueError("mode must be 'norm' or 'minus1to1'")

  # Make 2D grids
  x_grid = xs.view(1, W, 1).expand(H, W, 1)   # (1, H, W)
  y_grid = ys.view(H, 1, 1).expand(H, W, 1)   # (1, H, W)

  
  white_mask = (torch.from_numpy(img) == 255).all(dim=-1, keepdim=True)
  x_grid = x_grid.clone()
  x_grid[white_mask] = 0
  y_grid[white_mask] = 0

  
  # Concat: (3 + 2, H, W)
  out = torch.cat([torch.from_numpy(img), x_grid, y_grid], dim=2)
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