import os, csv, pathlib
from kmeans import kmeans
import cv2
import numpy as np

split = "test"
DATA_DIR = f"./data/{split}"  # <- change this to your folder
DEST_DIR = f"./data_letter/{split}"
rows = []
for root,_,files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith((".png",)):
            img = cv2.imread(f"{DATA_DIR}/{f}")
            k = len(f) - 6 + 1
            labels, centers = kmeans(img, k)

            cols = []
            for col in range(k):
              idxs = []
              for row in labels:
                for i, x in enumerate(row):
                  if x == col:
                    idxs.append(i)

              cols.append((sum(idxs) / len(idxs), col))
            
            cols.sort()

            running_i = 0
            for i, col_info in enumerate(cols):
              x, col = col_info
              if col == labels[0][0]:
                continue

              
              centers_tmp = np.array([[255, 255, 255] for i in range(k)])
              centers_tmp[col] = centers[col]

              res = centers_tmp[labels]

              l = max(0, round(x) - 40)
              r = min(res.shape[1] - 1, l + 80)
              l = r - 80
              res = res[:,l : r]
              # print(res)
              # print(x)
              res = np.float32(res)
              res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
              res = np.uint8(res)


              th = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

              cv2.imwrite(f"{DEST_DIR}/{len(rows)}.png", th)

              rows.append([f"{DEST_DIR}/{len(rows)}.png", f[running_i]])
              running_i += 1
              if f[running_i] == '(':
                print(f)


with open(f"{DEST_DIR}/labels.csv","w",newline="") as fp:
    w=csv.writer(fp); w.writerows(rows)
print("Wrote labels.csv:", len(rows))
