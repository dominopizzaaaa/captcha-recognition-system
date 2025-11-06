import cv2
import pandas as pd


df = pd.read_csv("./data/test/labels.csv")
for index, row in df.iterrows():
  path = f"./data/test/{row['path']}"
  img = cv2.imread(path)
  img[img == 0] = 255

  cv2.imwrite(path, img)