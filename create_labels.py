import os, csv, pathlib
DATA_DIR = r"./data/train"  # <- change this to your folder
rows = []
for root,_,files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith((".png",)):
            text = pathlib.Path(f).stem
            cnt = len(text) - 2
            rows.append([os.path.relpath(os.path.join(root,f), DATA_DIR), cnt])
with open(f"{DATA_DIR}/labels.csv","w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["path","count"]); w.writerows(rows)
print("Wrote labels.csv:", len(rows))
