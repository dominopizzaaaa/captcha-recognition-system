import os
import cv2
from tqdm import tqdm

def load_dataset(folder):
    data = []
    for file in tqdm(os.listdir(folder)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(file)[0].lower()  # label from filename
            path = os.path.join(folder, file)
            data.append((path, label))
    return data

train_data = load_dataset('data/train')
test_data = load_dataset('data/test')
print(f"Loaded {len(train_data)} training and {len(test_data)} test samples.")

bad = []
for path, _ in train_data:
    img = cv2.imread(path)
    if img is None:
        bad.append(path)
print("Corrupt images:", len(bad))

from collections import Counter

# String length stats
lengths = [len(label) for _, label in train_data]
print("=== String Length Statistics ===")
print(f"Total samples: {len(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")

length_counts = Counter(lengths)
print("\nCount by string length:")
for length, count in sorted(length_counts.items()):
    print(f"Length {length}: {count} samples")

# Character frequency
chars = Counter(''.join(label for _, label in train_data))
print("\n=== Character Frequency (A–Z, 0–9) ===")
for ch, freq in sorted(chars.items()):
    print(f"{ch}: {freq}")
