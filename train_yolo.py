"""
train_yolo_cls.py — YOLOv8 Classification on sliced character images
"""

from ultralytics import YOLO

# load a small pretrained model for classification
model = YOLO("yolo11x-cls.pt")  # can use yolov8s-cls.pt for higher accuracy

if __name__ == '__main__':
    # train
    model.train(
        data="./data_letter_yolo",   # path to prepared dataset folder
        epochs=1000,
        imgsz=80,                    # small characters, 64x64 is enough
        batch=128,
        device=0,                    # or "cpu"
    )
