from ultralytics import YOLO

# load a small pretrained backbone (good for quick training)
model = YOLO("yolov8n.pt")

model.train(
    data="data_yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
