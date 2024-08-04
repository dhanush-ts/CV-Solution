from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data="../datasets/HRW-2/config.yaml",epochs=5)