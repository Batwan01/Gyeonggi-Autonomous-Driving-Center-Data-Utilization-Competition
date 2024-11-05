from ultralytics import YOLO
import torch
import os

# 모델 설정 정보
model_configs = [
    {"yaml": "yolo11l.yaml", "weights": "ultralytics/yolo11l.pt", "save_dir": "runs/train/yolo11l"},
    {"yaml": "yolo11n.yaml", "weights": "ultralytics/yolo11n.pt", "save_dir": "runs/train/yolo11n"},
    {"yaml": "ultralytics/cfg/models/v10/yolov10l.yaml", "weights": "ultralytics/yolov10l.pt", "save_dir": "runs/train/yolov10l"},
    {"yaml": "ultralytics/cfg/models/v10/yolov10m.yaml", "weights": "ultralytics/yolov10m.pt", "save_dir": "runs/train/yolov10m"},
    {"yaml": "ultralytics/cfg/models/v10/yolov10b.yaml", "weights": "ultralytics/yolov10b.pt", "save_dir": "runs/train/yolov10b"},
    {"yaml": "ultralytics/cfg/models/v9/yolov9c.yaml", "weights": "ultralytics/yolov9c.pt", "save_dir": "runs/train/yolov9c"},
    {"yaml": "ultralytics/cfg/models/v8/yolov8.yaml", "weights": "ultralytics/yolov8l.pt", "save_dir": "runs/train/yolov8l"},
    {"yaml": "ultralytics/cfg/models/v8/yolov8.yaml", "weights": "ultralytics/yolov8m.pt", "save_dir": "runs/train/yolov8m"}
]

# 데이터 경로 및 공통 설정
data_path = "/data/ephemeral/home/jiwan/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/data.yaml"
epochs = 20
imgsz = 1024
batch_size = 8

# 순차적으로 모델을 훈련시키기
for config in model_configs:
    # 모델 불러오기 및 가중치 로드
    model = YOLO(config["yaml"]).load(config["weights"])
    
    # 모델 훈련
    results = model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch_size, save_dir=config["save_dir"])
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
