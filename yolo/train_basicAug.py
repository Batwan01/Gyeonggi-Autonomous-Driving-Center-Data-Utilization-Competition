from ultralytics import YOLO
import torch
import os

# 모델 설정 정보
model_configs = [
    {"yaml": "yolo11s.yaml", "weights": "ultralytics/yolo11s.pt", "save_dir": "runs/train/yolo11s"},
]

# 데이터 경로 및 공통 설정
data_path = "/data/ephemeral/home/hyeonwoo/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/data.yaml"
epochs = 100
imgsz = 640
batch_size = 8

# 순차적으로 모델을 훈련시키기
for config in model_configs:
    # 모델 불러오기 및 가중치 로드
    model = YOLO(config["yaml"]).load(config["weights"])
    
    # 모델 훈련 (증강 설정 적용)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        save_dir=config["save_dir"],
        augment=True,              # 기본 증강 활성화
        hsv_h=0.015,               # 색조 변화 범위 조절
        hsv_s=0.7,                 # 채도 변화 범위 조절
        hsv_v=0.4,                 # 밝기 변화 범위 조절
        flipud=0.1,                # 상하 반전 확률
        fliplr=0.5                 # 좌우 반전 확률
    )
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
