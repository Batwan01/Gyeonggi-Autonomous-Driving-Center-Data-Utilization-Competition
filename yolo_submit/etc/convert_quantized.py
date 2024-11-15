from ultralytics import YOLO
import torch

# 원본 YOLO 모델 불러오기
model_path = "/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/quantized_weight/best.pt"  # 원본 모델 경로
model = YOLO(model_path)

# 동적 양자화 적용
quantized_model = torch.quantization.quantize_dynamic(
    model.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# 양자화된 모델을 YOLO의 모델 형태로 저장
quantized_model_path = "/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/quantized_weight/quantized_yolo11s_2.pt"
model.model = quantized_model  # YOLO 모델 객체에 양자화된 모델 할당
model.save(quantized_model_path)  # YOLO 모델의 save() 메서드를 사용하여 저장
