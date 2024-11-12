from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Albumentations 변환 적용
        if self.transform:
            image = self.transform(image=image)['image']

        # 라벨 불러오기
        with open(label_path, 'r') as f:
            label = f.read()

        return image, label

# Albumentations 변환 설정
weather_augmentations = A.Compose([
    A.RandomSnow(p=0.3),       
    A.RandomFog(p=0.3),        
    A.RandomRain(p=0.3),       
    A.RandomShadow(p=0.3),     
    A.Resize(640, 640),        
    ToTensorV2(),              
])

# 훈련 및 검증 데이터셋 및 데이터로더 설정
train_img_dir = "/data/ephemeral/home/hyeonwoo/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo_filtered/train/images"
train_label_dir = "/data/ephemeral/home/hyeonwoo/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo_filtered/train/labels"
val_img_dir = "/data/ephemeral/home/hyeonwoo/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo_filtered/val/images"
val_label_dir = "/data/ephemeral/home/hyeonwoo/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo_filtered/val/labels"

# 훈련 및 검증 데이터셋 생성
train_dataset = CustomDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=weather_augmentations)
val_dataset = CustomDataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=weather_augmentations)

# 데이터로더 설정
train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 모델 설정 및 훈련
model = YOLO("yolo11s.yaml")  # 모델 설정 (yolo11s.yaml 사용)
model.load("ultralytics/yolo11s.pt")  # 미리 훈련된 모델 로드

# 훈련 루프
for epoch in range(10000):
    # 훈련 단계
    model.train()
    for images, labels in train_data_loader:
        # 모델 훈련
        results = model.train(data=images, labels=labels)  # 현재 배치에 대해 훈련
    
    # 검증 단계
    model.eval()
    with torch.no_grad():
        for images, labels in val_data_loader:
            # 검증 진행
            results = model.train(data=images, labels=labels)  # 검증 데이터에 대해 훈련
    
    torch.cuda.empty_cache()  # 메모리 해제
