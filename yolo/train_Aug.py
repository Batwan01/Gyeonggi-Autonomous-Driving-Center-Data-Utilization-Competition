import os
from pathlib import Path
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 커스텀 Trainer 클래스 정의
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 데이터 증강 파이프라인 설정
        self.augments = A.Compose([
            A.RandomCrop(width=640, height=640),  # 랜덤 크롭
            A.HorizontalFlip(p=0.5),              # 좌우 반전
            A.RandomBrightnessContrast(p=0.2),    # 밝기/대비 랜덤 조정
            A.HueSaturationValue(p=0.3),          # 색상 변화
            A.Resize(height=640, width=640),      # 이미지 크기 조정
            ToTensorV2()                          # 텐서 변환
        ])

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader with custom augmentation."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # 학습 모드에서 데이터 증강을 설정
        if mode == "train":
            # YOLODataset의 transform을 수정하여 albumentations 적용
            dataset.transform = self.augments
        
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        shuffle = False
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

# 프로젝트 루트 디렉토리 설정
ROOT = Path(__file__).parent

# 데이터 경로 및 공통 설정
data_path = str(ROOT / "data.yaml")
epochs = 100
imgsz = 640
batch_size = 8

# trainer 초기화 및 학습 시작
args = dict(
    model='yolo11s.pt',
    data=data_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    save_dir=str(ROOT / "runs/train/yolo11s_withAug"),
    mosaic=0
)

trainer = CustomDetectionTrainer(overrides=args)
trainer.train()
