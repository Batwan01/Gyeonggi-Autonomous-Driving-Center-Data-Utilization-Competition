import os
import torch
import torch.nn.functional as F
import albumentations as A
from pathlib import Path
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from albumentations.pytorch import ToTensorV2


class WeatherAugmentedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.labels  # dataset의 labels 속성 추가
        self.weather_transforms = A.Compose([
            A.RandomSnow(p=0.3),       
            A.RandomFog(p=0.3),        
            A.RandomRain(p=0.3),       
            A.RandomShadow(p=0.3),     
            A.Resize(640, 640),        
            ToTensorV2(),              
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]  # dictionary 형태로 받기
        img = batch['img']
        
        # numpy로 변환 (Albumentations 요구사항)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.shape[0] == 3:  # CHW to HWC
                img = img.transpose(1, 2, 0)
        
        # Albumentations 변환 적용 (bbox 건드리지 않음)
        transformed = self.weather_transforms(image=img)
        img = transformed['image']
        
        # 변환된 이미지를 배치에 저장
        batch['img'] = img
        
        return batch

    @staticmethod
    def collate_fn(batch):
        """YOLODataset의 collate_fn을 그대로 사용"""
        return YOLODataset.collate_fn(batch)


class CustomDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset with top cropping.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        
        if mode == "train":
            dataset = WeatherAugmentedDataset(dataset)
            LOGGER.info(f"Created top-cropped dataset for training")
        
        return dataset
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
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
data_path = str(ROOT / "../data/yaml/dataset_fold1.yaml")
epochs = 100
imgsz = 640
batch_size = 32

# trainer 초기화 및 학습 시작
args = dict(
    model='yolo11s.pt',
    data=data_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    save_dir=str(ROOT / "runs/train/yolo11s"),
    # mosaic=0.0,
    # erasing=0.0,
    # crop_fraction=0.0,
    # fliplr=0.0,
    # scale=0.0,
    # translate=0.0,
)

trainer = CustomDetectionTrainer(overrides=args)
trainer.train()