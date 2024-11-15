import os
from pathlib import Path
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first

class CustomDetectionTrainer(DetectionTrainer):
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
    save_dir=str(ROOT / "runs/train/yolo11s"),
)

trainer = CustomDetectionTrainer(overrides=args)
trainer.train()