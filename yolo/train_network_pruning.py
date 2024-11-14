import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def apply_pruning(model, amount=0.3):
    """Apply L1 unstructured pruning to the model's convolutional layers"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')

def train_with_pruning(amount):
    # 1. 학습된 모델 로드
    model = YOLO('gyeonggi_AD_competition/yolo11s_fold3/weights/best.pt')
    
    # 2. 프루닝 적용
    apply_pruning(model.model, amount=amount)
    
    # 3. 프루닝된 모델 fine-tuning
    model.train(data='../data/yaml/dataset_fold3.yaml',
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                project='gyeonggi_AD_competition',
                name=f'yolo11s_fold3_pruned{amount}'
                )
    
    # 4. 프루닝된 모델 저장
    model.save(f'yolo11s_fold3_pruned{amount}')

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    epochs = 100
    imgsz = 640
    batch_size = 32
    
    # 여러 amount 값으로 실험
    pruning_amounts = [0.1, 0.2, 0.3, 0.4, 0.5]  # 10%부터 50%까지
    
    for amount in pruning_amounts:
        print(f"\nStarting training with pruning amount: {amount}")
        train_with_pruning(amount)
        print(f"Completed training with pruning amount: {amount}")