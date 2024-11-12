from ultralytics import YOLO
import torch
import os
import yaml
import wandb

wandb.login(key="6bbca80c9b2d573046d3ffe19c0d407ba54ad774")

# 모델 설정 정보
model_configs = [
    {"yaml": "yolo11s.yaml", "weights": "ultralytics/yolo11s.pt", "save_dir": "runs/train/yolo11s"},
]

# 공통 설정
epochs = 100
imgsz = 640
batch_size = 8
n_folds = 5  # fold 개수

# 각 fold별로 순차적으로 모델을 훈련
for fold in range(1, n_folds + 1):
    print(f"\n=== Training Fold {fold} ===")
    
    # fold별 데이터 yaml 파일 경로
    data_yaml_path = f"../data/yaml/dataset_fold{fold}.yaml"
    
    # yaml 파일이 존재하는지 확인
    if not os.path.exists(data_yaml_path):
        print(f"Warning: {data_yaml_path} not found. Skipping fold {fold}")
        continue
    
    # 각 모델 설정에 대해 학습 수행
    for config in model_configs:
        print(f"\nTraining {os.path.basename(config['yaml'])} on fold {fold}")
        
        # fold별 저장 디렉토리 설정
        fold_save_dir = f"{config['save_dir']}_fold{fold}"
        
        try:
            # 모델 불러오기 및 가중치 로드
            model = YOLO(config["yaml"]).load(config["weights"])
            
            # 모델 훈련
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                save_dir=fold_save_dir,
                device=0,  # GPU 설정
                project='gyeonggi_AD_competition',
                name=f'yolo11s_fold{fold}'
            )
            
            # 학습 완료된 모델 저장
            model.save(f"{fold_save_dir}/weights/best.pt")
            
        except Exception as e:
            print(f"Error training fold {fold}: {str(e)}")
            continue
        
        finally:
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
        print(f"Completed training fold {fold}")

print("\nAll folds training completed!")
