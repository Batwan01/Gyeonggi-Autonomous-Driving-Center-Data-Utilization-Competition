import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_json(file_path):
    """JSON 파일을 로드"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_class_distribution(data):
    """데이터셋의 클래스 분포를 계산"""
    class_counts = {}
    for item in data:
        for ann in item['annotations']:
            category_id = ann['lbl_id']
            if category_id not in class_counts:
                class_counts[category_id] = 0
            class_counts[category_id] += 1
    return class_counts

def plot_class_distribution(distributions, title):
    """클래스 분포를 히트맵으로 시각화"""
    df = pd.DataFrame(distributions).fillna(0)
    df = df.reindex(sorted(df.columns), axis=1)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(df, annot=True, fmt='g', cmap='YlGnBu')
    plt.title(title)
    plt.xlabel('Class ID')
    plt.ylabel('Fold')
    plt.tight_layout()
    plt.show()

# 모든 fold에 대한 분포 계산
train_distributions = []
val_distributions = []

for fold in range(1, 6):  # 5-fold
    # 각 fold의 훈련 및 검증 데이터 로드
    train_data = load_json(f'./data/json/train_fold{fold}.json')
    val_data = load_json(f'./data/json/val_fold{fold}.json')
    
    # 클래스 분포 계산
    train_dist = get_class_distribution(train_data)
    val_dist = get_class_distribution(val_data)
    
    train_distributions.append(train_dist)
    val_distributions.append(val_dist)

# 분포 시각화
plot_class_distribution(train_distributions, 'Class Distribution in Training Sets')
plot_class_distribution(val_distributions, 'Class Distribution in Validation Sets')

# 전체 데이터셋의 클래스 분포 계산
total_distribution = {}
for dist in train_distributions + val_distributions:
    for class_id, count in dist.items():
        if class_id not in total_distribution:
            total_distribution[class_id] = 0
        total_distribution[class_id] += count

# 클래스 이름 매핑 생성 (첫 번째 fold의 데이터에서 추출)
class_names = {}
for item in train_data:
    for ann in item['annotations']:
        class_names[ann['lbl_id']] = ann['lbl_nm']

# 각 fold의 분포와 전체 분포 비교
# 각 fold의 분포와 전체 분포 비교
for fold in range(5):
    train_dist = train_distributions[fold]
    val_dist = val_distributions[fold]
    
    print(f"\nFold {fold+1}:")
    print("Class ID | Name              | Total %  | Train %  | Val %")
    print("---------+-------------------+----------+----------+---------")
    for class_id in sorted(total_distribution.keys()):
        class_name = class_names[class_id]
        total_percent = total_distribution[class_id] / sum(total_distribution.values()) * 100
        train_percent = train_dist.get(class_id, 0) / sum(train_dist.values()) * 100
        val_percent = val_dist.get(class_id, 0) / sum(val_dist.values()) * 100
        print(f"{class_id:8d} | {class_name:<17s} | {total_percent:7.2f}% | {train_percent:7.2f}% | {val_percent:7.2f}%")

# 클래스 간 불균형 정도 계산
class_imbalance = max(total_distribution.values()) / min(total_distribution.values())
print(f"\nClass imbalance ratio (max/min): {class_imbalance:.2f}")

# 클래스별 총 개수 출력
print("\nTotal instances per class:")
print("Class ID | Name              | Count")
print("---------+-------------------+-------")
for class_id in sorted(total_distribution.keys()):
    class_name = class_names[class_id]
    count = total_distribution[class_id]
    print(f"{class_id:8d} | {class_name:<17s} | {count:6d}")