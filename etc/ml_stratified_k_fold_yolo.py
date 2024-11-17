import json
import os
from tqdm import tqdm
import yaml
import shutil
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    [x, y, w, h] 형식의 bbox를 YOLO 형식으로 변환
    YOLO bbox: [x_center, y_center, width, height] (0~1로 정규화)
    """
    x, y, w, h = bbox
    
    # 중심점 계산 및 정규화
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # 너비, 높이 정규화
    width = w / img_width
    height = h / img_height

    return [x_center, y_center, width, height]

def split_data(json_file, n_splits=5):
    """
    데이터를 MultilabelStratifiedKFold를 사용하여 분할
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 이미지 정보 추출
    images_df = pd.DataFrame([item['images'] for item in data])
    
    # annotations 정보를 이미지별로 그룹화
    image_labels = {}
    for item in data:
        img_id = item['images']['img_id']
        if img_id not in image_labels:
            image_labels[img_id] = []
        for ann in item['annotations']:
            image_labels[img_id].append(ann['lbl_id'])
    
    # 각 이미지의 레이블 리스트를 데이터프레임에 추가
    images_df['labels'] = images_df['img_id'].map(image_labels)
    images_df['labels'] = images_df['labels'].fillna('').apply(list)
    
    # MultiLabelBinarizer를 사용하여 레이블을 이진 매트릭스로 변환
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(images_df['labels'])
    
    # MultilabelStratifiedKFold 적용
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(images_df, y), 1):
        train_img_ids = images_df.iloc[train_idx]['img_id'].values
        val_img_ids = images_df.iloc[val_idx]['img_id'].values
        
        # 각 폴드의 train/val 데이터 생성
        train_data = [item for item in data if item['images']['img_id'] in train_img_ids]
        val_data = [item for item in data if item['images']['img_id'] in val_img_ids]
        
        fold_data.append((train_data, val_data))
    
    return fold_data

def convert_to_yolo(data, output_path):
    """
    데이터를 YOLO 형식으로 변환
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 모든 카테고리 수집
    categories = {}
    for item in data:
        for ann in item['annotations']:
            categories[ann['lbl_id']] = ann['lbl_nm']
    
    # 정렬된 카테고리 리스트 생성
    categories = [{'id': id, 'name': name} for id, name in sorted(categories.items())]
    
    # 카테고리 ID를 인덱스로 변환
    category_map = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    # YOLO 형식으로 변환하여 저장
    for item in tqdm(data, desc="Converting to YOLO format"):
        img_info = item['images']
        img_id = img_info['img_id']
        img_width = img_info['img_width']
        img_height = img_info['img_height']
        
        label_path = os.path.join(output_path, f"{img_id}.txt")
        
        with open(label_path, 'w') as f:
            for ann in item['annotations']:
                category_idx = category_map[ann['lbl_id']]
                bbox = eval(ann['annotations_info'])
                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)
                f.write(f"{category_idx} {' '.join([str(x) for x in yolo_bbox])}\n")
    
    return categories

def copy_images(data, src_dir, dst_dir):
    """
    이미지 파일 복사
    """
    os.makedirs(dst_dir, exist_ok=True)
    for item in tqdm(data, desc=f"Copying images to {dst_dir}"):
        img_id = item['images']['img_id']
        src_path = os.path.join(src_dir, f"{img_id}.jpg")
        dst_path = os.path.join(dst_dir, f"{img_id}.jpg")
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 분할
    fold_data = split_data('data/train.json', n_splits=5)
    
    # 각 폴드 처리
    for fold, (train_data, val_data) in enumerate(fold_data, 1):
        # JSON 저장
        os.makedirs('data/json', exist_ok=True)
        with open(f'data/json/train_fold{fold}.json', 'w') as f:
            json.dump(train_data, f)
        with open(f'data/json/val_fold{fold}.json', 'w') as f:
            json.dump(val_data, f)
        
        # 이미지 복사
        os.makedirs('data/images', exist_ok=True)
        copy_images(train_data, 'data/train', f'data/images/train_fold{fold}')
        copy_images(val_data, 'data/train', f'data/images/val_fold{fold}')
        
        # YOLO 형식으로 변환
        os.makedirs('data/labels', exist_ok=True)
        categories = convert_to_yolo(train_data, f'data/labels/train_fold{fold}')
        convert_to_yolo(val_data, f'data/labels/val_fold{fold}')
        
        # YAML 설정 파일 생성
        os.makedirs('data/yaml', exist_ok=True)
        dataset_config = {
            'path': '../../data',
            'train': f'images/train_fold{fold}',
            'val': f'images/val_fold{fold}',
            'nc': len(categories),
            'names': [cat['name'] for cat in categories]  # lbl_nm 대신 name 사용
        }
        
        with open(f'data/yaml/dataset_fold{fold}.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"Fold {fold} processing complete")

    print("All folds processed successfully")