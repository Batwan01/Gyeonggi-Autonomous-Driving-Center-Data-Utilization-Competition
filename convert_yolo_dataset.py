import json
import os
import shutil
from tqdm import tqdm

# 경로 설정
train_json_path = '/hdd1/lim_data/2d_cctv/filtered_train_split.json'  # train_split.json 파일 경로
val_json_path = '/hdd1/lim_data/2d_cctv/filtered_val_split.json'      # val_split.json 파일 경로
image_folder = '/hdd1/lim_data/2d_cctv/train'               # 전체 이미지 폴더 경로
output_folder = '/hdd1/lim_data/2d_cctv_yolo_filtered'            # YOLO 형식 데이터셋이 저장될 폴더

# 출력 폴더 생성
train_image_folder = os.path.join(output_folder, 'train', 'images')
val_image_folder = os.path.join(output_folder, 'val', 'images')
train_label_folder = os.path.join(output_folder, 'train', 'labels')
val_label_folder = os.path.join(output_folder, 'val', 'labels')

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# YOLO 형식으로 변환하는 함수
def convert_to_yolo(annotation, img_width, img_height):
    class_id = annotation['lbl_id'] - 1  # 클래스 ID를 0부터 시작하도록 조정
    x, y, w, h = eval(annotation['annotations_info'])
    
    # YOLO 형식 변환 (비율 계산)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# JSON 파일을 YOLO 형식으로 변환하고 파일을 이동하는 함수
def process_data(json_path, image_folder, image_output_folder, label_output_folder):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in tqdm(data):
        # 이미지 정보 로드
        image_info = item['images']
        img_id = image_info['img_id']
        img_width = image_info['img_width']
        img_height = image_info['img_height']
        
        # 이미지 파일 이동
        image_name = f"{img_id}.jpg"
        src_image_path = os.path.join(image_folder, image_name)
        dst_image_path = os.path.join(image_output_folder, image_name)
        
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"이미지를 찾을 수 없습니다: {src_image_path}")
            continue
        
        # 라벨 파일 생성
        label_file_path = os.path.join(label_output_folder, f"{img_id}.txt")
        with open(label_file_path, 'w') as label_file:
            for annotation in item['annotations']:
                yolo_format = convert_to_yolo(annotation, img_width, img_height)
                label_file.write(yolo_format + '\n')

# train과 val 데이터를 각각 YOLO 형식으로 변환
process_data(train_json_path, image_folder, train_image_folder, train_label_folder)
process_data(val_json_path, image_folder, val_image_folder, val_label_folder)

print("YOLO 형식으로 변환 및 데이터셋 구분 완료.")