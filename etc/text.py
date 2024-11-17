import os
import json
# JSON 데이터 파일 경로 설정
json_file = '/root/dataset/train.json'

# 출력 폴더 설정
output_dir = '/root/dataset/train/labels'
os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

# JSON 파일 열기
with open(json_file, 'r') as file:
    data = json.load(file)

# JSON 데이터를 파싱하고 YOLO 형식으로 변환하여 저장하는 함수
def convert_to_yolo_format(data):
    for item in data:
        img_info = item['images']
        annotations = item['annotations']
        img_id = img_info['img_id']
        img_width = img_info['img_width']
        img_height = img_info['img_height']
        
        # YOLO 형식의 파일 생성
        with open(os.path.join(output_dir, f"{img_id}.txt"), 'w') as f:
            for annotation in annotations:
                lbl_id = annotation['lbl_id'] - 1  # YOLO 클래스는 0부터 시작
                x, y, w, h = eval(annotation['annotations_info'])  # 문자열을 리스트로 변환
                
                # 중심 x, y 좌표 계산
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                
                # 너비와 높이를 이미지 크기로 나눠서 상대 크기로 변환
                width = w / img_width
                height = h / img_height
                
                # YOLO 형식으로 데이터 작성
                f.write(f"{lbl_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 변환 함수 실행
convert_to_yolo_format(data)
