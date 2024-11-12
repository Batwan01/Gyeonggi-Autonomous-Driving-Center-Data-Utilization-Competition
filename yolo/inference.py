import json
import time
import torch
import os
from ultralytics import YOLO
import re

start_time1 = time.time()
torch.cuda.empty_cache()

# 파일 이름에서 숫자 추출 함수
def extract_img_id(file_name):
    return re.search(r'\d+', file_name).group()

# YOLO 모델 불러오기
config = "yolo_11s_1024"  # 모델 설정에 맞게 수정
pt = "best.pt"  # 사용할 모델의 가중치 파일 이름
model = YOLO(f"/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/weights/{config}_{pt}")

# 테스트 이미지 폴더 경로
test_folder = "/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/test"

# 클래스 infomations
category_mapping = {
    1: "bus",
    2: "car",
    3: "sign",
    4: "truck",
    5: "human",
    6: "special_vehicles",
    7: "taxi",
    8: "motorcyle"
    # Add all necessary mappings here for your dataset
}

# COCO 형식 JSON 구조 초기화
output_json = []
annotation_id = 1

# 추론 실행
start_time2 = time.time()
results = model.predict(source=test_folder, stream=True)
start_time3 = time.time()
# 결과 처리
for result in results:
    # 이미지 파일 이름에서 숫자 추출하여 img_id로 사용
    img_id = os.path.splitext(os.path.basename(result.path))[0]
    image_info = {
        "images": {
            "img_id": img_id,
            "img_width": 1920,
            "img_height": 1200
        },
        "annotations": []
    }

    # 각 객체에 대한 주석(annotation) 정보 추가
    for box in result.boxes:
        cls = int(box.cls.item())+1  # 클래스 ID
        conf = box.conf.item()     # 신뢰도 점수
        x_center, y_center, width, height = box.xywh[0].tolist()  # YOLO 좌표
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)

        # COCO 형식의 bbox 정보 생성
        annotation = {
            "annotations_id": f"{img_id}{annotation_id:04d}",
            "lbl_id": cls,
            "lbl_nm": category_mapping.get(cls, "unknown"),  
            "annotations_info": f"[{x_min:.2f}, {y_min:.2f}, {width:.2f}, {height:.2f}]",
            "confidence": conf
        }
        image_info["annotations"].append(annotation)
        annotation_id += 1

    output_json.append(image_info)

# JSON 파일로 저장
output_path = f"/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/inference_json/inference_{config}.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_json, json_file, indent=4, ensure_ascii=False)

end_time = time.time()
print(f"코드 처음부터 끝: {end_time - start_time1:.2f} 초")
print(f"model.predict() 시간: {start_time3 - start_time2:.2f} 초")
print(f"추론 완료 후 json 생성 시간 : {end_time - start_time3:.2f} 초")

print(f"inference 소요시간 : {end_time - start_time2:.2f} 초")
print(f"결과가 {output_path}에 저장되었습니다.")