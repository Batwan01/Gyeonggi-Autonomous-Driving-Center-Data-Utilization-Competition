from ultralytics import YOLO
import os
import json
import time

# 양자화된 모델이 저장된 경로
quantized_model_path = '/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/quantized_weight/quantized_yolo11s_2.pt'

# YOLO 모델 불러오기
model = YOLO(quantized_model_path)
model.to('cuda').half()
# model.export(format="onnx")

# 추론할 이미지 폴더 경로 설정
image_folder_path = "/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/test"  # 실제 이미지 폴더 경로로 수정

# 클래스 매핑 정보
class_mapping = {
    1: 'bus',
    2: 'car',
    3: 'sign',
    4: 'truck',
    5: 'human',
    6: 'special_vehicles',
    7: 'taxi',
    8: 'motorcycle'
}

# COCO 형식 JSON 구조 초기화
output_json = []
annotation_id = 1  # annotation ID 초기화

# 소요 시간 측정 시작
start_time = time.time()

# 이미지 폴더 내 모든 이미지 파일에 대해 추론 수행
for image_name in os.listdir(image_folder_path):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일 필터링
        image_path = os.path.join(image_folder_path, image_name)

        # 추론 수행
        results = model(image_path)

        for result in results:
            # 이미지 파일 이름에서 숫자 추출하여 img_id로 사용
            img_id = os.path.splitext(os.path.basename(result.path))[0]
            image_info = {
                "images": {
                    "img_id": img_id,
                    "img_width": 1920,  # 실제 이미지 너비로 수정 필요
                    "img_height": 1200  # 실제 이미지 높이로 수정 필요
                },
                "annotations": []
            }

            # 각 객체에 대한 주석(annotation) 정보 추가
            for box in result.boxes:
                cls = int(box.cls.item()) + 1  # 클래스 ID
                conf = box.conf.item()  # 신뢰도 점수
                x_center, y_center, width, height = box.xywh[0].tolist()  # YOLO 좌표
                x_min = x_center - (width / 2)
                y_min = y_center - (height / 2)

                # COCO 형식의 bbox 정보 생성
                annotation = {
                    "annotations_id": f"{img_id}{annotation_id:04d}",
                    "lbl_id": cls,
                    "lbl_nm": class_mapping.get(cls, "unknown"),
                    "annotations_info": f"[{x_min:.2f}, {y_min:.2f}, {width:.2f}, {height:.2f}]",
                    "confidence": conf
                }
                image_info["annotations"].append(annotation)
                annotation_id += 1

            output_json.append(image_info)


# JSON 파일로 저장
output_path = "/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/output_results.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_json, json_file, indent=4, ensure_ascii=False)

# 소요 시간 측정 종료 및 출력
elapsed_time = time.time() - start_time
print(f"전체 예측 작업 소요 시간: {elapsed_time:.2f}초")
print(f"추론 결과가 JSON 파일로 저장되었습니다: {output_path}")
