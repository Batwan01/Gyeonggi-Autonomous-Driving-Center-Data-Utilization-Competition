import json
import time
import torch
import os
from ultralytics import YOLO
import re
import numpy as np
from collections import Counter


def main():
    # 클래스 ID와 이름 매핑 규칙
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

    # 결과 정리용 리스트
    results_summary = []

    # 모델 파일들이 저장된 폴더와 test 데이터셋 input 경로, 결과 저장 폴더 설정
    model_folder = args.weigths_path
    # 모델 파일 목록
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]


    for model_file in model_files:
        weight_path = os.path.join(model_folder, model_file)
        output_path = f"{os.path.splitext(model_file)[0]}.json"

        # 소요 시간 측정 시작
        start_time = time.time()
        # test 데이터셋 input 경로
        test_folder = args.test_path
        model = YOLO(weight_path) # weights 경로

        # COCO 형식 JSON 구조 초기화
        output_json = []
        annotation_id = 1

        # 추론 실행

        results = model.predict(source=test_folder, stream=True)

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
                    "lbl_nm": class_mapping.get(cls, "unknown"),
                    "annotations_info": f"[{x_min:.2f}, {y_min:.2f}, {width:.2f}, {height:.2f}]",
                    "confidence": conf
                }
                image_info["annotations"].append(annotation)
                annotation_id += 1

            output_json.append(image_info)

        # JSON 파일로 저장
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(output_json, json_file, indent=4, ensure_ascii=False)

        # 소요 시간 측정 종료 및 출력
        elapsed_time = time.time() - start_time
        print(f"전체 예측 작업 소요 시간: {elapsed_time:.2f}초")

        results_summary.append({
            "model_file": model_file,
            "elapsed_time": elapsed_time,
        })
    # 최종 결과 요약 출력
    print("\n=== 모델별 결과 요약 ===")
    for result in results_summary:
        print(f"모델: {result['model_file']}, 소요 시간: {result['elapsed_time']:.2f}초")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weigths_path', type=str, default=r"../final_weights", help='가중치 폴더 경로')
    parser.add_argument('-t', '--test_path', type=str, default=r"../../dataset/test", help='test 이미지 경로')
    parser.add_argument('-g', '--gt_path', type=str, default=r"/root/evaluation/test_1730796030678.json", help='gt json 파일 경로')
    args = parser.parse_args()

    main(args)