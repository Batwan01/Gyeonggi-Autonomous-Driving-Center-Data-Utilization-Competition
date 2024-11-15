import json
import time
import torch
import os
from ultralytics import YOLO
import re
import numpy as np
from collections import Counter

# Ground Truth 데이터 불러오기
with open(r"/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/val_split.json", 'r') as f:
    gt = json.load(f)

ap_file = open("ap_info.txt", "w", encoding="utf-8")

# Ground Truth를 이미지 ID별로 정리
gt_list = []
for g in range(len(gt)):
    for n in range(len(gt[g]['annotations'])):
        gt_list.append([gt[g]['images']['img_id'],
                        gt[g]['annotations'][n]['lbl_nm'],
                        eval(gt[g]['annotations'][n]['annotations_info'])])

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
model_folder = r"/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/ttt"
# 모델 파일 목록
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]


for model_file in model_files:
    weight_path = os.path.join(model_folder, model_file)
    output_path = f"{os.path.splitext(model_file)[0]}.json"

    # 소요 시간 측정 시작
    start_time = time.time()

    # test 데이터셋 input 경로
    test_folder = r"/data/ephemeral/home/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolo/val"

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

    # ========== mAP 계산 ==========
    IOUThreshold = 0.5
    pr_list = []
    with open(output_path, 'r') as f:
        pr = json.load(f)

    # 예측 결과 리스트 생성
    for p in range(len(pr)):
        for n in range(len(pr[p]['annotations'])):
            pr_list.append([pr[p]['images']['img_id'],
                            pr[p]['annotations'][n]['lbl_nm'],
                            eval(pr[p]['annotations'][n]['annotations_info']),
                            float(pr[p]['annotations'][n]['confidence'])])

    pr_list = sorted(pr_list, key=lambda x: x[3], reverse=True)

    # mAP 계산 변수 초기화
    ap = 0
    for c in class_mapping.values():
        pr_list_t = [x for x in pr_list if x[1] == c]
        gt_list_t = [x for x in gt_list if x[1] == c]
        npos = len(gt_list_t)
        tp = np.zeros(len(pr_list_t))
        fp = np.zeros(len(pr_list_t))

        det = Counter(cc[0] for cc in gt_list_t)
        for key, val in det.items():
            det[key] = np.zeros(val)

        for i in range(len(pr_list_t)):
            area_pr = pr_list_t[i][2][2] * pr_list_t[i][2][3]
            gt = [gt for gt in gt_list_t if gt[0] == pr_list_t[i][0]]
            iouMax = 0
            jmax = 0
            for j in range(len(gt)):
                w_t = min(pr_list_t[i][2][0] + pr_list_t[i][2][2], gt[j][2][0] + gt[j][2][2]) - max(pr_list_t[i][2][0], gt[j][2][0])
                h_t = min(pr_list_t[i][2][1] + pr_list_t[i][2][3], gt[j][2][1] + gt[j][2][3]) - max(pr_list_t[i][2][1], gt[j][2][1])
                if w_t < 0 or h_t < 0:
                    continue
                area_u = w_t * h_t
                area_gt = gt[j][2][2] * gt[j][2][3]
                iou1 = area_u / (area_pr + area_gt - area_u)

                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j
                elif iou1 == iouMax:
                    if det[pr_list_t[i][0]][jmax] == 1:
                        jmax = j
            if iouMax >= IOUThreshold:
                if det[pr_list_t[i][0]][jmax] == 0:
                    tp[i] = 1
                    det[pr_list_t[i][0]][jmax] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        acc_FP = np.cumsum(fp)
        acc_TP = np.cumsum(tp)

        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        mrec = [e for e in rec]
        mpre = [e for e in prec]

        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp, recallValid = [], []

        for r in recallValues:
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])

            recallValid.append(r)
            rhoInterp.append(pmax)

        ap += sum(rhoInterp) / 11
        print(f"Class '{c}' AP: {sum(rhoInterp) / 11}")
        ap_file.write(f"Class '{c}' AP: {sum(rhoInterp) / 11}\n")
        ap_file.write("-----------------------------------")

    # 평균 AP (mAP) 계산 및 출력
    mAP = ap / len(class_mapping)
    print(f"모델 '{model_file}'에 대한 mAP: {mAP:.4f}")
    results_summary.append({
        "model_file": model_file,
        "elapsed_time": elapsed_time,
        "mAP": mAP
    })

# 최종 결과 요약 출력
ap_file.close()
print("\n=== 모델별 결과 요약 ===")
for result in results_summary:
    print(f"모델: {result['model_file']}, 소요 시간: {result['elapsed_time']:.2f}초, mAP: {result['mAP']:.4f}")
