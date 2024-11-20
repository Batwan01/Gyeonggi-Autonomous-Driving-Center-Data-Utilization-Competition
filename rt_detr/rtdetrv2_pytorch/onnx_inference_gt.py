import onnxruntime as ort
import torch
import torchvision.transforms as T
import json
import os
import time
from PIL import Image

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("ONNX Runtime Version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
print("Available providers:", ort.get_available_providers())

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

def inference_onnx(session, image_path, transforms):
    """
    ONNX Runtime으로 추론
    """
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).numpy()

    im_data = transforms(im_pil)[None].numpy()

    output = session.run(
        None,
        {'images': im_data, 'orig_target_sizes': orig_size}
    )
    print("Current Execution Providers:", session.get_providers())

    labels, boxes, scores = output
    return labels, boxes, scores

def main(args):


    with open(args.gt_path, 'r') as f:
        gt = json.load(f)

    # Ground Truth를 이미지 ID별로 정리
    gt_list = []
    for g in range(len(gt)):
        for n in range(len(gt[g]['annotations'])):
            gt_list.append([gt[g]['images']['img_id'],
                            gt[g]['annotations'][n]['lbl_nm'],
                            eval(gt[g]['annotations'][n]['annotations_info'])])


    start_time = time.time()
    # ONNX 세션 초기화
    session = ort.InferenceSession(
        args.onnx_file,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    test_folder = args.test_folder
    output_json = []
    annotation_id = 1

    for image_file in os.listdir(test_folder):
        print(f"Processing: {image_file}")
        image_path = os.path.join(test_folder, image_file)
        img_id = os.path.splitext(image_file)[0]

        # ONNX 추론 실행
        labels, boxes, scores = inference_onnx(session, image_path, transforms)

        labels = labels.squeeze()
        boxes = boxes.squeeze()
        scores = scores.squeeze()

        # COCO 형식으로 저장
        image_info = {
            "images": {
                "img_id": img_id,
                "img_width": 1920,
                "img_height": 1200
            },
            "annotations": []
        }

        for i, box in enumerate(boxes):
            cls = int(labels[i]) + 1
            conf = float(scores[i])  # numpy.float32를 Python float로 변환
            x_min, y_min, x_max, y_max = map(float, box)  # 박스 좌표도 float으로 변환
            width = x_max - x_min
            height = y_max - y_min

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

    # JSON 저장
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, indent=4, ensure_ascii=False)

    end_time = time.time()
    inference_time = end_time - start_time     

    # ========== mAP 계산 ==========
    IOUThreshold = 0.5
    pr_list = []
    with open(args_output_path, 'r') as f:
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

    mAP = ap / len(class_mapping)   

    print(f"모델: {args.onnx_file}, 소요 시간: {inference_time:.2f}초, mAP: {mAP:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--onnx_file', type=str, default="./final_onnxes/rtdetrv2_s_dsp_36.onnx", help='ONNX 모델 파일 경로')
    parser.add_argument('-t', '--test_folder', type=str, default="../../dataset/test", help='테스트 이미지 폴더 경로')
    parser.add_argument('-o', '--output_path', type=str, default='./json/rtdetrv2_s_dsp_36_onnx_2.json', help='결과 저장 JSON 파일 경로')
    parser.add_argument('-g', '--gt_path', type=str, default=r"/root/evaluation/test_1730796030678.json", help='gt json 파일 경로')
    args = parser.parse_args()

    main(args)