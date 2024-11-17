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

    print(f"추론 결과가 {args.output_path}에 저장되었습니다.")
    print(f"소요 시간 {inference_time}에 저장되었습니다.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--onnx-file', type=str, default="./final_onnxes/rtdetrv2_s_dsp_36.onnx", help='ONNX 모델 파일 경로')
    parser.add_argument('-t', '--test-folder', type=str, default="../../dataset/test", help='테스트 이미지 폴더 경로')
    parser.add_argument('-r', '--output-path', type=str, default='./json/rtdetrv2_s_dsp_36_onnx_2.json', help='결과 저장 JSON 파일 경로')
    args = parser.parse_args()

    main(args)