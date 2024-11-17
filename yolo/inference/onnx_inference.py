import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import time


def infer_onnx_model(model_path, test_folder, output_dir):
    print(f"Processing model: {model_path}")
    
    # ONNX 모델 로드
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    # 모델의 기대 입력 크기 추출
    input_shape = session.get_inputs()[0].shape  # (batch, channels, height, width)
    height, width = input_shape[2], input_shape[3]
    print(f"Model expects input shape: {height}x{width}")

    # COCO 형식 JSON 구조 초기화
    output_json = []
    annotation_id = 1

    # Test 폴더 내 이미지 추론
    for image_file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_file)
        img_id = os.path.splitext(os.path.basename(image_file))[0]
        im_pil = Image.open(image_path).convert('RGB')

        # 이미지 전처리 (모델 기대 크기로 리사이즈)
        im_resized = im_pil.resize((width, height))
        image_tensor = np.asarray(im_resized, dtype=np.float32) / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))  # HWC -> CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # 배치 차원 추가

        # ONNX 추론 실행
        outputs = session.run(output_names, {input_name: image_tensor})
        output_tensor = outputs[0]  # 단일 텐서

        # 출력 텐서 구조 확인 및 분리
        num_boxes = output_tensor.shape[2]  # 예측 박스 개수
        boxes = output_tensor[0, :4, :]  # 첫 4개는 박스 좌표 (x_min, y_min, x_max, y_max)
        scores = output_tensor[0, 4:, :]  # 나머지는 클래스 점수

        # 이미지 정보 초기화
        image_info = {
            "images": {
                "img_id": img_id,
                "img_width": im_pil.width,
                "img_height": im_pil.height
            },
            "annotations": []
        }

        # 각 객체에 대한 주석(annotation) 정보 추가
        for i in range(num_boxes):  # 추론 결과 처리
            x_min, y_min, x_max, y_max = boxes[:, i]
            width = x_max - x_min
            height = y_max - y_min
            conf = float(scores[:, i].max())  # 가장 높은 클래스 점수
            if conf < 0.5:  # Confidence threshold
                continue

            annotation = {
                "annotations_id": f"{img_id}{annotation_id:04d}",
                "bbox": [x_min, y_min, width, height],
                "confidence": conf
            }
            image_info["annotations"].append(annotation)
            annotation_id += 1

        output_json.append(image_info)

    # JSON 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(model_path).replace('.onnx', '.json')}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def main(args):
    # 모델 디렉토리 내 모든 ONNX 모델 처리
    for model_file in os.listdir(args.onnx_dir):
        if model_file.endswith(".onnx"):
            model_path = os.path.join(args.onnx_dir, model_file)
            infer_onnx_model(model_path, args.test_folder, args.output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX inference for YOLO models")
    parser.add_argument("--onnx_dir", type=str, default="../final_onnxes", help="Directory containing ONNX models.")
    parser.add_argument("--test_folder", type=str, default="../../dataset/test", help="Directory containing test images.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save JSON results.")
    args = parser.parse_args()

    main(args)