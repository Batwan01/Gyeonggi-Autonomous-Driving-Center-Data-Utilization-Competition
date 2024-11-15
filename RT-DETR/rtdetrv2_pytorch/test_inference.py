import torch
import torchvision.transforms as T
import os
import json
import time
from PIL import Image
from src.core import YAMLConfig

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

def inference(model, image_path, device):
    """
    RT-DETR 모델을 사용하여 추론
    """
    model.eval()
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    im_data = transforms(im_pil)[None].to(device)

    with torch.no_grad():
        output = model(im_data, orig_size)

    labels, boxes, scores = output
    return labels, boxes, scores


def main(args):

    start_time = time.time()
    # 모델 및 설정 로드
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')

    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    # Deploy 모드로 전환
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # test 이미지 폴더에서 모든 파일 처리
    test_folder = args.test_folder
    output_json = []
    annotation_id = 1

    for image_file in os.listdir(test_folder):
        print(f"{image_file}")
        image_path = os.path.join(test_folder, image_file)
        img_id = os.path.splitext(image_file)[0]

        # 추론 실행
        labels, boxes, scores = inference(model, image_path, args.device)

        labels = labels.squeeze()
        boxes = boxes.squeeze()  
        scores = scores.squeeze()
    
        # 이미지 정보
        image_info = {
            "images": {
                "img_id": img_id,
                "img_width": 1920,  # 실제 이미지 크기를 사용할 수도 있음
                "img_height": 1200
            },
            "annotations": []
        }

        # 각 객체에 대한 주석(annotation) 정보 추가
        for i, box in enumerate(boxes):
            if scores[i].item() < args.confidence_threshold:
                continue

            cls = int(labels[i].item()) + 1
            conf = scores[i].item()
            x_min, y_min, x_max, y_max = box.tolist()
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

    output_dir = os.path.dirname(args.output_path)  # 출력 경로에서 디렉토리 추출
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 디렉토리 생성

    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, indent=4, ensure_ascii=False)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"추론 결과가 {args.output_path}에 저장되었습니다.")
    print(f"소요 시간 {inference_time}에 저장되었습니다.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/home/hwang/leem/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml", help='모델 설정 파일 경로')
    parser.add_argument('-r', '--resume', type=str, default="/home/hwang/leem/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/RT-DETR/rtdetrv2_pytorch/rtdetrv2_m_dsp_finetunning/rtdetrv2_m_dsp.pth", help='모델 가중치 파일 경로')
    parser.add_argument('-t', '--test-folder', type=str, default="/hdd1/lim_data/2d_cctv/test", help='테스트 이미지 폴더 경로')
    parser.add_argument('-o', '--output-path', type=str, default='./results/output.json', help='결과 저장 JSON 파일 경로')
    parser.add_argument('-d', '--device', type=str, default='cuda:1', help='사용할 디바이스 (예: cuda:0, cpu)')
    parser.add_argument('--confidence-threshold', type=float, default=0.25, help='결과를 필터링할 confidence score 기준')
    args = parser.parse_args()

    main(args)