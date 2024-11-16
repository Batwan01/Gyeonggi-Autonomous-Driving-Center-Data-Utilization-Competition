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

def load_images_in_batches(folder, batch_size, device):
    """
    주어진 폴더에서 이미지를 배치 단위로 로드합니다.
    """
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    image_paths = []
    image_tensors = []
    original_sizes = []

    for file_name in os.listdir(folder):
        image_path = os.path.join(folder, file_name)
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        image_paths.append(image_path)
        image_tensors.append(transform(im_pil))
        original_sizes.append(orig_size)

        # 배치 크기만큼 쌓이면 반환
        if len(image_tensors) == batch_size:
            yield image_paths, torch.stack(image_tensors).to(device), torch.cat(original_sizes).to(device)
            image_paths, image_tensors, original_sizes = [], [], []

    # 남은 이미지 반환
    if image_tensors:
        yield image_paths, torch.stack(image_tensors).to(device), torch.cat(original_sizes).to(device)

def inference_batch(model, images, orig_sizes):
    """
    RT-DETR 모델을 사용하여 배치 추론
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images, orig_sizes)
    return outputs

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

    # 배치 추론
    test_folder = args.test_folder
    output_json = []
    annotation_id = 1
    batch_size = args.batch_size

    for image_paths, image_batch, orig_sizes in load_images_in_batches(test_folder, batch_size, args.device):
        outputs = inference_batch(model, image_batch, orig_sizes)
        labels_batch, boxes_batch, scores_batch = outputs

        for idx, image_path in enumerate(image_paths):
            img_id = os.path.splitext(os.path.basename(image_path))[0]
            print(f"{img_id}")
            labels = labels_batch[idx].squeeze()
            boxes = boxes_batch[idx].squeeze()
            scores = scores_batch[idx].squeeze()

            # 이미지 정보
            image_info = {
                "images": {
                    "img_id": img_id,
                    "img_width": 1920,
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
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, indent=4, ensure_ascii=False)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"추론 결과가 {args.output_path}에 저장되었습니다.")
    print(f"소요 시간 {inference_time}초")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml", help='모델 설정 파일 경로')
    parser.add_argument('-r', '--resume', type=str, default="./final_weights/rtdetr_s_dsp_72.pth", help='모델 가중치 파일 경로')
    parser.add_argument('-t', '--test-folder', type=str, default="../../yolo/test", help='테스트 이미지 폴더 경로')
    parser.add_argument('-o', '--output-path', type=str, default='./results/rtdetr_s_dsp_72.pth', help='결과 저장 JSON 파일 경로')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='사용할 디바이스 (예: cuda:0, cpu)')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    parser.add_argument('--confidence-threshold', type=float, default=0.25, help='결과를 필터링할 confidence score 기준')
    args = parser.parse_args()

    main(args)