import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
import json
from PIL import Image

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

# TensorRT inference helper
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 텐서 이름 및 동적 차원 처리
        self.input_tensors = {}
        self.output_tensors = {}
        for binding in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = self.engine.get_tensor_shape(binding)

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_tensors[binding] = {"shape": shape, "dtype": dtype}
            else:
                if -1 in shape:  # 동적 차원 처리
                    shape[0] = 1  # 배치 크기 고정
                self.output_tensors[binding] = {"shape": shape, "dtype": dtype}

        # 입력 및 출력 텐서 버퍼 생성
        self.device_buffers = {}
        for name, info in {**self.input_tensors, **self.output_tensors}.items():
            volume = trt.volume(info["shape"])
            self.device_buffers[name] = cuda.mem_alloc(volume * np.dtype(info["dtype"]).itemsize)

        self.bindings = [int(self.device_buffers[binding]) for binding in self.engine]

    def infer(self, image, orig_size):
        # 입력 데이터 준비 (연속 배열로 보장)
        input_data = {
            "images": np.ascontiguousarray(image, dtype=self.input_tensors["images"]["dtype"]),
            "orig_target_sizes": np.ascontiguousarray(orig_size, dtype=self.input_tensors["orig_target_sizes"]["dtype"]),
        }

        # 입력 텐서 복사
        for name, data in input_data.items():
            tensor_info = self.input_tensors[name]
            cuda.memcpy_htod_async(self.device_buffers[name], data, self.stream)

        # 실행
        self.context.execute_v2(self.bindings)

        # 출력 텐서 복사 및 반환
        outputs = {}
        for name, tensor_info in self.output_tensors.items():
            host_buffer = np.empty(tensor_info["shape"], dtype=tensor_info["dtype"])
            cuda.memcpy_dtoh_async(host_buffer, self.device_buffers[name], self.stream)
            outputs[name] = host_buffer

        self.stream.synchronize()
        return outputs

def preprocess_image(image_path, input_shape=(640, 640)):
    im_pil = Image.open(image_path).convert('RGB')
    im_resized = im_pil.resize(input_shape)
    image_np = np.asarray(im_resized).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image_np, axis=0)

def main(args):
    start_time = time.time()
    trt_inference = TRTInference(args.engine_file)
    test_folder = args.test_folder
    output_json = []
    annotation_id = 1

    for image_file in os.listdir(test_folder):
        print(f"Processing: {image_file}")
        image_path = os.path.join(test_folder, image_file)
        img_id = os.path.splitext(image_file)[0]
        original_image = Image.open(image_path)
        w, h = original_image.size
        preprocessed_image = preprocess_image(image_path)
        orig_size = np.array([[w, h]], dtype=np.float32)

        try:
            outputs = trt_inference.infer(preprocessed_image, orig_size)
            labels, boxes, scores = outputs["labels"], outputs["boxes"], outputs["scores"]

            image_info = {
                "images": {
                    "img_id": img_id,
                    "img_width": w,
                    "img_height": h
                },
                "annotations": []
            }

            for i in range(len(labels)):
                cls = int(labels[i])
                conf = float(scores[i])
                x_min, y_min, x_max, y_max = map(float, boxes[i])
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

        except Exception as e:
            print(f"Inference failed for {image_file}: {str(e)}")

    # JSON 저장
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, indent=4, ensure_ascii=False)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"추론 결과가 {args.output_path}에 저장되었습니다.")
    print(f"소요 시간 {inference_time:.2f} 초")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine-file', type=str, default="./final_trts/rtdetrv2_s_dsp_36.trt", help='TensorRT 엔진 파일 경로')
    parser.add_argument('-t', '--test-folder', type=str, default="../../dataset/test", help='테스트 이미지 폴더 경로')
    parser.add_argument('-r', '--output-path', type=str, default='./json/rtdetrv2_s_dsp_36_tensorrt.json', help='결과 저장 JSON 파일 경로')
    args = parser.parse_args()

    main(args)