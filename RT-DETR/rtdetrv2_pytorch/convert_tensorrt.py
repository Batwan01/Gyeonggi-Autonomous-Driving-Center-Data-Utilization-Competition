import torch
import os
from src.core import YAMLConfig

def convert_to_tensorrt(model, input_tensor, onnx_path, engine_path):
    """
    모델을 TensorRT로 변환하여 엔진 파일로 저장
    """
    # 모델을 ONNX로 변환
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"ONNX 파일 저장 완료: {onnx_path}")

    # ONNX 파일을 TensorRT 엔진으로 변환
    command = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16"
    os.system(command)
    print(f"TensorRT 엔진 저장 완료: {engine_path}")

def main(args):
    # 모델 및 설정 로드
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location="cpu")

    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    # Deploy 모드로 전환
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            return self.model(images)

    model = Model().to(args.device)

    # 더미 입력 데이터 생성 (640x640 기준)
    dummy_input = torch.randn(1, 3, 640, 640).to(args.device)

    # 변환 경로 설정
    onnx_path = "model.onnx"
    engine_path = "model.trt"

    # TensorRT 변환 및 저장
    convert_to_tensorrt(model, dummy_input, onnx_path, engine_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml", help="모델 설정 파일 경로")
    parser.add_argument("-r", "--resume", type=str, default="./rtdetrv2_s_dsp_finetunning/best.pth", help="모델 가중치 파일 경로")
    parser.add_argument("-d", "--device", type=str, default="cuda:1", help="사용할 디바이스 (예: cuda:0, cpu)")
    args = parser.parse_args()

    main(args)