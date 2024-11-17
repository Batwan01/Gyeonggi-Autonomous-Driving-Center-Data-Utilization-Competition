from ultralytics import YOLO
import os


def export_single_model_to_onnx(model_path, output_dir, imgsz):
    """
    Export a single YOLO model to ONNX format.

    Args:
        model_path (str): Path to the YOLO .pt weight file.
        output_dir (str): Directory to save the ONNX model.
        imgsz (int): Input image size (e.g., 640, 1024).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting model: {model_path} with input size: {imgsz}x{imgsz}")

    try:
        # Load YOLO model
        model = YOLO(model_path)

        # Set input size
        model.overrides['imgsz'] = imgsz

        # Export to ONNX
        export_path = model.export(format="onnx")

        # Move ONNX file to the output directory
        onnx_filename = os.path.basename(export_path)
        target_path = os.path.join(output_dir, onnx_filename)
        os.rename(export_path, target_path)

        print(f"ONNX model saved to {target_path}")

    except Exception as e:
        print(f"Failed to convert {model_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a single YOLO model to ONNX format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO .pt weight file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the ONNX model.")
    parser.add_argument("--imgsz", type=int, required=True, help="Input image size (e.g., 640, 1024).")
    args = parser.parse_args()

    export_single_model_to_onnx(args.model_path, args.output_dir, args.imgsz)