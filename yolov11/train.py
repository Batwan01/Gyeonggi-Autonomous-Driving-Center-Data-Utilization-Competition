from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.yaml")  # build a new model from YAML
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11l.yaml").load("yolo11l.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/root/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/yolov11/data.yaml", epochs=25, imgsz=1280, batch=4)