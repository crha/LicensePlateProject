from ultralytics import YOLO
from datasets import load_dataset

ds = load_dataset("keremberke/license-plate-object-detection", name="full")
example = ds['train'][0]

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolo11n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="datasets/data/license_plate.yaml", epochs=3, imgsz=640)

# Run inference with the yolo11n model on the 'bus.jpg' image
results = model("images/toyota example.jpg")
print(results)