from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model("images/toyota example.jpg", save=True, conf=0.25)
#model = YOLO("weights/yoloplate.pt")  # your fine-tuned weights
