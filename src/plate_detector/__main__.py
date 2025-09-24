import argparse
import cv2
from .detector import detect_plate_boxes, extract_plate_roi

def main():
    parser = argparse.ArgumentParser(description="Simple license plate detector")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    boxes, dbg, _ = detect_plate_boxes(img, debug=True)

    for i, b in enumerate(boxes):
        roi = extract_plate_roi(img, b)
        cv2.imwrite(f"plate_{i}.png", roi)

    cv2.imwrite("debug_boxes.jpg", dbg)
    print(f"Found {len(boxes)} plate(s). Saved ROIs + debug image.")
