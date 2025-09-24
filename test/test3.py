import os
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from PIL import Image

# ---------- CONFIG ----------
HF_REPO = "keremberke/license-plate-object-detection"
HF_CONFIG = "full"  # or None
OUT_ROOT = Path("datasets/license_plate")  # where to write YOLO folders
NAMES = ["license-plate"]  # class names (edit if dataset has more)
# ----------------------------

def ensure_dirs(root: Path):
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def find_bboxes(example) -> List[List[float]]:
    """
    Returns list of [x, y, w, h] in COCO xywh.
    Handles common HF schemas: 'bboxes', or 'objects'={'bbox':...}.
    Values might be pixels or normalized (0..1) depending on dataset.
    """
    if "bboxes" in example and example["bboxes"] is not None:
        return example["bboxes"]
    if "objects" in example and example["objects"] and "bbox" in example["objects"]:
        return example["objects"]["bbox"]
    if "annotations" in example and example["annotations"]:
        # Some datasets store list of dicts with xywh
        bb = []
        for ann in example["annotations"]:
            if all(k in ann for k in ("x", "y", "width", "height")):
                bb.append([ann["x"], ann["y"], ann["width"], ann["height"]])
        return bb
    return []

def find_labels(example, default_cls: int = 0) -> List[int]:
    """
    Tries to get per-box class ids; falls back to a single class.
    """
    # Common patterns:
    if "category_ids" in example and example["category_ids"] is not None:
        return example["category_ids"]
    if "objects" in example and example["objects"] and "category" in example["objects"]:
        cats = example["objects"]["category"]
        # Map string names to 0..K-1 if needed. Here assume single class.
        return [0 for _ in cats]
    if "labels" in example and example["labels"] is not None:
        # already numeric ids
        return example["labels"]
    # Fallback: single class
    b = find_bboxes(example)
    return [default_cls] * len(b)

def to_yolo_line(b, img_w: int, img_h: int, cls_id: int) -> str:
    """
    Convert COCO xywh (top-left) → YOLO x_center y_center w h (normalized).
    If bbox seems to be in pixels (>1), normalize; if already normalized, keep.
    """
    x, y, w, h = b
    # Normalize if needed
    if max(x, y, w, h) > 1.0:
        x /= img_w; y /= img_h; w /= img_w; h /= img_h
    xc = x + w / 2.0
    yc = y + h / 2.0
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"

def dump_split(ds_split, split_name: str, out_root: Path):
    if split_name not in ("train", "validation", "val", "test"):
        raise ValueError(f"Unexpected split: {split_name}")

    img_dir = out_root / ("images/train" if split_name == "train"
                          else "images/val" if split_name in ("validation", "val")
                          else "images/test")
    lbl_dir = out_root / ("labels/train" if split_name == "train"
                          else "labels/val" if split_name in ("validation", "val")
                          else "labels/test")

    count = 0
    for i, ex in enumerate(ds_split):
        img: Image.Image = ex["image"]
        w, h = img.size

        stem = f"{split_name}_{i:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        # save image
        img.convert("RGB").save(img_path, quality=95)

        # save labels
        bboxes = find_bboxes(ex)
        cls_ids = find_labels(ex, default_cls=0)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for b, cid in zip(bboxes, cls_ids):
                f.write(to_yolo_line(b, w, h, int(cid)))
        count += 1

    print(f"[{split_name}] wrote {count} images to {img_dir}")

def write_yaml(root: Path, names: List[str], yaml_path: Optional[Path] = None):
    yaml_path = yaml_path or (root.parent / "data" / "license_plate.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Auto-generated for Ultralytics
path: {root.as_posix()}
train: images/train
val: images/val
test: images/test
names:
"""
    for i, n in enumerate(names):
        content += f"  {i}: {n}\n"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[yaml] wrote {yaml_path}")

def main():
    ensure_dirs(OUT_ROOT)
    ds = load_dataset(HF_REPO, name=HF_CONFIG, trust_remote_code=True)

    # dump known splits if present
    for s in ("train", "validation", "val", "test"):
        if s in ds:
            dump_split(ds[s], s, OUT_ROOT)

    write_yaml(OUT_ROOT, NAMES)

if __name__ == "__main__":
    main()
