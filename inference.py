"""
inference.py — PyTorch Image Classification Inference
======================================================
支援三種推論模式：
  1. single   : 單張圖片
  2. batch    : 整個資料夾
  3. yolo     : 讀取 YOLO txt 偵測結果，自動 crop 後分類

用法範例
--------
# 單張圖片
python inference.py single --checkpoint best.pth --image img.jpg

# 批次推論整個資料夾
python inference.py batch --checkpoint best.pth --input-dir images/ --output-csv result.csv

# YOLO crop 模式（每張圖搭配同名 .txt 偵測框）
python inference.py yolo --checkpoint best.pth --image-dir images/ --label-dir labels/ --output-csv result.csv
"""

import sys
import csv
import json
import math
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

try:
    import timm
except ImportError:
    sys.exit("請先安裝 timm：pip install timm")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model aliases（與 train.py 保持一致）
# ──────────────────────────────────────────────
MODEL_ALIASES: Dict[str, str] = {
    "mobilenetv3":        "mobilenetv3_large_100",
    "mobilenetv3_large":  "mobilenetv3_large_100",
    "mobilenetv3_small":  "mobilenetv3_small_100",
    "efficientnetv2":     "efficientnetv2_s",
    "efficientnetv2_s":   "efficientnetv2_s",
    "efficientnetv2_m":   "efficientnetv2_m",
    "efficientnetv2_l":   "efficientnetv2_l",
    "efficientvit_m0":    "efficientvit_m0",
    "efficientvit_m1":    "efficientvit_m1",
    "efficientvit_m2":    "efficientvit_m2",
    "efficientvit_m3":    "efficientvit_m3",
    "efficientvit_m4":    "efficientvit_m4",
    "efficientvit_m5":    "efficientvit_m5",
    "convnext":           "convnext_tiny",
    "convnext_tiny":      "convnext_tiny",
    "convnext_small":     "convnext_small",
    "convnext_base":      "convnext_base",
    "resnet":             "resnet50",
    "resnet18":           "resnet18",
    "resnet34":           "resnet34",
    "resnet50":           "resnet50",
    "resnet101":          "resnet101",
    "resnet152":          "resnet152",
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def resolve_model_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    return MODEL_ALIASES.get(key, name)


# ──────────────────────────────────────────────
# Checkpoint & model loading
# ──────────────────────────────────────────────
def load_model(
    checkpoint_path: str,
    device: torch.device,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    img_size: Optional[int] = None,
) -> Tuple[nn.Module, List[str], int]:
    """
    載入 checkpoint，回傳 (model, class_names, img_size)。

    class_names 優先順序：
      1. checkpoint 內的 class_names
      2. checkpoint 同目錄下的 classes.json / classes.txt
      3. 自動產生 ["class_0", "class_1", ...]
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    log.info(f"  載入 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── 從 checkpoint 取得訓練時的 args ──
    saved_args: dict = ckpt.get("args", {})
    _model_name   = model_name  or saved_args.get("model")
    _img_size     = img_size    or saved_args.get("img_size", 224)
    _num_classes  = num_classes

    if _model_name is None:
        raise ValueError(
            "無法從 checkpoint 取得模型名稱，請用 --model 手動指定。"
        )

    # ── 從 model_state 推算 num_classes ──
    model_state = ckpt["model_state"]
    if _num_classes is None:
        # 嘗試從最後一層 weight shape 推算
        for key in reversed(list(model_state.keys())):
            if "weight" in key:
                _num_classes = model_state[key].shape[0]
                break
        if _num_classes is None:
            raise ValueError("無法推算 num_classes，請用 --num-classes 手動指定。")

    log.info(f"  模型: {_model_name}, 類別數: {_num_classes}, 輸入大小: {_img_size}")

    # ── 建立模型 ──
    timm_name = resolve_model_name(_model_name)
    model = timm.create_model(timm_name, pretrained=False, num_classes=_num_classes)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # ── 取得 class_names ──
    class_names: List[str] = ckpt.get("class_names", [])

    if not class_names:
        # 在 checkpoint 目錄旁尋找 classes.json / classes.txt
        for fname in ["classes.json", "classes.txt"]:
            cfile = ckpt_path.parent / fname
            if cfile.is_file():
                if fname.endswith(".json"):
                    class_names = json.loads(cfile.read_text())
                else:
                    class_names = [l.strip() for l in cfile.read_text().splitlines() if l.strip()]
                log.info(f"  類別清單: {cfile}")
                break

    if not class_names:
        class_names = [f"class_{i}" for i in range(_num_classes)]
        log.warning(
            "  未找到類別清單，使用自動命名 class_0 ~ class_N。\n"
            f"  可將 classes.json / classes.txt 放在 {ckpt_path.parent} 下。"
        )

    return model, class_names, _img_size


# ──────────────────────────────────────────────
# Transform（與 train.py val_tf 一致）
# ──────────────────────────────────────────────
def build_transform(img_size: int) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────
# VOC-style preprocessing（與 voc_to_cls.py 一致）
# 用於 yolo 模式，確保推論時的前處理與訓練資料一致
# ──────────────────────────────────────────────
def expand_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int,
    img_w: int, img_h: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    """將 bbox 以中心點為基準向外擴展 expand_ratio 倍，並 clamp 至圖片邊界。"""
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    new_w = (xmax - xmin) * expand_ratio
    new_h = (ymax - ymin) * expand_ratio
    new_xmin = max(0, int(math.floor(cx - new_w / 2.0)))
    new_ymin = max(0, int(math.floor(cy - new_h / 2.0)))
    new_xmax = min(img_w, int(math.ceil(cx + new_w / 2.0)))
    new_ymax = min(img_h, int(math.ceil(cy + new_h / 2.0)))
    return new_xmin, new_ymin, new_xmax, new_ymax


def resize_longest_side_and_pad(
    image: Image.Image,
    target_size: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """保持長寬比，將最長邊縮放至 target_size，再以 pad_color 填充為正方形。"""
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    pad_left   = (target_size - new_w) // 2
    pad_top    = (target_size - new_h) // 2
    pad_right  = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top
    return ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=pad_color,
    )


# ──────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_image(
    model: nn.Module,
    tf: transforms.Compose,
    image: Image.Image,
    device: torch.device,
    topk: int = 1,
) -> List[Tuple[int, float]]:
    """對單一 PIL Image 進行推論，回傳 [(class_idx, confidence), ...]。"""
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    topk   = min(topk, probs.shape[0])
    values, indices = probs.topk(topk)
    return [(int(idx), float(val)) for idx, val in zip(indices, values)]


# ──────────────────────────────────────────────
# YOLO label parser（YOLO normalized xywh）
# ──────────────────────────────────────────────
def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """
    解析 YOLO 格式 txt（class cx cy w h，全為相對座標）。
    回傳 [(class_id, x1, y1, x2, y2), ...]（絕對像素座標）。
    """
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        # 確保座標在圖像範圍內
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


# ──────────────────────────────────────────────
# Mode: single
# ──────────────────────────────────────────────
def run_single(args):
    device = _get_device(args.device)
    model, class_names, img_size = load_model(
        args.checkpoint, device, args.model, args.num_classes, args.img_size
    )
    tf    = build_transform(img_size)
    image = Image.open(args.image)

    results = predict_image(model, tf, image, device, topk=args.topk)

    print(f"\nImage : {args.image}")
    print(f"{'Rank':<6} {'Class':<30} {'Confidence':>10}")
    print("-" * 50)
    for rank, (idx, conf) in enumerate(results, 1):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"{rank:<6} {name:<30} {conf*100:>9.2f}%")


# ──────────────────────────────────────────────
# Mode: batch
# ──────────────────────────────────────────────
def run_batch(args):
    device = _get_device(args.device)
    model, class_names, img_size = load_model(
        args.checkpoint, device, args.model, args.num_classes, args.img_size
    )
    tf = build_transform(img_size)

    input_dir = Path(args.input_dir)
    images = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])
    if not images:
        log.warning(f"在 {input_dir} 中找不到圖片")
        return

    log.info(f"  共找到 {len(images)} 張圖片")

    rows = []
    for img_path in images:
        try:
            image   = Image.open(img_path)
            results = predict_image(model, tf, image, device, topk=args.topk)
            top1_idx, top1_conf = results[0]
            top1_name = class_names[top1_idx] if top1_idx < len(class_names) else f"class_{top1_idx}"

            all_preds = "|".join(
                f"{class_names[i] if i < len(class_names) else f'class_{i}'}:{c*100:.2f}%"
                for i, c in results
            )
            rows.append({
                "image":      str(img_path),
                "top1_class": top1_name,
                "top1_conf":  f"{top1_conf*100:.2f}%",
                "topk":       all_preds,
            })
            log.info(f"  {img_path.name:<40} → {top1_name} ({top1_conf*100:.1f}%)")
        except Exception as e:
            log.warning(f"  跳過 {img_path.name}: {e}")

    if args.output_csv:
        _write_csv(args.output_csv, rows)
        log.info(f"  結果已儲存至 {args.output_csv}")


# ──────────────────────────────────────────────
# Mode: yolo
# ──────────────────────────────────────────────
def run_yolo(args):
    """
    讀取 YOLO 偵測結果 txt，對每個 bbox crop 出 ROI 後進行分類。

    目錄結構預期：
      --image-dir  images/  (含 .jpg / .png …)
      --label-dir  labels/  (含同名 .txt，YOLO 格式)

    若 --save-crops 指定輸出目錄，crop 影像也會一併儲存。
    """
    device = _get_device(args.device)
    model, class_names, img_size = load_model(
        args.checkpoint, device, args.model, args.num_classes, args.img_size
    )
    tf = build_transform(img_size)

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    images    = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])

    if not images:
        log.warning(f"在 {image_dir} 中找不到圖片")
        return

    save_dir: Optional[Path] = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  共找到 {len(images)} 張圖片")

    rows = []
    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")
        try:
            image  = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
            boxes  = parse_yolo_label(label_path, img_w, img_h)

            if not boxes:
                log.info(f"  {img_path.name:<40} → 無偵測框，跳過")
                continue

            for det_idx, (yolo_cls, x1, y1, x2, y2) in enumerate(boxes):
                # ── 與 voc_to_cls.py 相同的前處理：expand → crop → resize+pad ──
                ex1, ey1, ex2, ey2 = expand_bbox(
                    x1, y1, x2, y2, img_w, img_h, args.expand_ratio
                )
                crop = image.crop((ex1, ey1, ex2, ey2))
                crop = resize_longest_side_and_pad(
                    crop,
                    target_size=img_size,
                    pad_color=tuple(args.pad_color),
                )

                if save_dir:
                    crop_name = f"{img_path.stem}_det{det_idx:03d}.jpg"
                    crop.save(save_dir / crop_name)

                results       = predict_image(model, tf, crop, device, topk=args.topk)
                top1_idx, top1_conf = results[0]
                top1_name = class_names[top1_idx] if top1_idx < len(class_names) else f"class_{top1_idx}"

                all_preds = "|".join(
                    f"{class_names[i] if i < len(class_names) else f'class_{i}'}:{c*100:.2f}%"
                    for i, c in results
                )
                rows.append({
                    "image":        str(img_path),
                    "det_index":    det_idx,
                    "yolo_class":   yolo_cls,
                    "bbox":         f"{x1},{y1},{x2},{y2}",
                    "top1_class":   top1_name,
                    "top1_conf":    f"{top1_conf*100:.2f}%",
                    "topk":         all_preds,
                })
                log.info(
                    f"  {img_path.name:<35} det[{det_idx}] bbox=({x1},{y1},{x2},{y2})"
                    f"  → {top1_name} ({top1_conf*100:.1f}%)"
                )

        except Exception as e:
            log.warning(f"  跳過 {img_path.name}: {e}")

    if args.output_csv:
        _write_csv(args.output_csv, rows)
        log.info(f"  結果已儲存至 {args.output_csv}")

    log.info(f"  共處理 {len(rows)} 個偵測框")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _get_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Classification Inference (timm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 共用參數 ──
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint",   required=True,
                        help="訓練產生的 best.pth / last.pth 路徑")
    common.add_argument("--device",       default="auto",
                        help="cuda / mps / cpu / auto")
    common.add_argument("--topk",         type=int, default=1,
                        help="顯示 Top-K 預測結果")
    common.add_argument("--model",        default=None,
                        help="覆寫 checkpoint 內的模型名稱（通常不需要設定）")
    common.add_argument("--img-size",     type=int, default=None,
                        help="覆寫輸入圖片大小（通常不需要設定）")
    common.add_argument("--num-classes",  type=int, default=None,
                        help="覆寫類別數（通常不需要設定）")

    sub = p.add_subparsers(dest="mode", required=True)

    # ── single ──
    ps = sub.add_parser("single", parents=[common],
                        help="對單張圖片推論")
    ps.add_argument("--image", required=True, help="輸入圖片路徑")

    # ── batch ──
    pb = sub.add_parser("batch", parents=[common],
                        help="對資料夾內所有圖片批次推論")
    pb.add_argument("--input-dir",   required=True, help="輸入圖片資料夾")
    pb.add_argument("--output-csv",  default=None,  help="結果輸出 CSV 路徑")

    # ── yolo ──
    py = sub.add_parser("yolo", parents=[common],
                        help="讀取 YOLO 偵測 txt，crop 後分類")
    py.add_argument("--image-dir",    required=True, help="圖片資料夾")
    py.add_argument("--label-dir",    required=True,
                    help="YOLO txt 標籤資料夾（與圖片同名，副檔名 .txt）")
    py.add_argument("--output-csv",   default=None,  help="結果輸出 CSV 路徑")
    py.add_argument("--save-crops",   default=None,
                    help="若指定，將每個前處理後的 crop 影像儲存到此目錄")
    py.add_argument("--expand-ratio", type=float, default=1.25,
                    help="BBox 擴展比例，須與 voc_to_cls.py 訓練時一致（預設 1.25）")
    py.add_argument("--pad-color",    nargs=3, type=int, default=[0, 0, 0],
                    metavar=("R", "G", "B"),
                    help="Padding 顏色，須與 voc_to_cls.py 訓練時一致（預設 0 0 0）")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dispatch = {"single": run_single, "batch": run_batch, "yolo": run_yolo}
    dispatch[args.mode](args)
