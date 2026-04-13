#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop classification ROIs from PASCAL VOC annotations.

Features:
- Read Pascal VOC XML annotations
- Crop only target labels
- Expand bbox by a given ratio (default: 1.25)
- Keep aspect ratio
- Resize longest side to target size (default: 224)
- Pad to square image (default: 224x224)
- Export as classification dataset:
    output_dir/
      class_a/
      class_b/
- Save metadata.csv for traceability

Example:
    python voc_to_cls.py \
        --images /path/to/images \
        --annotations /path/to/annotations \
        --output /path/to/output \
        --labels screwdriver left_hand right_hand \
        --expand-ratio 1.25 \
        --size 224 \
        --min-size 32
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageOps


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC object crops into a classification dataset."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory containing Pascal VOC XML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for classification dataset.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Target object labels to crop. Example: --labels screwdriver left_hand",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=1.25,
        help="BBox expansion ratio. Default: 1.25",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Final square output size. Default: 224",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum original bbox width/height threshold. Objects smaller than this are skipped. Default: 32",
    )
    parser.add_argument(
        "--image-exts",
        nargs="*",
        default=None,
        help="Optional image extensions to search, e.g. .jpg .png",
    )
    parser.add_argument(
        "--pad-color",
        nargs=3,
        type=int,
        default=[0, 0, 0],
        metavar=("R", "G", "B"),
        help="Padding color, default: 0 0 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents.",
    )
    parser.add_argument(
        "--copy-empty-classes",
        action="store_true",
        help="Create class folders even if no crop is found.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.images.is_dir():
        raise FileNotFoundError(f"--images directory not found: {args.images}")
    if not args.annotations.is_dir():
        raise FileNotFoundError(f"--annotations directory not found: {args.annotations}")
    if args.expand_ratio < 1.0:
        raise ValueError("--expand-ratio must be >= 1.0")
    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.min_size < 1:
        raise ValueError("--min-size must be >= 1")

    if args.image_exts is not None and len(args.image_exts) > 0:
        normalized = set()
        for ext in args.image_exts:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = "." + ext
            normalized.add(ext)
        args.image_exts = normalized
    else:
        args.image_exts = IMG_EXTENSIONS


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        for item in output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                # Safe recursive delete without shutil.rmtree to keep control explicit.
                for sub in sorted(item.rglob("*"), reverse=True):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                    elif sub.is_dir():
                        sub.rmdir()
                item.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)


def find_image_for_xml(xml_path: Path, images_dir: Path, allowed_exts: set[str]) -> Optional[Path]:
    """
    Try to match image by XML filename stem.
    Example: sample001.xml -> sample001.jpg / png / ...
    """
    stem = xml_path.stem
    for ext in allowed_exts:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: try case-insensitive scan if direct lookup fails.
    for path in images_dir.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in allowed_exts:
            return path

    return None


def parse_voc_objects(xml_path: Path) -> Tuple[Optional[str], List[dict]]:
    """
    Return:
        filename_from_xml: optional string
        objects: list of dict(name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    filename_from_xml = filename_node.text.strip() if filename_node is not None and filename_node.text else None

    objects = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_node is None or bndbox is None or not name_node.text:
            continue

        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
        except (AttributeError, TypeError, ValueError):
            continue

        objects.append(
            {
                "name": name_node.text.strip(),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return filename_from_xml, objects


def clamp_bbox(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> Tuple[int, int, int, int]:
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width - 1))
    ymax = max(0, min(ymax, height - 1))
    return xmin, ymin, xmax, ymax


def expand_bbox(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_w: int,
    img_h: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox around center by scale ratio.
    """
    box_w = xmax - xmin
    box_h = ymax - ymin

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    new_w = box_w * expand_ratio
    new_h = box_h * expand_ratio

    new_xmin = int(math.floor(cx - new_w / 2.0))
    new_ymin = int(math.floor(cy - new_h / 2.0))
    new_xmax = int(math.ceil(cx + new_w / 2.0))
    new_ymax = int(math.ceil(cy + new_h / 2.0))

    new_xmin, new_ymin, new_xmax, new_ymax = clamp_bbox(
        new_xmin, new_ymin, new_xmax, new_ymax, img_w, img_h
    )

    return new_xmin, new_ymin, new_xmax, new_ymax


def resize_longest_side_and_pad(
    image: Image.Image,
    target_size: int,
    pad_color: Tuple[int, int, int],
) -> Image.Image:
    """
    Keep aspect ratio, resize longest side to target_size,
    then pad to target_size x target_size.
    """
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid crop size.")

    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    padded = ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=pad_color,
    )
    return padded


def safe_crop(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    xmin, ymin, xmax, ymax = bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox after expansion/clamp: {bbox}")
    return image.crop((xmin, ymin, xmax, ymax))


def make_output_filename(
    image_stem: str,
    label: str,
    obj_index: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> str:
    return f"{image_stem}__{label}__{obj_index:03d}__{xmin}_{ymin}_{xmax}_{ymax}.jpg"


def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output, args.overwrite)

    target_labels = set(args.labels)

    if args.copy_empty_classes:
        for label in target_labels:
            (args.output / label).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(args.annotations.glob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found in: {args.annotations}", file=sys.stderr)
        return 1

    metadata_rows = []
    saved_count_by_class = defaultdict(int)
    skipped_count = 0
    missing_image_count = 0

    for xml_path in xml_files:
        try:
            filename_from_xml, objects = parse_voc_objects(xml_path)
        except Exception as exc:
            print(f"[WARN] Failed to parse XML: {xml_path} | {exc}", file=sys.stderr)
            skipped_count += 1
            continue

        image_path = None

        # First try XML filename field if present.
        if filename_from_xml:
            candidate = args.images / filename_from_xml
            if candidate.exists() and candidate.suffix.lower() in args.image_exts:
                image_path = candidate

        # Fallback by stem match.
        if image_path is None:
            image_path = find_image_for_xml(xml_path, args.images, args.image_exts)

        if image_path is None:
            print(f"[WARN] No matching image found for XML: {xml_path.name}", file=sys.stderr)
            missing_image_count += 1
            continue

        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                img_w, img_h = image.size

                obj_counter = 0
                for obj in objects:
                    label = obj["name"]
                    if label not in target_labels:
                        continue

                    xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
                    box_w = xmax - xmin
                    box_h = ymax - ymin

                    if box_w < args.min_size or box_h < args.min_size:
                        skipped_count += 1
                        continue

                    exp_xmin, exp_ymin, exp_xmax, exp_ymax = expand_bbox(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        img_w=img_w,
                        img_h=img_h,
                        expand_ratio=args.expand_ratio,
                    )

                    try:
                        crop = safe_crop(image, (exp_xmin, exp_ymin, exp_xmax, exp_ymax))
                    except Exception as exc:
                        print(
                            f"[WARN] Failed crop on {image_path.name}, label={label}, bbox={exp_xmin, exp_ymin, exp_xmax, exp_ymax} | {exc}",
                            file=sys.stderr,
                        )
                        skipped_count += 1
                        continue

                    try:
                        processed = resize_longest_side_and_pad(
                            crop,
                            target_size=args.size,
                            pad_color=tuple(args.pad_color),
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] Failed resize/pad on {image_path.name}, label={label} | {exc}",
                            file=sys.stderr,
                        )
                        skipped_count += 1
                        continue

                    class_dir = args.output / label
                    class_dir.mkdir(parents=True, exist_ok=True)

                    out_name = make_output_filename(
                        image_stem=image_path.stem,
                        label=label,
                        obj_index=obj_counter,
                        xmin=exp_xmin,
                        ymin=exp_ymin,
                        xmax=exp_xmax,
                        ymax=exp_ymax,
                    )
                    out_path = class_dir / out_name
                    processed.save(out_path, quality=95)

                    metadata_rows.append(
                        {
                            "source_xml": str(xml_path),
                            "source_image": str(image_path),
                            "label": label,
                            "orig_xmin": xmin,
                            "orig_ymin": ymin,
                            "orig_xmax": xmax,
                            "orig_ymax": ymax,
                            "expanded_xmin": exp_xmin,
                            "expanded_ymin": exp_ymin,
                            "expanded_xmax": exp_xmax,
                            "expanded_ymax": exp_ymax,
                            "output_path": str(out_path),
                        }
                    )

                    saved_count_by_class[label] += 1
                    obj_counter += 1

        except Exception as exc:
            print(f"[WARN] Failed to open image: {image_path} | {exc}", file=sys.stderr)
            missing_image_count += 1
            continue

    metadata_path = args.output / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_xml",
                "source_image",
                "label",
                "orig_xmin",
                "orig_ymin",
                "orig_xmax",
                "orig_ymax",
                "expanded_xmin",
                "expanded_ymin",
                "expanded_xmax",
                "expanded_ymax",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n=== Done ===")
    print(f"XML files scanned      : {len(xml_files)}")
    print(f"Missing images         : {missing_image_count}")
    print(f"Skipped objects        : {skipped_count}")
    print(f"Metadata saved         : {metadata_path}")
    print("Saved crops by class   :")
    if saved_count_by_class:
        for label in sorted(saved_count_by_class):
            print(f"  - {label}: {saved_count_by_class[label]}")
    else:
        print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())