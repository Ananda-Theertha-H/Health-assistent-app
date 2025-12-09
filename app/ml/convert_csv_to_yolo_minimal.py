# app/ml/convert_csv_to_yolo_from_json.py
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return (x_center / img_w, y_center / img_h, w / img_w, h / img_h)

def parse_json_annotations_field(val):
    try:
        return json.loads(val)
    except Exception:
        return None

def gather_rows(csv_path: str):
    rows = []
    with open(csv_path, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for r in reader:
            rows.append(r)
    return headers, rows

def convert(args):
    headers, rows = gather_rows(args.csv)
    images_root = Path(args.images_root or ".")
    out = Path(args.output)
    train_img_dir = out / "images" / "train"
    val_img_dir = out / "images" / "val"
    train_lbl_dir = out / "labels" / "train"
    val_lbl_dir = out / "labels" / "val"
    ensure_dir(train_img_dir); ensure_dir(val_img_dir); ensure_dir(train_lbl_dir); ensure_dir(val_lbl_dir)

    # detect layout
    # possible image columns: 'image','image_path','filename','file'
    image_cols = [c for c in headers if c.lower() in ("image","image_path","filename","file","img")]
    # possible annotation columns: 'annotations','annotation','bboxes','label','labels'
    ann_cols = [c for c in headers if c.lower() in ("annotations","annotation","bboxes","label","labels","annotations_json")]
    bbox_cols = None
    if not ann_cols:
        # see if per-row bbox layout (image,xmin,ymin,xmax,ymax,class)
        candidates = {"xmin","x_min","x1","left","ymin","y_min","y1","top","xmax","x_max","x2","right","ymax","y_max","x_center"}
        if any(h.lower().startswith("x") or h.lower() in candidates for h in headers):
            bbox_cols = headers

    # collect class names
    class_map: Dict[str,int] = {}
    next_id = 0

    # helper chooses train or val by simple split index
    def choose_split(idx, total):
        if args.val_frac > 0 and (idx % int(1/args.val_frac) == 0):
            return "val"
        return "train"

    # If CSV is one-row-per-image with an annotations JSON field
    if image_cols and ann_cols:
        img_col = image_cols[0]
        ann_col = ann_cols[0]
        for idx, row in enumerate(rows):
            img_name = row.get(img_col, "").strip()
            if not img_name:
                continue
            img_path = (images_root / img_name)
            if not img_path.exists():
                # try with common extensions
                found = None
                for ext in (".png",".jpg",".jpeg",".tiff"):
                    if (images_root / (img_name + ext)).exists():
                        found = images_root / (img_name + ext)
                        break
                    if Path(img_name).suffix and (images_root / img_name).exists():
                        found = images_root / img_name
                        break
                if found:
                    img_path = found
                else:
                    print("MISSING IMAGE:", img_name, "tried under", images_root)
                    continue
            try:
                from PIL import Image
                w,h = Image.open(img_path).size
            except Exception as e:
                print("Failed to open image", img_path, ":", e)
                continue

            anns = parse_json_annotations_field(row.get(ann_col, ""))
            if not anns:
                # maybe it's a string like "[{...},...]" or a semicolon separated label list
                print("No annotations parsed for", img_name)
                continue

            split = choose_split(idx, len(rows))
            target_img_dir = train_img_dir if split=="train" else val_img_dir
            target_lbl_dir = train_lbl_dir if split=="train" else val_lbl_dir

            # copy image if requested
            if args.copy_images:
                shutil.copy2(img_path, target_img_dir / Path(img_path).name)
            else:
                # create symlink if possible, fallback to copy
                try:
                    os.symlink(os.path.abspath(img_path), target_img_dir / Path(img_path).name)
                except Exception:
                    shutil.copy2(img_path, target_img_dir / Path(img_path).name)

            lbl_lines = []
            for a in anns:
                # handle possible keys
                name = a.get("test_name") or a.get("class") or a.get("label") or a.get("name") or "0"
                xmin = a.get("x_min") or a.get("xmin") or a.get("x1") or a.get("left")
                ymin = a.get("y_min") or a.get("ymin") or a.get("y1") or a.get("top")
                xmax = a.get("x_max") or a.get("xmax") or a.get("x2") or a.get("right")
                ymax = a.get("y_max") or a.get("ymax") or a.get("y2") or a.get("bottom")
                if None in (xmin, ymin, xmax, ymax):
                    # skip if no bbox
                    continue
                try:
                    xmin = float(xmin); ymin = float(ymin); xmax = float(xmax); ymax = float(ymax)
                except:
                    continue
                if name not in class_map:
                    class_map[name] = next_id; next_id += 1
                cid = class_map[name]
                cx, cy, rw, rh = xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h)
                lbl_lines.append(f"{cid} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}")

            # write label file
            if lbl_lines:
                with open(target_lbl_dir / (Path(img_path).stem + ".txt"), "w", encoding="utf8") as f:
                    f.write("\n".join(lbl_lines))

    # Else if CSV is per-bbox (rows contain one bbox each)
    elif bbox_cols:
        # attempt to detect column names
        col_map = {c.lower():c for c in bbox_cols}
        def get_col(row, names):
            for n in names:
                if n in col_map:
                    return row[col_map[n]]
            return None

        for idx, row in enumerate(rows):
            img_name = get_col(row, ("image","image_path","filename","file","img")) or get_col(row, ("img_path","path","filepath"))
            if not img_name:
                continue
            img_path = (images_root / img_name)
            if not img_path.exists():
                print("MISSING IMAGE:", img_name, "tried under", images_root)
                continue
            try:
                from PIL import Image
                w,h = Image.open(img_path).size
            except Exception as e:
                print("Failed to open image", img_path, ":", e)
                continue

            xmin = get_col(row, ("x_min","xmin","x1","left"))
            ymin = get_col(row, ("y_min","ymin","y1","top"))
            xmax = get_col(row, ("x_max","xmax","x2","right"))
            ymax = get_col(row, ("y_max","ymax","y2","bottom"))
            name = get_col(row, ("class","label","test_name","test"))

            try:
                xmin = float(xmin); ymin = float(ymin); xmax = float(xmax); ymax = float(ymax)
            except:
                continue

            if name not in class_map:
                class_map[name] = next_id; next_id += 1
            cid = class_map[name]

            cx, cy, rw, rh = xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h)

            # choose split (naive: by image stem hash)
            split = "val" if (hash(img_name) % int(1/args.val_frac)) == 0 else "train"
            target_img_dir = train_img_dir if split=="train" else val_img_dir
            target_lbl_dir = train_lbl_dir if split=="train" else val_lbl_dir

            # ensure image present in target dir
            if args.copy_images and not (target_img_dir / Path(img_path).name).exists():
                shutil.copy2(img_path, target_img_dir / Path(img_path).name)

            # append label line
            lbl_path = target_lbl_dir / (Path(img_path).stem + ".txt")
            ensure_dir(lbl_path.parent)
            with open(lbl_path, "a", encoding="utf8") as f:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}\n")

            # ensure image copied/symlinked
            if not (target_img_dir / Path(img_path).name).exists():
                if args.copy_images:
                    shutil.copy2(img_path, target_img_dir / Path(img_path).name)
                else:
                    try:
                        os.symlink(os.path.abspath(img_path), target_img_dir / Path(img_path).name)
                    except Exception:
                        shutil.copy2(img_path, target_img_dir / Path(img_path).name)

    else:
        print("Unknown CSV layout. The script expects either:")
        print(" - one-row-per-image with a JSON 'annotations' column, or")
        print(" - one-row-per-bbox with columns image,xmin,ymin,xmax,ymax,class")
        return

    # write data.yaml
    names = [None] * len(class_map)
    for k,v in class_map.items():
        names[v] = k
    data_yaml = {
        "train": str((out / 'images' / 'train').resolve()),
        "val": str((out / 'images' / 'val').resolve()),
        "nc": len(names),
        "names": names
    }
    import yaml
    with open(out / "data.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(data_yaml, f)
    print("Wrote dataset to", out)
    print("Classes:", names)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--images-root", default=".")
    p.add_argument("--output", default="ml/ml_yolo")
    p.add_argument("--copy-images", action="store_true", help="Copy images to output (default: symlink/copy fallback)")
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction for val split (simple heuristic)")
    args = p.parse_args()
    convert(args)
