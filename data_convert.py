import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm

# ==================== 配置路径 ====================
DATA_ROOT = "/root/autodl-tmp/TT100k_coco/data"
ANNO_FILE = os.path.join(DATA_ROOT, "annotations.json")
OUTPUT_ROOT = "/root/autodl-tmp/tt100k_coco_format"

SPLITS = ["train", "test", "other"]


def convert_tt100k_to_coco():
    print("正在加载原始 annotations.json ...")
    with open(ANNO_FILE, 'r', encoding='utf-8') as f:
        tt100k_data = json.load(f)

    if 'imgs' in tt100k_data:
        images_dict = tt100k_data['imgs']
    elif 'images' in tt100k_data:
        images_dict = tt100k_data['images']
    else:
        raise KeyError("未在 JSON 中找到 'imgs' 或 'images' 键。")

    print("正在统计类别实例数量...")
    class_counts = defaultdict(int)
    for img_id, img_info in images_dict.items():
        for ann in img_info.get('objects', []):
            class_counts[ann['category']] += 1

    selected_classes = [cls for cls, count in class_counts.items() if count >= 100]
    selected_classes = sorted(selected_classes)

    print(f"\n成功筛选出 {len(selected_classes)} 个核心类别。")

    # ====== 【核心修正点】: 为了防止 RT-DETR/DAB-DETR 等架构类别越界，ID 必须从 0 开始严格递增 (0 ~ 44) ======
    category_map = {cls: idx for idx, cls in enumerate(selected_classes)}

    coco_categories = [{"id": c_id, "name": cls, "supercategory": "traffic_sign"} for cls, c_id in category_map.items()]

    for split in SPLITS:
        print(f"\n正在处理 {split} 集合...")
        split_img_dir = os.path.join(OUTPUT_ROOT, "images", split)
        split_anno_dir = os.path.join(OUTPUT_ROOT, "annotations")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_anno_dir, exist_ok=True)

        coco_images = []
        coco_annotations = []

        img_id_counter = 1
        ann_id_counter = 1

        for img_id, img_info in tqdm(images_dict.items()):
            img_path = img_info.get('path', '')
            if f"{split}/" not in img_path:
                continue

            valid_objects = []
            for obj in img_info.get('objects', []):
                if obj['category'] in category_map:
                    bbox = obj['bbox']
                    xmin = float(bbox['xmin'])
                    ymin = float(bbox['ymin'])
                    xmax = float(bbox['xmax'])
                    ymax = float(bbox['ymax'])

                    if xmin > xmax: xmin, xmax = xmax, xmin
                    if ymin > ymax: ymin, ymax = ymax, ymin

                    w = xmax - xmin
                    h = ymax - ymin

                    # 严苛过滤掉小噪点（宽高均不小于8像素），提高数据在RandomCrop时的耐受度
                    if w >= 8.0 and h >= 8.0:
                        obj['clean_bbox'] = [xmin, ymin, w, h]
                        valid_objects.append(obj)

            if not valid_objects:
                continue

            src_img_path = os.path.join(DATA_ROOT, img_path)
            dst_img_name = f"{img_id}.jpg"
            dst_img_path = os.path.join(split_img_dir, dst_img_name)

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            else:
                alternative_path = os.path.join(DATA_ROOT, split, f"{img_id}.jpg")
                if os.path.exists(alternative_path):
                    shutil.copy(alternative_path, dst_img_path)
                else:
                    continue

            width = img_info.get('width', 2048)
            height = img_info.get('height', 2048)

            coco_images.append({
                "id": img_id_counter,
                "file_name": dst_img_name,
                "width": width,
                "height": height
            })

            for obj in valid_objects:
                w_c, h_c = obj['clean_bbox'][2], obj['clean_bbox'][3]
                coco_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": category_map[obj['category']],  # 从0开始的ID
                    "bbox": obj['clean_bbox'],
                    "area": w_c * h_c,
                    "iscrowd": 0
                })
                ann_id_counter += 1

            img_id_counter += 1

        coco_output = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories
        }

        output_json_path = os.path.join(split_anno_dir, f"instances_{split}.json")
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(coco_output, f_out, ensure_ascii=False, indent=4)

        print(f"-> 成功导出清洗后的 {split} ！图片数: {len(coco_images)}, 框数: {len(coco_annotations)}")

    print(f"\n数据集类别越界修正完成！数据保存在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    convert_tt100k_to_coco()