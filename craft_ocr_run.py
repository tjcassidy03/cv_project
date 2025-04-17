import cv2
import numpy as np
import easyocr
import os
from shapely.geometry import Polygon
import csv
import unicodedata
import glob
import Levenshtein
import itertools
import difflib

image_dir = 'test_imgs'
output_dir = 'vietsignboard/craft_output'
bound_types = ['polys', 'boxes']

reader = easyocr.Reader(['en', 'vi', 'es'], verbose = False)

transcriptions_and_bbox_hashmap = {}

with open('signboardTranscriptions.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  

    for row in csv_reader:
        image_id = row[0]
        transcription = row[1]
        bbox_str = row[2]

        bbox = [int(x.strip()) for x in bbox_str.strip('[]').split(',')] if bbox_str else []

        entry = {
            "transcription": transcription,
            "bbox": bbox
        }

        if image_id in transcriptions_and_bbox_hashmap:
            transcriptions_and_bbox_hashmap[image_id].append(entry)
        else:
            transcriptions_and_bbox_hashmap[image_id] = [entry]

def normalize_text(text):
        text = text.lower().strip()
        text = unicodedata.normalize('NFD', text)
        text = "".join([ch for ch in text if not unicodedata.combining(ch)])
        text = text.translate(str.maketrans('', '', ",.:;'()[]{}!?\""))
        return text

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_cer(ref_words, pred_words):
    cer = Levenshtein.distance("".join(ref_words), "".join(pred_words)) / max(len("".join(ref_words)), 1)
    return min(cer, 1.0)

def calculate_wer(ref_words, pred_words):
    wer = Levenshtein.distance(ref_words, pred_words) / max(len(ref_words), 1)
    return min(wer, 1.0)

def calc_poly_box_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def get_image_files(directory):
    return [img for img in os.listdir(directory) 
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

def load_regions_from_file(file_path):
    regions = []
    
    if not os.path.exists(file_path):
        return regions
        
    with open(file_path, 'r') as f:
        content = f.read()
        coord_blocks = content.strip().split('\n\n')
        
        for block in coord_blocks:
            if not block.strip():
                continue
            
            coords = block.strip().split(',')
            coords = [int(float(coord)) for coord in coords]
            points = np.array(coords).reshape(-1, 2)
            regions.append(points)
            
    return regions

def determine_regions_to_use(poly_regions, box_regions):
    regions_to_use = []
    used_box_indices = set()
    
    for poly_idx, poly_points in enumerate(poly_regions):
        best_match_idx = -1
        best_iou = 0.0
        
        for box_idx, box_points in enumerate(box_regions):
            iou = calc_poly_box_iou(poly_points, box_points)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = box_idx
        
        if best_iou > 0.5 or best_iou < 0.1:
            regions_to_use.append(('poly', poly_idx))
            if best_match_idx >= 0 and best_iou > 0.5:
                used_box_indices.add(best_match_idx)

    for box_idx, box_points in enumerate(box_regions):
        if box_idx not in used_box_indices:
            regions_to_use.append(('box', box_idx))
            
    return regions_to_use

def preprocess_crop(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return opening

# def recognize_text(crop_img, preprocessed_img):
#     results_original = reader.readtext(crop_img)
#     results_processed = reader.readtext(preprocessed_img)
    
#     text = ""
#     if results_original and results_processed:
#         avg_conf_original = sum(res[2] for res in results_original) / len(results_original)
#         avg_conf_processed = sum(res[2] for res in results_processed) / len(results_processed)
        
#         if avg_conf_original > avg_conf_processed:
#             text = " ".join([res[1] for res in results_original])
#         else:
#             text = " ".join([res[1] for res in results_processed])
#     elif results_original:
#         text = " ".join([res[1] for res in results_original])
#     elif results_processed:
#         text = " ".join([res[1] for res in results_processed])
    
#     return text

def draw_annotations(image, recognized_texts):
    result_img = image.copy()
    
    for (points, text, conf, region_type) in recognized_texts:
        color = (0, 255, 0) if region_type == 'poly' else (255, 0, 0)
        cv2.polylines(result_img, [points], True, color, 2)
        
        x, y = points.min(axis=0)
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            result_img,
            (x, y - text_size[1] - 8),
            (x + text_size[0], y),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            result_img, text, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
    
    return result_img

def process_single_image(image_path, transcriptions_and_bbox_hashmap, metrics_list):
    # print(f"Processing image: {image_path}")
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Extract base name from image path
    img_base = os.path.basename(os.path.splitext(image_path)[0])
    path_key = os.path.basename(image_path)  # For looking up in hashmap
    
    boxes_out_path = os.path.join(output_dir, img_base, 'boxes')
    polys_out_path = os.path.join(output_dir, img_base, 'polys')
    
    if not os.path.exists(boxes_out_path) or not os.path.exists(polys_out_path):
        print(f"Missing detection directories for {image_path}. Skipping.")
        return
    
    box_bbox_file = os.path.join(boxes_out_path, 'image_text_detection.txt')
    box_crop_path = os.path.join(boxes_out_path, 'image_crops')
    
    poly_bbox_file = os.path.join(polys_out_path, 'image_text_detection.txt')
    poly_crop_path = os.path.join(polys_out_path, 'image_crops')
    
    box_regions = load_regions_from_file(box_bbox_file)
    if not box_regions:
        print(f"No box regions found for {image_path}")
        
    poly_regions = load_regions_from_file(poly_bbox_file)
    if not poly_regions:
        print(f"No poly regions found for {image_path}")
    
    regions_to_use = determine_regions_to_use(poly_regions, box_regions)
    
    recognized_texts = []
    for region_type, region_idx in regions_to_use:
        if region_type == 'box':
            crop_file = os.path.join(box_crop_path, f"crop_{region_idx}.png")
            points = box_regions[region_idx]
        else:  # poly
            crop_file = os.path.join(poly_crop_path, f"crop_{region_idx}.png")
            points = poly_regions[region_idx]
        
        crop_img = cv2.imread(crop_file)
        if crop_img is None:
            print(f"Could not read crop image: {crop_file}")
            continue

        preprocessed_img = preprocess_crop(crop_img)
        results = reader.readtext(preprocessed_img)
        
        for (bbox, text, conf) in results:
            recognized_texts.append((points, text, conf, region_type))
    
    if not recognized_texts:
        print(f"No text regions found for {image_path}")
        return

    ocr_words_raw = list(itertools.chain(*[text.split() for (_,  text, _, _) in recognized_texts]))
    gt_entries = transcriptions_and_bbox_hashmap.get(path_key, [])
    gt_words_raw = list(itertools.chain(*[entry["transcription"].split() for entry in gt_entries]))

    ocr_words = [normalize_text(word) for word in ocr_words_raw]
    gt_words = [normalize_text(word) for word in gt_words_raw]

    correct_count = sum(1 for word in gt_words if word in ocr_words)
    total_count = len(gt_words)
    total_characters = sum(len(word) for word in gt_words)

    partial_correct_characters = 0
    used = [False] * len(ocr_words)
    for gt_word in gt_words:
        for i, ocr_word in enumerate(ocr_words):
            if used[i]:
                continue
            match = difflib.SequenceMatcher(None, gt_word, ocr_word).find_longest_match(0, len(gt_word), 0, len(ocr_word)).size
            partial_correct_characters += match
            if match > 0:
                used[i] = True
                break

    accuracy = (correct_count / total_count) * 100 if total_count else 0
    partial_accuracy = (partial_correct_characters / total_characters) * 100 if total_characters else 0

    cer = calculate_cer(gt_words, ocr_words)
    wer = calculate_wer(gt_words, ocr_words)


    # SOMETHING WRONG WITH GETTING GT LABELS INTO THE LIST


    # IoU comparison (CRAFT regions vs GT)
    print(gt_entries)
    ious = []
    for gt in gt_entries:
        print('-')
        for points, _, _, region_type in recognized_texts:
            # Use calc_poly_box_iou instead of compute_iou
            iou = calc_poly_box_iou(gt['bbox'], points)
            if iou > 0:
                ious.append(iou)
                break

    avg_iou = np.mean(ious) if ious else 0.0
    
    tp = sum(1 for word in gt_words if word in ocr_words)
    fp = sum(1 for word in ocr_words if word not in gt_words)
    fn = sum(1 for word in gt_words if word not in ocr_words)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    hmean = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "image": image_path,
        "accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "cer": cer,
        "wer": wer,
        "iou": avg_iou,
        "ocr_words": ocr_words,
        "gt_words": gt_words,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
    }

    metrics_list.append(metrics)

    print(
        f"{path_key} - EA: {accuracy:.2f}% | "
        f"PCA: {partial_accuracy:.2f}% | "
        f"CER: {cer * 100:.2f}% | WER: {wer * 100:.2f}% | "
        f"IoU: {avg_iou:.4f} | "
        f"TP: {tp} | FP: {fp} | FN: {fn} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | H-Mean: {hmean:.4f}"
    )
    
    result_img = draw_annotations(original_img, recognized_texts)
    
    output_img_path = os.path.join(output_dir, f"{img_base}_annotated.jpg")
    cv2.imwrite(output_img_path, result_img)
    print(f"Created annotated image: {output_img_path}")

# def craft_easyocr():
#     image_files = get_image_files(image_dir)
#     for img in image_files:
#         image_path = os.path.join(image_dir, img)
#         process_single_image(img, image_path)
    
#     print("All images processed.")

def calculate_averages(metrics_list):
    keys = [
        'accuracy', 'partial_accuracy', 'cer', 'wer', 
        'iou', 'tp', 'fp', 'fn', 
        'precision', 'recall', 'hmean'
    ]
    filtered_metrics = [metric for metric in metrics_list if metric.get('iou', 0) >= 0.4]
    
    averages = {}
    for key in keys:
        values = [metric[key] for metric in filtered_metrics if key in metric]
        averages[key] = np.mean(values) if values else 0
    return averages

def save_to_csv(metrics_list, averages, output_csv):
    fieldnames = [
        'image', 'accuracy', 'partial_accuracy', 'cer', 'wer',
        'iou', 'tp', 'fp', 'fn', 
        'precision', 'recall', 'hmean'
    ]
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            row = {key: metrics.get(key, '') for key in fieldnames}
            writer.writerow(row)
        averages['image'] = 'Average'
        writer.writerow({key: averages.get(key, '') for key in fieldnames})

# craft_easyocr()


all_files = False
metrics_list = []

if not all_files:
    img_path = 'vietsignboard/baker-2.jpg'
    process_single_image(img_path, transcriptions_and_bbox_hashmap, metrics_list)

else:
    folder_path = 'vietsignboard'
    image_paths = [path.replace('\\', '/') for path in glob(os.path.join(folder_path, '*.jpg'))]

    for img_path in image_paths:
        process_single_image(img_path, transcriptions_and_bbox_hashmap, metrics_list)

averages = calculate_averages(metrics_list)
save_to_csv(metrics_list, averages, 'easyOCRMetrics.csv')