import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import easyocr
import itertools
import os
import difflib
import Levenshtein

from east import EastModel, input_size
from glob import glob

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    preprocessed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return preprocessed

def resize_with_padding(image, target_size):
    if len(image.shape) == 2:  # Single-channel image
        h, w = image.shape
        padded_img = np.zeros((target_size, target_size), dtype=np.uint8)
    else:  # Three-channel image
        h, w, _ = image.shape
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h))
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded_img[top:top+new_h, left:left+new_w] = resized_img
    return padded_img, scale, top, left, new_w, new_h

def decode_bounding_boxes(score_map, geo_map, score_thresh=0.5):
    h, w = score_map.shape
    boxes = []
    scores = []
    
    for y in range(h):
        for x in range(w):
            if score_map[y, x] >= score_thresh:
                top, right, bottom, left = geo_map[:4, y, x]
                x1, y1 = x * 4 - left, y * 4 - top
                x2, y2 = x * 4 + right, y * 4 + bottom
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(score_map[y, x]))
    
    return np.array(boxes), np.array(scores)

def non_maximum_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[indices[1:]] - inter)
        
        indices = indices[np.where(iou <= iou_threshold)[0] + 1]
    
    return boxes[keep]

model = EastModel(None)
model_data = torch.load("east.pt", map_location=torch.device("cpu"))
model.load_state_dict(model_data)
model.eval()

# Create Transcriptions and Border Box Hash Map
import csv
transcriptions_and_bbox_hashmap = {}

with open('signboardTranscriptions.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  

    for row in reader:
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

# for image_id, entries in transcriptions_and_bbox_hashmap.items():
#     print(f"{image_id}:")
#     for entry in entries:
#         print(f"  Transcription: {entry['transcription']}, BBox: {entry['bbox']}")

def process_image(img_path, transcriptions_and_bbox_hashmap, model, input_size, metrics_list):
    path_key = img_path.split('./images/')[1]
    original_img = cv2.imread(img_path)

    # Preprocessing and resizing
    preprocessed_img = preprocess_image(original_img)
    img_resized, scale, top_pad, left_pad, new_w, new_h = resize_with_padding(preprocessed_img, input_size)

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        score, geo = model(img_tensor)

    score_np = score.squeeze().cpu().numpy()
    geo_np = geo.squeeze().cpu().numpy()
    boxes, scores = decode_bounding_boxes(score_np, geo_np)
    nms_boxes = non_maximum_suppression(boxes, scores)

    # Scaling boxes
    scaled_boxes = []
    for x1, y1, x2, y2 in nms_boxes:
        x1 = int((x1 - left_pad) / scale)
        y1 = int((y1 - top_pad) / scale)
        x2 = int((x2 - left_pad) / scale)
        y2 = int((y2 - top_pad) / scale)
        scaled_boxes.append((x1, y1, x2, y2))

    # Text recognition
    recognized_texts = []
    reader = easyocr.Reader(['en', 'vi', 'es'], gpu=False, verbose=False)
    for (x1, y1, x2, y2) in scaled_boxes:
        text_roi = original_img[y1:y2, x1:x2]
        result = []
        if text_roi is not None and text_roi.size != 0:
            result = reader.readtext(text_roi, detail=0)
        if result:
            recognized_texts.append((x1, y1, x2, y2, result[0]))

    # Normalize text
    def normalize_text(text):
        import unicodedata
        text = text.lower().strip()
        text = unicodedata.normalize('NFD', text)
        text = "".join([ch for ch in text if not unicodedata.combining(ch)])
        text = text.translate(str.maketrans('', '', ",.:;'()[]{}!?\""))
        return text

    ocr_words_raw = list(itertools.chain(*[text.split() for (_, _, _, _, text) in recognized_texts]))
    gt_entries = transcriptions_and_bbox_hashmap.get(path_key, [])
    gt_words_raw = list(itertools.chain(*[entry["transcription"].split() for entry in gt_entries]))

    ocr_words = [normalize_text(word) for word in ocr_words_raw]
    gt_words = [normalize_text(word) for word in gt_words_raw]

    # Word accuracy
    correct_count = sum(1 for word in gt_words if word in ocr_words)
    total_count = len(gt_words)
    total_characters = sum(len(word) for word in gt_words)

    # Partial accuracy
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

    # CER and WER
    def calculate_cer(ref_words, pred_words):
        cer = Levenshtein.distance("".join(ref_words), "".join(pred_words)) / max(len("".join(ref_words)), 1)
        return min(cer, 1.0)  

    def calculate_wer(ref_words, pred_words):
        wer = Levenshtein.distance(ref_words, pred_words) / max(len(ref_words), 1)
        return min(wer, 1.0)  

    cer = calculate_cer(gt_words, ocr_words)
    wer = calculate_wer(gt_words, ocr_words)

    # IoU Metrics
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

    gt_boxes_raw = [entry['bbox'] for entry in gt_entries]
    gt_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in gt_boxes_raw]
    pred_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in recognized_texts]

    iou_scores = []
    matched_gt, matched_pred = set(), set()
    threshold = 0.1

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= threshold:
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)
            iou_scores.append(best_iou)

    avg_iou = np.mean(iou_scores) if iou_scores else 0.0

    TP = len(matched_pred)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    hmean = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_list.append({
        'image': path_key,
        'accuracy': accuracy,
        'partial_accuracy': partial_accuracy,
        'cer': cer,
        'wer': wer,
        "iou": avg_iou,
        "ocr_words": ocr_words,
        "gt_words": gt_words,
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'precision': precision,
        'recall': recall,
        'hmean': hmean,
    })

    print(
        f"{path_key} - EA: {accuracy:.2f}% | "
        f"PCA: {partial_accuracy:.2f}% | "
        f"CER: {cer * 100:.2f}% | WER: {wer * 100:.2f}% | "
        f"IoU: {avg_iou:.4f} | "
        f"TP: {TP} | FP: {FP} | FN: {FN} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | H-Mean: {hmean:.4f}"
    )

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

all_files = False
metrics_list = []

if not all_files:
    img_path = './images/vietsignboard/baker-2.jpg'
    process_image(img_path, transcriptions_and_bbox_hashmap, model, input_size, metrics_list)

else:
    folder_path = './images/vietsignboard'
    image_paths = [path.replace('\\', '/') for path in glob(os.path.join(folder_path, '*.jpg'))]

    for img_path in image_paths:
        process_image(img_path, transcriptions_and_bbox_hashmap, model, input_size, metrics_list)

averages = calculate_averages(metrics_list)
save_to_csv(metrics_list, averages, 'eastAndEasyOCRMetrics.csv')
