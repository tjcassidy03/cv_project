import numpy as np
import cv2
import easyocr
import csv
import itertools
import difflib
import Levenshtein
from glob import glob

# Create Transcriptions Hash Map

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

def process_image(img_path, transcriptions_and_bbox_hashmap, metrics_list):
    import unicodedata

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

    path_key = img_path.split('./images/')[1]
    original_img = cv2.imread(img_path)

    reader = easyocr.Reader(['en', 'vi', 'es'], gpu=False, verbose=False)
    results = reader.readtext(original_img)

    recognized_texts = []
    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(int)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        recognized_texts.append((x1, y1, x2, y2, text))

    ocr_words_raw = list(itertools.chain(*[text.split() for (_, _, _, _, text) in recognized_texts]))
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

    # IoU comparison (EasyOCR boxes vs GT)
    ious = []
    for gt in gt_entries:
        for pred in recognized_texts:
            iou = compute_iou(gt['bbox'], pred[:4])
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
        "image": path_key,
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

def calculate_averages(metrics_list):
    keys = [
        'accuracy', 'partial_accuracy', 'cer', 'wer', 
        'iou', 'tp', 'fp', 'fn', 
        'precision', 'recall', 'hmean'
    ]
    averages = {}
    for key in keys:
        values = [metric[key] for metric in metrics_list if key in metric]
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
    process_image(img_path, transcriptions_and_bbox_hashmap, metrics_list)

else:
    folder_path = './images/vietsignboard'
    image_paths = [path.replace('\\', '/') for path in glob(os.path.join(folder_path, '*.jpg'))]

    for img_path in image_paths:
        process_image(img_path, transcriptions_and_bbox_hashmap, metrics_list)

averages = calculate_averages(metrics_list)
save_to_csv(metrics_list, averages, 'easyOCRMetrics.csv')
