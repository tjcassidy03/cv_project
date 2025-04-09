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

# Create Transcriptions Hash Map
import csv
transcriptions_hashmap = {}

with open('signboardTranscriptions.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        image_id = row[0]  
        transcription = row[1]  
        
        transcription_list = transcription.split(" | ")
        
        if image_id in transcriptions_hashmap:
            transcriptions_hashmap[image_id].extend(transcription_list)
        else:
            transcriptions_hashmap[image_id] = transcription_list

# for image_id, transcriptions in transcriptions_hashmap.items():
#     print(f"{image_id}: {transcriptions}")

all_files = True

if not all_files:
    # Place images folder in root directory
    img_path = './images/vietsignboard/baker-1.jpg'
    original_img = cv2.imread(img_path)

    # Preprocess the image
    preprocessed_img = preprocess_image(original_img)

    # plt.figure(figsize=(10, 6))
    # plt.imshow(preprocessed_img, cmap='gray')
    # plt.title("Preprocessed Image")
    # plt.show()


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

    # get back to original shape
    scaled_boxes = []
    for x1, y1, x2, y2 in nms_boxes:
        x1 = int((x1 - left_pad) / scale)
        y1 = int((y1 - top_pad) / scale)
        x2 = int((x2 - left_pad) / scale)
        y2 = int((y2 - top_pad) / scale)
        scaled_boxes.append((x1, y1, x2, y2))

    reader = easyocr.Reader(['en', 'vi', 'es'])
    recognized_texts = []
    for (x1, y1, x2, y2) in scaled_boxes:
        text_roi = original_img[y1:y2, x1:x2]
        if text_roi is not None and text_roi.size != 0:
            result = reader.readtext(text_roi, detail=0)
        if result:
            recognized_texts.append((x1, y1, x2, y2, result[0]))

    for (x1, y1, x2, y2, text) in recognized_texts:
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Get actual image annotations

    img_annotations = [text[-1] for text in recognized_texts]
    img_annotations = list(itertools.chain(*[element.split() if ' ' in element else [element] for element in img_annotations]))

    path = img_path.split('./images/')[1]
    actual_annotations = transcriptions_hashmap[path]
    actual_annotations = list(itertools.chain(*[element.split() if ' ' in element else [element] for element in actual_annotations]))

    print(img_annotations)
    print(actual_annotations)

    # Calculate Accuracy and Partial Accuracy
    correct_count = 0
    total_count = len(actual_annotations)
    partial_correct_characters = 0
    total_characters = 0

    used_recognized = [False] * len(img_annotations)

    for annotation in actual_annotations:
        total_characters += len(annotation)
        if annotation in img_annotations:
            correct_count += 1

        for i, recognized in enumerate(img_annotations):
            if used_recognized[i]:
                continue  
            seq_match = difflib.SequenceMatcher(None, annotation, recognized)
            match_length = seq_match.find_longest_match(0, len(annotation), 0, len(recognized)).size
            partial_correct_characters += match_length

            if match_length > 0:
                used_recognized[i] = True

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    partial_accuracy = (partial_correct_characters / total_characters) * 100 if total_characters > 0 else 0

    print(f"Exact Accuracy: {accuracy:.2f}%")
    print(f"Partial Accuracy (based on characters): {partial_accuracy:.2f}%")

    # Calculate CER and WER
    def calculate_cer(reference, recognized):
        reference_str = "".join(reference)
        recognized_str = "".join(recognized)
        
        if len(reference_str) == 0:
            return 0
        
        lev_distance = Levenshtein.distance(reference_str, recognized_str)
        cer = lev_distance / len(reference_str)
        return cer

    def calculate_wer(reference, recognized):
        reference_str = " ".join(reference)
        recognized_str = " ".join(recognized)
        
        if len(reference_str.split()) == 0:
            return 0
        
        lev_distance = Levenshtein.distance(reference_str.split(), recognized_str.split())
        
        wer = lev_distance / len(reference_str.split())
        return wer

    cer = calculate_cer(img_annotations, actual_annotations)
    wer = calculate_wer(img_annotations, actual_annotations)
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

    # plt.figure(figsize=(15, 10))
    # plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Text")
    # plt.show()

    # cv2.imwrite("text_detection_result.png", original_img)
    # cv2.imshow("Detected Text", original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

elif all_files:
    folder_path = './images/vietsignboard'
    image_paths = [path.replace('\\', '/') for path in glob(os.path.join(folder_path, '*.jpg'))]

    all_metrics = []
    for img_path in image_paths:
        original_img = cv2.imread(img_path)
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

        scaled_boxes = []
        for x1, y1, x2, y2 in nms_boxes:
            x1 = int((x1 - left_pad) / scale)
            y1 = int((y1 - top_pad) / scale)
            x2 = int((x2 - left_pad) / scale)
            y2 = int((y2 - top_pad) / scale)
            scaled_boxes.append((x1, y1, x2, y2))

        recognized_texts = []
        for (x1, y1, x2, y2) in scaled_boxes:
            text_roi = original_img[y1:y2, x1:x2]
            reader = easyocr.Reader(['en', 'vi', 'es'], gpu=False, verbose=False)
            if text_roi is not None and text_roi.size != 0:
                result = reader.readtext(text_roi, detail=0)
            if result:
                recognized_texts.append(result[0])

        img_annotations = list(itertools.chain(*[text.split() for text in recognized_texts]))

        path_key = img_path.split('./images/')[1]
        actual_annotations = transcriptions_hashmap.get(path_key, [])
        actual_annotations = list(itertools.chain(*[text.split() for text in actual_annotations]))

        correct_count = 0
        total_count = len(actual_annotations)
        partial_correct_characters = 0
        total_characters = 0
        used_recognized = [False] * len(img_annotations)

        for annotation in actual_annotations:
            total_characters += len(annotation)
            if annotation in img_annotations:
                correct_count += 1

            for i, recognized in enumerate(img_annotations):
                if used_recognized[i]:
                    continue
                seq_match = difflib.SequenceMatcher(None, annotation, recognized)
                match_length = seq_match.find_longest_match(0, len(annotation), 0, len(recognized)).size
                partial_correct_characters += match_length
                if match_length > 0:
                    used_recognized[i] = True

        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        partial_accuracy = (partial_correct_characters / total_characters) * 100 if total_characters > 0 else 0

        # CER and WER functions
        def calculate_cer(reference, recognized):
            reference_str = "".join(reference)
            recognized_str = "".join(recognized)
            if len(reference_str) == 0:
                return 0
            lev_distance = Levenshtein.distance(reference_str, recognized_str)
            return lev_distance / len(reference_str)

        def calculate_wer(reference, recognized):
            if len(reference) == 0:
                return 0
            lev_distance = Levenshtein.distance(reference, recognized)
            return lev_distance / len(reference)

        cer = calculate_cer(actual_annotations, img_annotations) * 100
        wer = calculate_wer(actual_annotations, img_annotations) * 100

        print(f"📄 Image: {os.path.basename(img_path)}")
        print(f"   🔹 Exact Accuracy: {accuracy:.2f}%")
        print(f"   🔹 Partial Accuracy: {partial_accuracy:.2f}%")
        print(f"   🔹 CER: {cer:.2f}%")
        print(f"   🔹 WER: {wer:.2f}%\n")

        all_metrics.append((accuracy, partial_accuracy, cer, wer))

    # Calculate averages
    avg_acc = np.mean([m[0] for m in all_metrics])
    avg_partial = np.mean([m[1] for m in all_metrics])
    avg_cer = np.mean([m[2] for m in all_metrics])
    avg_wer = np.mean([m[3] for m in all_metrics])

    print("📊 AVERAGE METRICS ACROSS ALL IMAGES:")
    print(f"   ✅ Average Exact Accuracy: {avg_acc:.2f}%")
    print(f"   ✅ Average Partial Accuracy: {avg_partial:.2f}%")
    print(f"   ✅ Average CER: {avg_cer:.2f}%")
    print(f"   ✅ Average WER: {avg_wer:.2f}%")