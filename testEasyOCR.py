import cv2
import easyocr

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

# Place images folder in root directory
img_path = './images/vietsignboard/baker-1.jpg'
image = cv2.imread(img_path)

reader = easyocr.Reader(['en', 'vi', 'es'])
results = reader.readtext(img_path)

for (bbox, text, prob) in results:
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox 
    x_min, y_min = int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))
    x_max, y_max = int(max(x1, x2, x3, x4)), int(max(y1, y2, y3, y4))

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

recognized_texts = [result[1] for result in results]
# cv2.imwrite("output.jpg", image)

# Get actual image annotations
import itertools

img_annotations = recognized_texts
img_annotations = list(itertools.chain(*[element.split() if ' ' in element else [element] for element in img_annotations]))

path = img_path.split('./images/')[1]
actual_annotations = transcriptions_hashmap[path]
actual_annotations = list(itertools.chain(*[element.split() if ' ' in element else [element] for element in actual_annotations]))

print(img_annotations)
print(actual_annotations)

# Calculate Accuracy and Partial Accuracy
import difflib

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
import Levenshtein

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
