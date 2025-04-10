import json
import csv

with open('annotations_line.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

annotations_list = data.get('annotations', [])
if isinstance(annotations_list, dict) and 'annotations' in annotations_list:
    annotations_list = annotations_list['annotations']

with open('signboardTranscriptions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'transcription', 'bbox'])  # Add bbox column

    for annotation in annotations_list:
        image_id = annotation['image_id']
        transcription = annotation.get('transcription', '')
        bbox = annotation.get('bbox', [])

        # Convert bbox to a readable string
        bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if bbox else ""

        writer.writerow([image_id, transcription, bbox_str])
