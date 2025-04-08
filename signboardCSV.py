import json
import csv

with open('annotations_line.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

transcriptions_map = {}

annotations_list = data.get('annotations', [])
if isinstance(annotations_list, dict) and 'annotations' in annotations_list:
    annotations_list = annotations_list['annotations']

for annotation in annotations_list:
    image_id = annotation['image_id']
    transcription = annotation.get('transcription', '')

    if image_id not in transcriptions_map:
        transcriptions_map[image_id] = []
    transcriptions_map[image_id].append(transcription)

with open('signboardTranscriptions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'transcriptions'])

    for image_id, transcriptions in transcriptions_map.items():
        joined = ' | '.join(transcriptions)
        writer.writerow([image_id, joined])