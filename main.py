import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import easyocr

from east import EastModel, input_size

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

img_path = 'img18.jpg'
original_img = cv2.imread(img_path)

# Preprocess the image
# preprocessed_img = preprocess_image(original_img)

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

reader = easyocr.Reader(['en'])
recognized_texts = []
for (x1, y1, x2, y2) in scaled_boxes:
    text_roi = original_img[y1:y2, x1:x2]
    result = reader.readtext(text_roi, detail=0)
    if result:
        recognized_texts.append((x1, y1, x2, y2, result[0]))

for (x1, y1, x2, y2, text) in recognized_texts:
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print([text[-1] for text in recognized_texts])
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Text")
plt.show()

cv2.imwrite("text_detection_result.png", original_img)
cv2.imshow("Detected Text", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
