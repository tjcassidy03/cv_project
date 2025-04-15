import cv2
import numpy as np
import easyocr
import os
from shapely.geometry import Polygon

image_dir = 'test_imgs'
output_dir = 'test_imgs/craft_output'
bound_types = ['polys', 'boxes']

reader = easyocr.Reader(['en'])

def calculate_iou(box1, box2):
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
            iou = calculate_iou(poly_points, box_points)
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

def recognize_text(crop_img, preprocessed_img):
    results_original = reader.readtext(crop_img)
    results_processed = reader.readtext(preprocessed_img)
    
    text = ""
    if results_original and results_processed:
        avg_conf_original = sum(res[2] for res in results_original) / len(results_original)
        avg_conf_processed = sum(res[2] for res in results_processed) / len(results_processed)
        
        if avg_conf_original > avg_conf_processed:
            text = " ".join([res[1] for res in results_original])
        else:
            text = " ".join([res[1] for res in results_processed])
    elif results_original:
        text = " ".join([res[1] for res in results_original])
    elif results_processed:
        text = " ".join([res[1] for res in results_processed])
    
    return text

def draw_annotations(image, recognized_texts):
    result_img = image.copy()
    
    for points, text, region_type in recognized_texts:
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

def process_single_image(img, image_path):
    print(f"Processing image: {img}")
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Could not read image: {image_path}")
        return
        
    img_base = os.path.splitext(img)[0]
    boxes_out_path = os.path.join(output_dir, img_base, 'boxes')
    polys_out_path = os.path.join(output_dir, img_base, 'polys')
    
    if not os.path.exists(boxes_out_path) or not os.path.exists(polys_out_path):
        print(f"Missing detection directories for {img}. Skipping.")
        return
    
    box_bbox_file = os.path.join(boxes_out_path, 'image_text_detection.txt')
    box_crop_path = os.path.join(boxes_out_path, 'image_crops')
    
    poly_bbox_file = os.path.join(polys_out_path, 'image_text_detection.txt')
    poly_crop_path = os.path.join(polys_out_path, 'image_crops')
    
    box_regions = load_regions_from_file(box_bbox_file)
    if not box_regions:
        print(f"No box regions found for {img}")
        
    poly_regions = load_regions_from_file(poly_bbox_file)
    if not poly_regions:
        print(f"No poly regions found for {img}")
    
    regions_to_use = determine_regions_to_use(poly_regions, box_regions)
    
    recognized_texts = []
    for region_type, region_idx in regions_to_use:
        if region_type == 'box':
            crop_file = os.path.join(box_crop_path, f"crop_{region_idx}.png")
            points = box_regions[region_idx]
        else:  
            crop_file = os.path.join(poly_crop_path, f"crop_{region_idx}.png")
            points = poly_regions[region_idx]
        
        crop_img = cv2.imread(crop_file)

        preprocessed_img = preprocess_crop(crop_img)
        text = recognize_text(crop_img, preprocessed_img)
        recognized_texts.append((points, text, region_type))
    
    if not recognized_texts:
        print(f"No text regions found for {img}")
        return

    result_img = draw_annotations(original_img, recognized_texts)
    
    output_img_path = os.path.join(output_dir, f"{img_base}_annotated.jpg")
    cv2.imwrite(output_img_path, result_img)
    print(f"Created annotated image: {output_img_path}")

def craft_easyocr():
    image_files = get_image_files(image_dir)
    for img in image_files:
        image_path = os.path.join(image_dir, img)
        process_single_image(img, image_path)
    
    print("All images processed.")

craft_easyocr()