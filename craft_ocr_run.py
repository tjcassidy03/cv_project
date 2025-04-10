import cv2
import numpy as np
import easyocr
import os

image_dir = 'angled_test'
output_dir = 'angled_test/craft_output'
bound_types = ['polys', 'boxes']

reader = easyocr.Reader(['en'])

def craft_easyocr():
    for img in os.listdir(image_dir):
        image_path = os.path.join(image_dir, img)

        original_img = cv2.imread(image_path)

        img_base = os.path.splitext(img)[0]
        img_out_path = os.path.join(output_dir, img_base, 'boxes')
        crop_path = os.path.join(img_out_path, 'image_crops')
        
        bbox_file = os.path.join(img_out_path, 'image_text_detection.txt')
        
        with open(bbox_file, 'r') as f:
            content = f.read()
            coord_blocks = content.strip().split('\n\n')
        
        recognized_texts = []
        
        crop_index = 0  
        for block in coord_blocks:
            if not block.strip():
                continue
            
            crop_file = os.path.join(crop_path, f"crop_{crop_index}.png")
            
            coords = block.strip().split(',')
            
            coords = [int(float(coord)) for coord in coords]


            points = np.array(coords).reshape(-1, 2)
            
            crop_img = cv2.imread(crop_file)
            if crop_img is None:
                print(f"Warning: Could not read crop file {crop_file}")
                crop_index += 1
                continue
                
            results = reader.readtext(crop_img)
            
            text = ""
            if results:
                text = " ".join([res[1] for res in results])
            
            recognized_texts.append((points, text))
            crop_index += 1

        result_img = original_img.copy()
        
        for points, text in recognized_texts:
            cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
            
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
        
        output_img_path = os.path.join(output_dir, f"{img_base}_annotated.jpg")
        cv2.imwrite(output_img_path, result_img)
        print(f"Created annotated image: {output_img_path}")
        
        cv2.imshow("Annotated Text", result_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

craft_easyocr()