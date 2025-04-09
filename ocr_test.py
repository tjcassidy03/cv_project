import cv2
import easyocr

img_path = 'img18.jpg'
image = cv2.imread(img_path)

reader = easyocr.Reader(['en', 'vi', 'es'])
results = reader.readtext(img_path)

for (bbox, text, prob) in results:
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox 
    x_min, y_min = int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))
    x_max, y_max = int(max(x1, x2, x3, x4)), int(max(y1, y2, y3, y4))

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print([result[1] for result in results])
cv2.imwrite("output.jpg", image)
