import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from east import EastModel, input_size

model = EastModel(None)
model_data = torch.load("east.pt", map_location=torch.device("cpu"))
model.load_state_dict(model_data)
model.eval()

img_path = 'test2.jpg'
original_img = cv2.imread(img_path)

img_resized = cv2.resize(original_img, (input_size, input_size))

img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

img_normalized = img_rgb.astype(np.float32) / 255.0

# Convert to PyTorch tensor and add batch dimension
img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]

with torch.no_grad():
    score, geo = model(img_tensor)

score_np = score.squeeze().cpu().numpy()
geo_np = geo.squeeze().cpu().numpy()

print(f"Score shape: {score_np.shape}")
print(f"Geo shape: {geo_np.shape}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 2)
plt.title("Score Map")
plt.imshow(score_np, cmap='jet')

# Display the first 4 channels of geo map (distances)
for i in range(4):
    plt.subplot(2, 3, 3+i)
    plt.title(f"Geo Map Channel {i}")
    plt.imshow(geo_np[i], cmap='jet')

plt.tight_layout()
plt.savefig('east_visualization2.png')
plt.show()

print("Visualization saved to east_visualization.png")

cv2.imshow("Original Image", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

