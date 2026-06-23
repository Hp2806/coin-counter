import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main tkinter window
Tk().withdraw()

# Open file picker dialog
file_path = askopenfilename(
    title="Select Coin Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

# If user cancels file selection
if not file_path:
    print("No image selected!")
    exit()

# Load selected image
image = cv2.imread(file_path)

if image is None:
    print("Error: Could not load the selected image!")
    exit()

output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (11, 11), 0)

# Otsu's thresholding — automatically picks the best cutoff
# Use THRESH_BINARY if coins are darker than background,
# THRESH_BINARY_INV if coins are lighter than background
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(thresh, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(image, markers)



# --- Count and draw ---
coin_count = 0
for marker_id in np.unique(markers):
    if marker_id <= 1:  # 0 = unknown, 1 = background
        continue

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers == marker_id] = 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area > 500:  # tune this based on your image
        coin_count += 1
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(output, f"{coin_count}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.putText(output, f"Total Coins: {coin_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

cv2.imshow("Threshold Image", thresh)
cv2.imshow("Coin Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Total number of coins detected:", coin_count)
