import cv2
import numpy as np

# Load the image
image = cv2.imread("image.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges in HSV
color_ranges = {
    "Red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),  # Lower red
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # Upper red
    "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "Green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))]
}

# Function to detect and draw bounding boxes
def detect_color(color_name, masks):
    for mask in masks:
        result = cv2.bitwise_and(image, image, mask=cv2.inRange(hsv, mask[0], mask[1]))
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

# Run detection
for color, ranges in color_ranges.items():
    detect_color(color, ranges)

# Show the result
cv2.imshow("Detected Traffic Lights", image)
cv2.waitKey(0)
cv2.destroyAllWindows()