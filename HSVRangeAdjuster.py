import cv2
import numpy as np


class HSVRangeAdjuster:
    def __init__(self, buffer_size=30):
        self.v_history = []
        self.buffer_size = buffer_size

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        avg_v = np.mean(v)
        self.v_history.append(avg_v)
        if len(self.v_history) > self.buffer_size:
            self.v_history.pop(0)

    def get_hsv_bounds(self, base_hue_range=(20, 40), s_range=(100, 255), v_padding=40):
        if not self.v_history:
            return (20, 100, 100), (40, 255, 255)

        avg_v = np.mean(self.v_history)
        lower_v = max(0, avg_v - v_padding)
        upper_v = min(255, avg_v + v_padding)

        lower = np.array([base_hue_range[0], s_range[0], lower_v], dtype=np.uint8)
        upper = np.array([base_hue_range[1], s_range[1], upper_v], dtype=np.uint8)

        return lower, upper


adjuster = HSVRangeAdjuster()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

adjuster.update(frame)
lower, upper = adjuster.get_hsv_bounds()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
