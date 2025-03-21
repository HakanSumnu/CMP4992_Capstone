import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kf.statePost = np.array([[0], [0], [0], [0]], np.float32)

    def update(self, measured_x, measured_y):
        measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
        self.kf.correct(measurement)  # Correct Kalman filter state

    def predict(self):
        predicted_state = self.kf.predict()
        predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
        return predicted_x, predicted_y


def detect_tennis_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = None

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 100:
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            circle_area = np.pi * radius ** 2
            circularity = area / circle_area if circle_area > 0 else 0

            if circularity > 0.7:
                center = (int(x), int(y))
                radius = int(radius)

                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Ball ({center[0]}, {center[1]})",
                            (center[0] - 50, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ball_position = center

    return frame, ball_position


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    kf = KalmanFilter()
    ball_trail = []
    max_trail_length = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame, ball_position = detect_tennis_ball(frame)

        if ball_position:

            kf.update(*ball_position)
        else:

            ball_position = kf.predict()

        ball_trail.append(ball_position)
        if len(ball_trail) > max_trail_length:
            ball_trail.pop(0)

        for i in range(1, len(ball_trail)):
            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (255, 0, 0), 2)

        predicted_x, predicted_y = kf.predict()
        cv2.drawMarker(frame, (predicted_x, predicted_y), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                       thickness=2)

        cv2.imshow('Tennis Ball Tracking with Kalman Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
