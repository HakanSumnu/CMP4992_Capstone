import cv2
import numpy as np
import math
import sys
import time
import tkinter as tk
from tkinter import Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 29.5 / 24 #For Hakan's phone cam
#RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 29.5 * (9 / 16) / 24 #For Hakan's phone cam
RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 32 / 43 #For Hakan's computer cam
RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 32 / 43 * (3 / 4) #For Hakan's computer cam
#RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 29.5 / 22 #For Zeynep's computer cam
#RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 29.5 / 22 * (2214 / 3420) #For Zeynep's computer cam
#DIAMETER_OF_BALL_IN_PIXEL_AT_CERTAIN_DISTANCE = 212 * 30 #In here 212 is the diameter in pixel and 30 cm is the distance of the camera from the ball.
#RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = math.pi * 106 * 106 / (960 * 540) #In photo taken from 30cm distance and with Hakan's phone.
RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = (math.pi * 78 * 78) / (640 * 480) #In photo taken from 50cm distance with Hakan's computer cam.
#RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = (math.pi * 230 * 230) / (3420 * 2214) #In photo taken from 50cm distance with Zeynep's computer cam.
#CERTAIN_DISTANCE_BETWEEN_BALL_AND_PHOTO = 30.0
CERTAIN_DISTANCE_BETWEEN_BALL_AND_PHOTO = 50.0

DISTANCE_BETWEEN_ROBOT_INITIAL_AND_CAMERA: float = 0.0 # In cm. It should be added to line equation since we use right hand-side coordinate system, which causes robot's initisl point to stay at
                                                 # some point whose z component is negative. Therefore, every point in the coordinate system should be shifted this amount in Z axis 
                                                 # to locate robot's initial point to origin (0, 0).
ROBOT_RADIUS: float = 5.0
BALL_RADIUS: float = 4.2
directionVector: np.ndarray = np.array([0, 0, -1]) # Before I change my mathematical formulation, this vector's z componet had to be -1. Now, it does not matter whether it is -1 or 1. 
                                       # The only thing that matter is that this vector should be a unit vector and all of its components must be zero except it z component.
LEFT_BOUNDARY = -500.0 #In cm
RIGHT_BOUNDARY = 500.0 #In cm
NUMBER_OF_REGIONS = 10
WEIGHT_FOR_PASSES = 10

# -- UI Data Storage --
path_history = []
current_path = []
path_colors = ["red", "blue", "green", "purple", "orange", "cyan"]

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

def detectBall(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerYellow = np.array([45, 76, 25])
    upperYellow = np.array([75, 179, 255])
    mask = cv2.inRange(hsv, lowerYellow, upperYellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    cv2.imshow("Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ballPosition = None
    locationVector = None

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

                ballPosition = center

                horizontalAngle = (x - frame.shape[1] / 2) / frame.shape[1]
                verticalAngle = (y - frame.shape[0] / 2) / frame.shape[0]
                if frame.shape[1] > frame.shape[0]:
                    horizontalAngle = math.atan(horizontalAngle * RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE)
                    verticalAngle = math.atan(verticalAngle * RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE)
                else:
                    horizontalAngle = math.atan(horizontalAngle * RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE)
                    verticalAngle = math.atan(verticalAngle * RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE)
                locationVector = directionVector.copy()
                locationVector = np.array([
                    locationVector[0] * math.cos(-horizontalAngle) + locationVector[2] * math.sin(-horizontalAngle),
                    locationVector[1],
                    locationVector[0] * -math.sin(-horizontalAngle) + locationVector[2] * math.cos(-horizontalAngle)
                                           ])
                locationVector = np.array([
                    locationVector[0],
                    locationVector[1] * math.cos(-verticalAngle) + locationVector[2] * -math.sin(-verticalAngle),
                    locationVector[1] * math.sin(-verticalAngle) + locationVector[2] * math.cos(-verticalAngle)
                                           ])
                perpendicularDistance = math.sqrt(RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE / ((math.pi * radius * radius) / (frame.shape[0] * frame.shape[1]))) * CERTAIN_DISTANCE_BETWEEN_BALL_AND_PHOTO
                distance = perpendicularDistance / (math.cos(horizontalAngle) * math.cos(verticalAngle))
                locationVector *= distance
                #print(distance)
                #print(f"Location: {locationVector}, hoav: {horizontalAngle / math.pi * 180}, vaov: {verticalAngle / math.pi * 180}, {distance}")

    return frame, ballPosition, locationVector

def findPath(locations):
    X = np.empty(len(locations))
    Y = np.empty(len(locations))
    Z = np.empty(len(locations))
    counter = 0

    for location in locations:
        X[counter] = location[0]
        Y[counter] = location[1]
        Z[counter] = location[2]
        counter += 1

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Calculating best fit line in 3D.
    A: np.ndarray = np.array([[np.dot(X, X), np.sum(X)], [np.sum(X), X.shape[0]]])
    B1: np.ndarray = np.array([np.dot(X, Y), np.sum(Y)])
    x1: np.ndarray = np.linalg.solve(A, B1)
    #print(f"Equation of y: {x1}")

    B2: np.ndarray = np.array([np.dot(X, Z), np.sum(Z)])
    x2: np.ndarray = np.linalg.solve(A, B2)
    #print(f"Equation of z: {x2}")

    # Calculating the angle between XZ plane at y = 0 and YZ components of the vector that is parallel to best fit line.
    parallelVectorToLineClamppedToYZ: np.ndarray = np.array([0, x1[0], x2[0]])
    parallelVectorToLineClamppedToYZ /= np.linalg.norm(parallelVectorToLineClamppedToYZ)
    angle: float = math.acos(np.dot(parallelVectorToLineClamppedToYZ, directionVector))

    if angle >= math.pi - angle:
        angle = math.pi - angle

    if (x1[0] > 0) != (x2[0] > 0):
        angle *= -1.0

    #print(f"Angle: {angle * 180.0 / math.pi}")

    # Rotating the best fit line and considering it in XZ coordinate system
    coefficientsOfNewLine: np.ndarray = np.array([math.sin(angle) * x1[0] + math.cos(angle) * x2[0],\
                                                  math.sin(angle) * x1[1] + math.cos(angle) * x2[1] + DISTANCE_BETWEEN_ROBOT_INITIAL_AND_CAMERA])
    #print(f"Coefficients of the new line: {coefficientsOfNewLine}")

    # Finding two points whose distances to best fit line are equal to the sum of diameters of both robot and the ball.
    point1: float = ((ROBOT_RADIUS + BALL_RADIUS) * math.sqrt(coefficientsOfNewLine[0] * coefficientsOfNewLine[0] + 1.0) + coefficientsOfNewLine[1]) / -coefficientsOfNewLine[0]
    point2: float = ((ROBOT_RADIUS + BALL_RADIUS) * math.sqrt(coefficientsOfNewLine[0] * coefficientsOfNewLine[0] + 1.0) - coefficientsOfNewLine[1]) / coefficientsOfNewLine[0]
    #print(f"Point 1: {point1}, Point 2: {point2}")
    return (x1, x2), coefficientsOfNewLine, (point1, point2)

def findEvasionPoint(points, numberOfPassesPerRegion, robotCurrentPosition):
    leftPoint = min(points[0], points[1])
    rightPoint = max(points[0], points[1])

    if robotCurrentPosition >= leftPoint and robotCurrentPosition <= rightPoint:
        #print("Crash")
        #print(f"Regions: {numberOfPassesPerRegion}")
        minCost = 999999
        pointWithMinimumCost = 0.0
        for i in range(NUMBER_OF_REGIONS):
            regionLeft = LEFT_BOUNDARY + i * (RIGHT_BOUNDARY - LEFT_BOUNDARY) / NUMBER_OF_REGIONS
            regionRight = regionLeft + (RIGHT_BOUNDARY - LEFT_BOUNDARY) / NUMBER_OF_REGIONS
            if leftPoint <= regionLeft + 2 * ROBOT_RADIUS and rightPoint >= regionRight - 2 * ROBOT_RADIUS:
                continue
            elif leftPoint > regionLeft + 2 * ROBOT_RADIUS and leftPoint < regionRight - 2 * ROBOT_RADIUS and rightPoint >= regionRight - 2 * ROBOT_RADIUS:
                if abs(regionLeft + 2 * ROBOT_RADIUS - robotCurrentPosition) <= abs(leftPoint - robotCurrentPosition):
                    distanceToClosestPointInRegion = abs(regionLeft + 2 * ROBOT_RADIUS - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = regionLeft + 2 * ROBOT_RADIUS
                else:
                    distanceToClosestPointInRegion = abs(leftPoint - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = leftPoint
            elif rightPoint > regionLeft + 2 * ROBOT_RADIUS and rightPoint < regionRight - 2 * ROBOT_RADIUS and leftPoint <= regionLeft + 2 * ROBOT_RADIUS:
                if abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition) <= abs(rightPoint - robotCurrentPosition):
                    distanceToClosestPointInRegion = abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = regionRight - 2 * ROBOT_RADIUS
                else:
                    distanceToClosestPointInRegion = abs(rightPoint - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = rightPoint
            else:
                if abs(regionLeft + 2 * ROBOT_RADIUS - robotCurrentPosition) <= abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition):
                    distanceToClosestPointInRegion = abs(regionLeft + 2 * ROBOT_RADIUS - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = regionLeft + 2 * ROBOT_RADIUS
                else:
                    distanceToClosestPointInRegion = abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition)
                    if minCost > distanceToClosestPointInRegion:
                        minCost = distanceToClosestPointInRegion
                        pointWithMinimumCost = regionRight - 2 * ROBOT_RADIUS

        #print(f"Min cost: {minCost}")
        return pointWithMinimumCost - robotCurrentPosition
    else:
        #print("Not move")
        return 0.0

def show_path_popup():
    print("Popup triggered")
    popup = Toplevel()
    popup.title("Trajectory Viewer")
    popup.geometry("700x500")
    fig, ax = plt.subplots()
    ax.set_title("Previous Ball Paths")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    def plot_previous_paths():
        for idx, path in enumerate(path_history):
            if len(path) >= 3:
                path = np.array(path)
                x_vals = path[:, 0]
                z_vals = path[:, 2]
                coeffs = np.polyfit(x_vals, z_vals, 2)
                x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
                z_fit = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, z_fit, color=path_colors[idx % len(path_colors)], label=f"Past Path {idx+1}")
        ax.legend()
        canvas.draw()
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    btn = tk.Button(popup, text="Show Previous Paths", command=plot_previous_paths)
    btn.pack(side=tk.RIGHT, padx=10, pady=10)

if __name__ == "__main__":
    robotCurrentPosition = 0.0
    numberOfPassesPerRegion = np.zeros(NUMBER_OF_REGIONS)

    camera = cv2.VideoCapture(0)
    root = tk.Tk()
    root.title("Trajectory Control Panel")
    button = tk.Button(root, text="Show Past Trajectories", command=show_path_popup)
    button.pack()
    locations = []
    start = time.time()

    trajectory_pixels = []

    while True:
        ret, frame = camera.read()
        root.update_idletasks()
        root.update()
        if not ret:
            break
        #frame = cv2.flip(frame, 1)
        frame, ballPosition2d, location = detectBall(frame)

        if ballPosition2d:
            trajectory_pixels.append(ballPosition2d)  # 🟥 store pixel center
            locations.append(location)
            current_path.append(location) 
            start = time.time()

        if (time.time() - start > 1.0):
            if current_path:
                path_history.append(current_path.copy())
                current_path.clear()
                trajectory_pixels.clear()
                locations.clear()

        # ✅ Draw ball trajectory using center pixels
        if len(trajectory_pixels) >= 2:
            pts = np.array(trajectory_pixels, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)  # 🟡 Yellow curve

        if len(locations) > 20:
            locations.pop(0)
            coefficientsOfLine3D, coefficientsOfLine2D, boundries = findPath(locations=locations)
            numberOfPassesPerRegion[int((-coefficientsOfLine2D[1] / coefficientsOfLine2D[0] + LEFT_BOUNDARY) / ((RIGHT_BOUNDARY - LEFT_BOUNDARY) / NUMBER_OF_REGIONS))] += 1
            amountOfMovementRequired = findEvasionPoint(boundries, numberOfPassesPerRegion, robotCurrentPosition)
            #predictedLineUIPoint1 = (int(((1e-4 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0]) / 1e-4), int(((1e-4 - coefficientsOfLine3D[1][1]) * (coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0]) + coefficientsOfLine3D[0][1]) / 1e-4))
            #predictedLineUIPoint2 = (int(((1e+4 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0]) / 1e+4), int(((1e+4 - coefficientsOfLine3D[1][1]) * (coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0]) + coefficientsOfLine3D[0][1]) / 1e+4))
            predictedLineUIPoint1 = [10000, coefficientsOfLine3D[0][0] * 10000 + coefficientsOfLine3D[0][1], coefficientsOfLine3D[1][0] * 10000 + coefficientsOfLine3D[1][1]]
            predictedLineUIPoint2 = [-10000, coefficientsOfLine3D[0][0] * -10000 + coefficientsOfLine3D[0][1], coefficientsOfLine3D[1][0] * -10000 + coefficientsOfLine3D[1][1]]

            predictedLineUIPoint1[0] = predictedLineUIPoint1[0] / (RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE * predictedLineUIPoint1[2]) #Left of the screen is -0.5 and right of the screen is 0.5
            predictedLineUIPoint1[1] = predictedLineUIPoint1[1] / (RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE * predictedLineUIPoint1[2]) #Bottom of the screen is -0.5 and top of the screen is 0.5
            predictedLineUIPoint1[0] += 0.5 #Left of the screen is 0.0 and right of the screen is 1.0
            predictedLineUIPoint1[1] += 0.5 #Bottom of the screen is 0.0 and top of the screen is 1.0
            predictedLineUIPoint1[1] *= -1 #BottoTopm of the screen is 0.0 and bottom of the screen is 1.0

            predictedLineUIPoint2[0] = predictedLineUIPoint2[0] / (RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE * predictedLineUIPoint2[2]) #Left of the screen is -0.5 and right of the screen is 0.5
            predictedLineUIPoint2[1] = predictedLineUIPoint2[1] / (RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE * predictedLineUIPoint2[2]) #Bottom of the screen is -0.5 and top of the screen is 0.5
            predictedLineUIPoint2[0] += 0.5 #Left of the screen is 0.0 and right of the screen is 1.0
            predictedLineUIPoint2[1] += 0.5 #Bottom of the screen is 0.0 and top of the screen is 1.0
            predictedLineUIPoint2[1] *= -1 #BottoTopm of the screen is 0.0 and bottom of the screen is 1.0

            #To find in pixel
            predictedLineUIPoint1[0] = int(predictedLineUIPoint1[0] * frame.shape[1])
            predictedLineUIPoint1[1] = int(predictedLineUIPoint1[1] * frame.shape[0])

            predictedLineUIPoint2[0] = int(predictedLineUIPoint2[0] * frame.shape[1])
            predictedLineUIPoint2[1] = int(predictedLineUIPoint2[1] * frame.shape[0])

            print(f"pt1: {predictedLineUIPoint1}, pt2: {predictedLineUIPoint2}")

            cv2.line(frame, \
                     (predictedLineUIPoint1[0], predictedLineUIPoint1[1]), \
                     (predictedLineUIPoint2[0], predictedLineUIPoint2[1]), \
                     color=(0, 0, 0), \
                        thickness=5)
            sys.stdout.write("\033[K")  # Clear the entire line
            #print(f"Location: {location}, coefficients of line 3D {coefficientsOfLine3D}, coefficients of line 2D: {coefficientsOfLine2D}, boundries: {boundries} amount of movement required: {amountOfMovementRequired} pt1: {predictedLineUIPoint1}, pt2: {predictedLineUIPoint2}")
            sys.stdout.write("\033[F")  # Move cursor up one line
            sys.stdout.flush()

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    root.destroy()
    cv2.destroyAllWindows()