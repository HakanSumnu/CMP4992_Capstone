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
#RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 32 / 43 #For Hakan's computer cam
#RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 32 / 43 * (3 / 4) #For Hakan's computer cam
#RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 29.5 / 22 #For Zeynep's computer cam
#üRATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 29.5 / 22 * (2214 / 3420) #For Zeynep's computer cam
RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 0.966244 #For Hakan's web cam
RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 0.966244 * (9 / 16) #For Hakan's web cam
#DIAMETER_OF_BALL_IN_PIXEL_AT_CERTAIN_DISTANCE = 212 * 30 #In here 212 is the diameter in pixel and 30 cm is the distance of the camera from the ball.
#RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = math.pi * 106 * 106 / (960 * 540) #In photo taken from 30cm distance and with Hakan's phone.
#RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = (math.pi * 78 * 78) / (640 * 480) #In photo taken from 50cm distance with Hakan's computer cam.
#RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = (math.pi * 230 * 230) / (3420 * 2214) #In photo taken from 50cm distance with Zeynep's computer cam.
RATIO_OF_AREA_OF_BALL_AT_PHOTO_AT_CERTAIN_DISTANCE = (math.pi * 178 * 178) / (1920 * 1080) #In photo taken from 50cm distance with Hakan's web cam.
#CERTAIN_DISTANCE_BETWEEN_BALL_AND_PHOTO = 30.0
CERTAIN_DISTANCE_BETWEEN_BALL_AND_PHOTO = 50.0

DISTANCE_BETWEEN_ROBOT_INITIAL_AND_CAMERA: float = 27.5 # In cm. It should be added to line equation since we use right hand-side coordinate system, which causes robot's initisl point to stay at
                                                 # some point whose z component is negative. Therefore, every point in the coordinate system should be shifted this amount in Z axis 
                                                 # to locate robot's initial point to origin (0, 0).
ROBOT_RADIUS: float = 3.5 * 1.2
BALL_RADIUS: float = 4.2 * 1.2
directionVector: np.ndarray = np.array([0, 0, -1]) # Before I change my mathematical formulation, this vector's z componet had to be -1. Now, it does not matter whether it is -1 or 1. 
                                       # The only thing that matter is that this vector should be a unit vector and all of its components must be zero except it z component.
LEFT_BOUNDARY = -90.0 #In cm
RIGHT_BOUNDARY = 90.0 #In cm
NUMBER_OF_REGIONS = 5
WEIGHT_FOR_PASSES = 0

GLOBAL_X_MOMENTUM = 0
GLOBAL_Y_MOMENTUM = 0
GLOBAL_Z_MOMENTUM = 0
GLOBAL_STEP = 0
GLOBAL_MOMENTUM_POWER = 0.0

# -- UI Data Storage --
path_history = []
current_path = []
path_colors = ["red", "blue", "green", "purple", "orange", "cyan"]

def clamp(number, min, max):
    return (number < min) * min + (number > max) * max + (number >= min and number <= max) * number

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

#I wrote this function since I may use it in findPath part to filter outlier locations
def findIQR(array):
    arrayLength = len(array)
    sortedArray = np.sort(array)
    q1Index = (arrayLength + 1.0) / 4.0 - 1.0
    q3Index = 3 * (arrayLength + 1.0) / 4.0 - 1.0
    q1 = sortedArray[int(q1Index)] * (1.0 - (q1Index - int(q1Index))) + sortedArray[int(q1Index) + 1] * (q1Index - int(q1Index))
    q3 = sortedArray[int(q3Index)] * (1.0 - (q3Index - int(q3Index))) + sortedArray[int(q3Index) + 1] * (q3Index - int(q3Index))
    IQR = q3 - q1
    return IQR, q1, q3

def detectBall(frame):
    global GLOBAL_X_MOMENTUM, GLOBAL_Y_MOMENTUM, GLOBAL_Z_MOMENTUM, GLOBAL_MOMENTUM_POWER, GLOBAL_STEP

    blurredFrame = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))

    #lowerGreen = np.array([45, 56, 25])
    #upperGreen = np.array([79, 190, 255])
    lowerGreen = np.array([35, 56, 25])
    upperGreen = np.array([75, 199, 255])
    mask = cv2.inRange(hsv, lowerGreen, upperGreen)

    kernel = np.ones((5, 5), np.uint8)
    #mask = cv2.erode(mask, kernel, iterations=3)
    #mask = cv2.dilate(mask, kernel, iterations=3)

    cv2.imshow("Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ballPosition = None
    locationVector = None

    validContours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 400:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius ** 2
            circularity = area / circle_area if circle_area > 0 else 0

            if circularity > 0.6:
                validContours.append(contour)

    if len(validContours) > 0:
        max_contour = max(validContours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)

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
        GLOBAL_STEP += 1
        GLOBAL_X_MOMENTUM = GLOBAL_MOMENTUM_POWER * GLOBAL_X_MOMENTUM + (1 - GLOBAL_MOMENTUM_POWER) * locationVector[0]
        GLOBAL_Y_MOMENTUM = GLOBAL_MOMENTUM_POWER * GLOBAL_Y_MOMENTUM + (1 - GLOBAL_MOMENTUM_POWER) * locationVector[1]
        GLOBAL_Z_MOMENTUM = GLOBAL_MOMENTUM_POWER * GLOBAL_Z_MOMENTUM + (1 - GLOBAL_MOMENTUM_POWER) * locationVector[2]
        locationVector[0] = GLOBAL_X_MOMENTUM / (1 - GLOBAL_MOMENTUM_POWER**GLOBAL_STEP)
        locationVector[1] = GLOBAL_Y_MOMENTUM / (1 - GLOBAL_MOMENTUM_POWER**GLOBAL_STEP)
        locationVector[2] = GLOBAL_Z_MOMENTUM / (1 - GLOBAL_MOMENTUM_POWER**GLOBAL_STEP)
        #print(distance)
        #print(f"Location: {locationVector}, hoav: {horizontalAngle / math.pi * 180}, vaov: {verticalAngle / math.pi * 180}, {distance}")

    return frame, ballPosition, locationVector

def findPath(locations, momentum = 0.0):
    X = np.empty(len(locations))
    Y = np.empty(len(locations))
    Z = np.empty(len(locations))
    counter = 0

    #Sorting the locations from nearest to farthest. Assuming that point with greatest z component is the nearest.
    locationsSorted = sorted(locations, key= lambda location: location[2], reverse=True)
    momentumX: float = 0.0
    momentumY: float = 0.0
    momentumZ: float = 0.0

    for location in locationsSorted:
        momentumX = momentum * momentumX + (1 - momentum) * location[0]
        momentumY = momentum * momentumY + (1 - momentum) * location[1]
        momentumZ = momentum * momentumZ + (1 - momentum) * location[2]
        X[counter] = momentumX / (1 - pow(momentum, counter + 1))
        Y[counter] = momentumY / (1 - pow(momentum, counter + 1))
        Z[counter] = momentumZ / (1 - pow(momentum, counter + 1))
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
        print("Crash")
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
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = regionLeft + 2 * ROBOT_RADIUS
                else:
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = leftPoint
            elif rightPoint > regionLeft + 2 * ROBOT_RADIUS and rightPoint < regionRight - 2 * ROBOT_RADIUS and leftPoint <= regionLeft + 2 * ROBOT_RADIUS:
                if abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition) <= abs(rightPoint - robotCurrentPosition):
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = regionRight - 2 * ROBOT_RADIUS
                else:
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = rightPoint
            else:
                if abs(regionLeft + 2 * ROBOT_RADIUS - robotCurrentPosition) <= abs(regionRight - 2 * ROBOT_RADIUS - robotCurrentPosition):
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = regionLeft + 2 * ROBOT_RADIUS
                else:
                    if minCost > numberOfPassesPerRegion[i]:
                        minCost = numberOfPassesPerRegion[i]
                        pointWithMinimumCost = regionRight - 2 * ROBOT_RADIUS

        #print(f"Min cost: {minCost}")
        return pointWithMinimumCost - robotCurrentPosition
    else:
        print("Not move")
        return 0.0
    
def reset_data():
    global path_history, current_path, trajectory_pixels, locations
    path_history.clear()
    current_path.clear()
    trajectory_pixels.clear()
    locations.clear()
    print("All trajectory data has been reset.")


def show_path_popup():
    print("Popup triggered")
    popup = Toplevel()
    popup.title("Trajectory Viewer")
    popup.geometry("800x500")

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title("Previous Ball Paths")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")

    # Automatically plot all previous paths on popup
    for idx, path in enumerate(path_history):
        if len(path) >= 3:
            path = np.array(path)
            x_vals = path[:, 0]
            z_vals = -path[:, 2]
            z_vals = -z_vals
            coeffs = np.polyfit(x_vals, z_vals, 2)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
            z_fit = -np.polyval(coeffs, x_fit)
            ax.plot(x_fit, z_fit, color=path_colors[idx % len(path_colors)], label=f"Path {idx+1}")

    ax.legend(loc='upper right')  # Inside corner
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

if __name__ == "__main__":
    robotCurrentPosition = 0.0
    numberOfPassesPerRegion = np.zeros(NUMBER_OF_REGIONS)
    pathFinded: bool = False
    prevLocation = None
    firstLocation = True

    #camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW) #For Hakan's web cam
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800) #For Hakan's web cam
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 448) #For Hakan's web cam
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #For Hakan's web cam

    root = tk.Tk()
    root.withdraw()  # 🧹 Hides the default tk window completely

    control_panel = Toplevel()
    control_panel.title("Trajectory Control Panel")
    control_panel.geometry("+1200+100")

    button = tk.Button(control_panel, text="Show Past Trajectories", command=show_path_popup)
    button.pack()

    reset_button = tk.Button(control_panel, text="Reset Trajectories", command=reset_data)
    reset_button.pack(pady=10)

    locations = []
    start = time.time()
    trajectory_pixels = []

    while True:
        ret, frame = camera.read()
        timer = time.time()
        #frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
        frame = cv2.resize(frame, (800, 448))
        root.update_idletasks()
        root.update()
        if not ret:
            break
        #frame = cv2.flip(frame, 1)
        frame, ballPosition2d, location = detectBall(frame)

        if ballPosition2d:
            trajectory_pixels.append(ballPosition2d)  # 🟥 store pixel center
            if firstLocation == True:
                firstLocation = False
            else:
                if len(locations) >= 1:
                    generateLocation = (location + prevLocation) / 2.0
                    if np.linalg.norm(locations[len(locations) - 1] - generateLocation) >= 10.0:
                        print(np.linalg.norm(locations[len(locations) - 1] - generateLocation))
                        locations.append(generateLocation)
                else:
                    locations.append((location + prevLocation) / 2.0)
            prevLocation = location.copy()
            current_path.append(location)
            start = time.time()

        if (time.time() - start > 1.0):
            if current_path:
                path_history.append(current_path.copy())
                current_path.clear()
                trajectory_pixels.clear()
                locations.clear()
                pathFinded = False
                prevLocation = None
                firstLocation = True
                GLOBAL_STEP = 0
                GLOBAL_X_MOMENTUM = 0
                GLOBAL_Y_MOMENTUM = 0
                GLOBAL_Z_MOMENTUM = 0

        # ✅ Draw ball trajectory using center pixels
        if len(trajectory_pixels) >= 2:
            pts = np.array(trajectory_pixels, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)  # 🟡 Yellow curve

        if len(locations) >= 5 and pathFinded == False:
            coefficientsOfLine3D, coefficientsOfLine2D, boundries = findPath(locations=locations, momentum=0.3)
            numberOfPassesPerRegion[clamp(int((-coefficientsOfLine2D[1] / coefficientsOfLine2D[0] + LEFT_BOUNDARY) / ((RIGHT_BOUNDARY - LEFT_BOUNDARY) / NUMBER_OF_REGIONS)),0 , NUMBER_OF_REGIONS - 1)] += 1
            amountOfMovementRequired = findEvasionPoint(boundries, numberOfPassesPerRegion, robotCurrentPosition)
            pathFinded = True

            #Calculating path in UI
            if (abs(coefficientsOfLine3D[0][0]) < 1 and abs(coefficientsOfLine3D[1][0]) < 1):
                predictedLineUIPoint1 = [1000, 1000 * coefficientsOfLine3D[0][0] + coefficientsOfLine3D[0][1], 1000 * coefficientsOfLine3D[1][0] + coefficientsOfLine3D[1][1]]
                predictedLineUIPoint2 = [-1000, -1000 * coefficientsOfLine3D[0][0] + coefficientsOfLine3D[0][1], -1000 * coefficientsOfLine3D[1][0] + coefficientsOfLine3D[1][1]]

            elif (abs(coefficientsOfLine3D[0][0]) >= abs(coefficientsOfLine3D[1][0])):
                predictedLineUIPoint1 = [(1000 - coefficientsOfLine3D[0][1]) / coefficientsOfLine3D[0][0], 1000, coefficientsOfLine3D[1][0] / coefficientsOfLine3D[0][0] * (1000 - coefficientsOfLine3D[0][1]) + coefficientsOfLine3D[1][1]]
                predictedLineUIPoint2 = [(-1000 - coefficientsOfLine3D[0][1]) / coefficientsOfLine3D[0][0], -1000, coefficientsOfLine3D[1][0] / coefficientsOfLine3D[0][0] * (-1000 - coefficientsOfLine3D[0][1]) + coefficientsOfLine3D[1][1]]
            else:
                predictedLineUIPoint1 = [(-1000 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0], (-1000 - coefficientsOfLine3D[1][1]) * coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0] + coefficientsOfLine3D[0][1], -1000]
                predictedLineUIPoint2 = [(-1 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0], (-1 - coefficientsOfLine3D[1][1]) * coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0] + coefficientsOfLine3D[0][1], -1]

            if (predictedLineUIPoint1[2] > 0):
                predictedLineUIPoint1 = [(-1 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0], (-1 - coefficientsOfLine3D[1][1]) * coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0] + coefficientsOfLine3D[0][1], -1]
            elif (predictedLineUIPoint2[2] > 0):
                predictedLineUIPoint2 = [(-1 - coefficientsOfLine3D[1][1]) / coefficientsOfLine3D[1][0], (-1 - coefficientsOfLine3D[1][1]) * coefficientsOfLine3D[0][0] / coefficientsOfLine3D[1][0] + coefficientsOfLine3D[0][1], -1]

            predictedLineUIPoint1[0] /= -predictedLineUIPoint1[2]
            predictedLineUIPoint1[1] /= -predictedLineUIPoint1[2]
            predictedLineUIPoint1[2] = -1

            predictedLineUIPoint1[0] /= RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE
            predictedLineUIPoint1[1] /= RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE

            predictedLineUIPoint1[1] *= -1

            predictedLineUIPoint1[0] += 0.5
            predictedLineUIPoint1[1] += 0.5

            predictedLineUIPoint1[0] *= frame.shape[1]
            predictedLineUIPoint1[1] *= frame.shape[0]

            predictedLineUIPoint2[0] /= -predictedLineUIPoint2[2]
            predictedLineUIPoint2[1] /= -predictedLineUIPoint2[2]
            predictedLineUIPoint2[2] = -1

            predictedLineUIPoint2[0] /= RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE
            predictedLineUIPoint2[1] /= RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE

            predictedLineUIPoint2[1] *= -1

            predictedLineUIPoint2[0] += 0.5
            predictedLineUIPoint2[1] += 0.5

            predictedLineUIPoint2[0] *= frame.shape[1]
            predictedLineUIPoint2[1] *= frame.shape[0]

            #print(f"pt1: {predictedLineUIPoint1}, pt2: {predictedLineUIPoint2}")

            cv2.line(frame, \
                     (int(predictedLineUIPoint1[0]), int(predictedLineUIPoint1[1])), \
                     (int(predictedLineUIPoint2[0]), int(predictedLineUIPoint2[1])), \
                     color=(0, 0, 0), \
                        thickness=5)
            #sys.stdout.write("\033[K")  # Clear the entire line
            #print(f"Location: {location}, coefficients of line 3D {coefficientsOfLine3D}, coefficients of line 2D: {coefficientsOfLine2D}, boundries: {boundries} amount of movement required: {amountOfMovementRequired} pt1: {predictedLineUIPoint1}, pt2: {predictedLineUIPoint2}")
            #sys.stdout.write("\033[F")  # Move cursor up one line
            #sys.stdout.flush()
            
            leftPoint = min(boundries[0], boundries[1])
            rightPoint = max(boundries[0], boundries[1])

            minAmountMovement = 0.0
            if (leftPoint <= robotCurrentPosition and rightPoint >= robotCurrentPosition):
                if robotCurrentPosition - leftPoint < rightPoint - robotCurrentPosition:
                    minAmountMovement = leftPoint - robotCurrentPosition
                else:
                    minAmountMovement = rightPoint - robotCurrentPosition

            robotCurrentPosition += amountOfMovementRequired

            print(f"Current position of the robot: {robotCurrentPosition} boundries: {boundries} minumum amount of movement: {minAmountMovement} decision: {amountOfMovementRequired}")
            print(locations)

        elif pathFinded == True:
            cv2.line(frame, \
                     (int(predictedLineUIPoint1[0]), int(predictedLineUIPoint1[1])), \
                     (int(predictedLineUIPoint2[0]), int(predictedLineUIPoint2[1])), \
                     color=(0, 0, 0), \
                        thickness=5)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    root.destroy()
    cv2.destroyAllWindows()