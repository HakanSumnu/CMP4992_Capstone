import cv2
import numpy as np
import math

RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE = 29.5 / 24
RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE = 29.5 * (9 / 16) / 24

if __name__ == "__main__":
    screenShotCounter = 0
    #camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("data/videos/green_ball_movement_slow.mp4")
    kernel = np.empty((7, 7), dtype=np.uint8)
    cv2.circle(kernel, (3, 3), 3, 255, -1)
    kernel2 = 255 - kernel

    locations = []

    while True:
        ret, frame = camera.read()
        if ret == False:
            break

        if frame.shape[1] > frame.shape[0]:
            frame = cv2.resize(frame, (960, 540))
        else:
            frame = cv2.resize(frame, (540, 960))

        #cv2.GaussianBlur(frame, (3, 3), 1.0, dst=frame)

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(frameHSV, np.array([45, 76, 25]), np.array([75, 179, 255]))

        colorMaskedFrame = cv2.bitwise_and(frame, frame, mask=colorMask)
        colorMaskedFrame = cv2.cvtColor(colorMaskedFrame, cv2.COLOR_BGR2GRAY)
        ret, colorMaskedFrame = cv2.threshold(colorMaskedFrame, 1, 255, cv2.THRESH_BINARY)

        #cv2.dilate(colorMaskedFrame, kernel=kernel2, dst=colorMaskedFrame, iterations=2)
        #cv2.erode(colorMaskedFrame, np.ones((3,3)), dst=colorMaskedFrame, iterations=4)

        contours, hierarchy = cv2.findContours(colorMaskedFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filteredContours = []

        for contour in contours:
            if cv2.contourArea(contour=contour) >= 100:
                filteredContours.append(contour)
                x,y,w,h = cv2.boundingRect(contour)
                if w / (w + h) >= 0.45 and w / (w + h) <= 0.55:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, "Ball", (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
                    horizontalAngle = (x + w / 2 - frame.shape[1] / 2) / frame.shape[1]
                    verticalAngle = (y + h / 2 - frame.shape[0] / 2) / frame.shape[0]
                    if frame.shape[1] > frame.shape[0]:
                        horizontalAngle = math.atan(horizontalAngle * RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE)
                        verticalAngle = math.atan(verticalAngle * RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE)
                    else:
                        horizontalAngle = math.atan(horizontalAngle * RATIO_OF_HEIGHT_AND_PERPENDICULAR_DISTANCE)
                        verticalAngle = math.atan(verticalAngle * RATIO_OF_WIDTH_AND_PERPENDICULAR_DISTANCE)
                    locationVector = np.array([0, 0, -1])
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
                    perpendicularDistance = 212 / w * 30.0
                    distance = perpendicularDistance / (math.cos(horizontalAngle) * math.cos(verticalAngle))
                    locationVector *= distance
                    locations.append(f"{locationVector[0]},{locationVector[1]},{locationVector[2]}\n")

        cv2.drawContours(frame, filteredContours, -1, (0, 0, 255), 3)

        cv2.imshow("Camera", frame)
        cv2.imshow("Camera color masked", colorMaskedFrame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord('s'):
            cv2.imwrite("screenshot" + str(screenShotCounter) + ".jpg", frame)
            screenShotCounter += 1

    with open("locations.csv", 'w') as file:
        for location in locations:
            file.write(location)

    cv2.destroyAllWindows()
    camera.release()