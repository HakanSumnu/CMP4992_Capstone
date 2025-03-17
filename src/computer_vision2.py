import cv2
import numpy as np

if __name__ == "__main__":
    #camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("data/videos/green_ball_movement_slow.mp4")
    kernel = np.empty((7, 7), dtype=np.uint8)
    cv2.circle(kernel, (3, 3), 3, 255, -1)
    kernel2 = 255 - kernel

    while True:
        ret, frame = camera.read()
        if ret == False:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

        #cv2.GaussianBlur(frame, (3, 3), 1.0, dst=frame)

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(frameHSV, np.array([45, 76, 0]), np.array([75, 179, 255]))

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

        cv2.drawContours(frame, filteredContours, -1, (0, 0, 255), 3)

        cv2.imshow("Camera", frame)
        cv2.imshow("Camera color masked", colorMaskedFrame)

        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()