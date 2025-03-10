import numpy as np
import csv
import math

def getPositionsAndLine(points, distanceBetweenRobotInitialAndCamera: float = 10.0, robotDiameter: float = 5.0, ballDiameter: float = 5.0, robotCurrentPosition: float = 0.0):
    """
    This is a function to find two points on X axis where the robot touches the boundry of the ball's path.

    :param points: list of points with XYZ components
    :param distanceBetweenRobotInitialAndCamera: Since camera will not be on the robot, there must be distance between them. This parameter requires that distance in cm.
    :param robotDiameter: Diameter of the robot in cm.
    :param ballDiameter: Diameter of the ball in cm.
    :param robotCurrentPosition: Current position of the robot on X axis. Since Z component of it will always be zero, only getting value for X axis is enough.

    :return: It returns two points on X axis where the robot touches the boundry of the ball's path.
    """
    directionVector: np.ndarray = np.array([0, 0, -1]) # Before I change my mathematical formulation, this vector's z componet had to be -1. Now, it does not matter whether it is -1 or 1.
                                                       # The only thing that matter is that this vector should be a unit vector and all of its components must be zero except it z component.

    # Collecting XYZ compenets of the dots and storing these components in separate arrays
    X = []
    Y = []
    Z = []
    for row in points:
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[2]))

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

    # Calculating the angle between XZ plane at y = 0 and YZ components of the vector that is parallel to best fit line
    parallelVectorToLineClamppedToYZ: np.ndarray = np.array([0, x1[0], x2[0]])
    parallelVectorToLineClamppedToYZ /= np.linalg.norm(parallelVectorToLineClamppedToYZ)
    angle: float = math.acos(np.dot(parallelVectorToLineClamppedToYZ, directionVector))

    if angle >= math.pi - angle:
        angle = math.pi - angle

    if (x1[0] > 0) != (x2[0] > 0):
        angle *= -1.0

    #print(f"Angle: {angle}")

    # Rotating the best fit line and considering it in XZ coordinate system where robot's initial position is at the origin
    coefficientsOfNewLine: np.ndarray = np.array([math.sin(angle) * x1[0] + math.cos(angle) * x2[0],\
         math.sin(angle) * x1[1] + math.cos(angle) * x2[1] + distanceBetweenRobotInitialAndCamera])
    #print(f"Coefficients of the new line: {coefficientsOfNewLine}")

    # Finding two points on the X axis whose distances to best fit line are equal to the sum of diameters of both robot and the ball
    point1: float = ((robotDiameter + ballDiameter) * math.sqrt(coefficientsOfNewLine[0] * coefficientsOfNewLine[0] + 1.0) + coefficientsOfNewLine[1]) / -coefficientsOfNewLine[0]
    point2: float = ((robotDiameter + ballDiameter) * math.sqrt(coefficientsOfNewLine[0] * coefficientsOfNewLine[0] + 1.0) - coefficientsOfNewLine[1]) / coefficientsOfNewLine[0]
    #print(f"Point 1: {point1}, Point 2: {point2}")

    return {"positions": [point1, point2], "line": [coefficientsOfNewLine[0], coefficientsOfNewLine[1]]}