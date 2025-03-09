import numpy as np
import csv
import math

DISTANCE_BETWEEN_ROBOT_INITIAL_AND_CAMERA: float = 10.0 # In cm. It should be added to line equation since we use right hand-side coordinate system, which causes robot's initisl point to stay at
                                                 # some point whose z component is negative. Therefore, every point in the coordinate system should be shifted this amount in Z axis 
                                                 # to locate robot's initial point to origin (0, 0).
directionVector: np.ndarray = np.array([0, 0, -1]) # Before I change my mathematical formulation, this vector's z componet had to be -1. Now, it does not matter whether it is -1 or 1. 
                                       # The only thing that matter is that this vector should be a unit vector and all of its components must be zero except it z component.

# Opening test file and collecting XYZ compenets of the dots and storing these components in separate arrays.
with open("data/test.csv") as file:
    file = csv.reader(file)
    X = []
    Y = []
    Z = []
    for row in file:
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
print(f"Equation of y: {x1}")

B2: np.ndarray = np.array([np.dot(X, Z), np.sum(Z)])
x2: np.ndarray = np.linalg.solve(A, B2)
print(f"Equation of z: {x2}")

# Calculating the angle between XZ plane at y = 0 and YZ components of the vector that is parallel to best fit line.
parallelVectorToLineClamppedToYZ: np.ndarray = np.array([0, x1[0], x2[0]])
parallelVectorToLineClamppedToYZ /= np.linalg.norm(parallelVectorToLineClamppedToYZ)
angle: float = math.acos(np.dot(parallelVectorToLineClamppedToYZ, directionVector))

if angle >= math.pi - angle:
    angle = math.pi - angle

if (x1[0] > 0) != (x2[0] > 0):
    angle *= -1.0

print(f"Angle: {angle}")

## Calculating rotated positions
##newCoordinates = np.empty((X.shape[0], 3))
##
##for i in range(X.shape[0]):
##    newCoordinates[i, 0] = X[i]
##    newCoordinates[i, 1] = math.cos(angle) * Y[i] - math.sin(angle) * Z[i]
##    newCoordinates[i, 2] = math.sin(angle) * Y[i] + math.cos(angle) * Z[i]
##
##print(newCoordinates)

#parallelVectorToLineIn2D = np.array([1, math.cos(angle) * x1[0] - math.sin(angle) * x2[0], math.sin(angle) * x1[0] + math.cos(angle) * x2[0]])
#print(f"A vector that is parallel to the rotated line: {parallelVectorToLineIn2D}")
#
#constantForComponents = np.array([0, math.cos(angle) * x1[1] - math.sin(angle) * x2[1], math.sin(angle) * x1[1] + math.cos(angle) * x2[1]])
#print(f"Constants for components after the rotation: {constantForComponents}")

# Rotating the best fit line and considering it in XZ coordinate system
coefficientsOfNewLine: np.ndarray = np.array([math.sin(angle) * x1[0] + math.cos(angle) * x2[0], math.sin(angle) * x1[1] + math.cos(angle) * x2[1]])
print(f"Coefficients of the new line: {coefficientsOfNewLine}")