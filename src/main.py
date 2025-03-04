import numpy as np
import csv
import math

directionVector = np.array([0, 0, -1])

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

A = np.array([[np.dot(X, X), np.sum(X)], [np.sum(X), X.shape[0]]])
B1 = np.array([np.dot(X, Y), np.sum(Y)])
x1 = np.linalg.solve(A, B1)
print(f"Equation of y: {x1}")

B2 = np.array([np.dot(X, Z), np.sum(Z)])
x2 = np.linalg.solve(A, B2)
print(f"Equation of z: {x2}")

parallelVectorToLineInYZ = np.array([0, x1[0], x2[0]])
parallelVectorToLineInYZ /= np.linalg.norm(parallelVectorToLineInYZ)
angle = math.acos(np.dot(parallelVectorToLineInYZ, directionVector))

if angle >= math.pi - angle:
    angle = math.pi - angle

if (x1[0] > 0) != (x2[0] > 0):
    angle *= -1.0

print(f"Angle: {angle}")

newCoordinates = np.empty((X.shape[0], 3))

for i in range(X.shape[0]):
    newCoordinates[i, 0] = X[i]
    newCoordinates[i, 1] = math.cos(angle) * Y[i] - math.sin(angle) * Z[i]
    newCoordinates[i, 2] = math.sin(angle) * Y[i] + math.cos(angle) * Z[i]

print(newCoordinates)