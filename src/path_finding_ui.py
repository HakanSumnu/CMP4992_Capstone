from path_finding import getPositionsAndLine
import time
import pyglet as pg

class PathFindingWindow(pg.window.Window):

    def __init__(self, windowWidth, windowHeight, caption):
        super().__init__(width=windowWidth, height=windowHeight, caption=caption)
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.robotRadius: float = 30.0
        self.robotShape = pg.shapes.Circle(x=0.5 * self.windowWidth, y=0.75 * self.windowHeight, radius=self.robotRadius / 1000.0 * self.windowWidth, color=(255, 255, 255))
        self.robotSelected: bool = False

        self.ballRadius: float = 30.0

        points = [[-1,1,-2], [0,0,-1], [1,-1,0]]
        positionsAndLine = getPositionsAndLine(points, robotDiameter=self.robotRadius, ballDiameter=self.ballRadius)
        self.positions = positionsAndLine["positions"]
        self.positions.sort()
        print(self.positions)
        self.line = positionsAndLine["line"]
        print(self.line)
        self.pathShape = pg.shapes.Line(0, ((-500.0 * self.line[0] + self.line[1] + 500) / 1000.0 + 0.25) * self.windowHeight,\
            self.windowWidth, ((500.0 * self.line[0] + self.line[1] + 500) / 1000.0 + 0.25) * self.windowHeight, color=(255, 255, 0),\
                width=self.ballRadius * 2.0 / 1000.0 * self.windowWidth)

    def on_draw(self):
        self.clear()
        self.pathShape.draw()
        robotCurrentPosition = self.robotShape.x / self.windowWidth * 1000.0 - 500.0
        if robotCurrentPosition >= self.positions[0] and robotCurrentPosition <= self.positions[1]:
            self.robotShape.color = (255, 0, 0)
        else:
            self.robotShape.color = (255, 255, 255)
        self.robotShape.draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pg.window.mouse.LEFT:
            if self.robotSelected == True or (x - self.robotShape.x) * (x - self.robotShape.x) + (y - self.robotShape.y) * (y - self.robotShape.y) < self.robotRadius * self.robotRadius:
                self.robotSelected = True
                self.robotShape.x = x

    def on_mouse_release(self, x, y, button, modifiers):
        self.robotSelected = False

window = PathFindingWindow(600, 600, "Path Finding")
pg.app.run(1.0 / 60.0)