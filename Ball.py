import math


def final_pos(x0, y0, yf, angle):  # angle is between (-math.pi/2,math.pi/2)
    xf = (math.tan(angle) * (y0 - yf)) + x0
    return round(xf, 3)


class Ball:
    def __init__(self, x0, y0, angle, yf):
        self.x0 = x0
        self.y0 = y0
        self.angle = angle
        self.yf = yf
        self.xf = final_pos(self.x0, self.y0, self.yf, self.angle)
        self.x = self.x0
        self.y = self.y0
        self.deltax = self.xf - self.x0
        self.deltay = self.yf - self.y0

    def update(self, n_passos):
        self.x += self.deltax / n_passos
        self.y = self.y + (self.deltay / n_passos)



