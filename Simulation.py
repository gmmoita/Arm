import numpy as np
import pyglet
import time
import math
from sklearn import linear_model

import Ball
import Arm

yf = 400.0
total_steps = 200

def get_joint_positions():
    """This method finds the (x,y) coordinates of each joint"""

    x = np.array([0,
                  arm.L[0] * np.cos(arm.q[0]),
                  arm.L[0] * np.cos(arm.q[0]) + arm.L[1] * np.cos(arm.q[0] + arm.q[1]),
                  arm.L[0] * np.cos(arm.q[0]) + arm.L[1] * np.cos(arm.q[0] + arm.q[1]) +
                  arm.L[2] * np.cos(np.sum(arm.q))]) + window.width / 2

    y = np.array([0,
                  arm.L[0] * np.sin(arm.q[0]),
                  arm.L[0] * np.sin(arm.q[0]) + arm.L[1] * np.sin(arm.q[0] + arm.q[1]),
                  arm.L[0] * np.sin(arm.q[0]) + arm.L[1] * np.sin(arm.q[0] + arm.q[1]) +
                  arm.L[2] * np.sin(np.sum(arm.q))])

    return np.array([x, y]).astype('int')

def calculate_pesos(first,second):
    regr_first = linear_model.LinearRegression(fit_intercept=False)
    regr_second = linear_model.LinearRegression(fit_intercept=False)

    steps_matrix = np.matrix([[1 for t in range(total_steps)],
                              [t for t in range(total_steps)],
                              [(t**2) for t in range(total_steps)],
                              [(t**3) for t in range(total_steps)]]).T

    regr_first.fit(steps_matrix,first)
    regr_second.fit(steps_matrix,second)

    return regr_first.coef_,regr_second.coef_


def angles_function(t):
    first_joint = pesos_first[0] + pesos_first[1] * t + pesos_first[2] * (t**2) + pesos_first[3] * (t**3)
    second_joint = pesos_second[0] + pesos_second[1] * t + pesos_second[2] * (t**2) + pesos_second[3] * (t**3)

    return [first_joint,second_joint,0.0]


def update(dt):
    label.text = '(x,y) = (%.3f, %.3f)' % (ball.x, ball.y)
    arm.q = angles_function(window.step)
    window.step = window.step + 1
    window.jps = get_joint_positions()  # get new joint (x,y) positions
    print window.jps
    ball.update(total_steps)

class Simulation(pyglet.window.Window):
    def __init__(self):

        #push the handlers
        #self.push_handlers(EventHandler(self))

        pyglet.window.Window.__init__(self, 1000, 600)

        pyglet.clock.schedule_interval(update, 1/60.0)

        self.step = 0

    def set_jps(self):
        window.jps = get_joint_positions()  # get new joint (x,y) positions

    def on_draw(self):
        self.clear()
        label.draw()
        for i in range(3):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i',
                                                         (self.jps[0][i], self.jps[1][i],
                                                          self.jps[0][i + 1], self.jps[1][i + 1])))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                               ('v2f', (ball.x - 10, ball.y - 10,
                                        ball.x + 10, ball.y - 10,
                                        ball.x + 10, ball.y + 10,
                                        ball.x - 10, ball.y + 10)),
                                ('c3B', (255, 0, 0,
                                         255, 0, 0,
                                         255, 0, 0,
                                         255, 0, 0)))

        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i',
                                                     (self.width // 2 , self.height,
                                                      self.width // 2, 0)),
                                                    ('c3B', (0, 0, 255,
                                                             0, 0, 255)))

        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f',
                                                     (ball.x0, ball.y0,
                                                      ball.xf, ball.yf)),
                                                    ('c3B', (0, 255, 0,
                                                             0, 255, 0)))

    '''def on_mouse_motion(self,x, y, dx, dy):
        # call the inverse kinematics function of the arm
        # to find the joint angles optimal for pointing at
        # this position of the mouse
        label.text = '(x,y) = (%.3f, %.3f)' % (x, y)
        arm.q = arm.inv_kin([x - self.width / 2, y])  # get new arm angles
        window.jps = get_joint_positions()  # get new joint (x,y) positions'''


# create an instance of the arm
arm = Arm.Arm3Link(L=np.array([400, 200, 0]))

window = Simulation()
window.set_jps()

# create an instance of the ball
ball = Ball.Ball(float(window.width / 2), float(window.height), math.radians(1), yf)

distance = math.hypot((ball.xf - (window.width / 2)), (ball.yf - 0))

if distance > arm.reach:
    print "NAO VAI ALCANCAR!"
    raise

label = pyglet.text.Label('Mouse (x,y)', font_name='Times New Roman',
                                  font_size=36, x=window.width // 2, y=window.height // 2,
                                  anchor_x='center', anchor_y='center')

#initial configuration of the arm
q0 = arm.q
#calculate final angles
qf = arm.inv_kin([ball.xf, ball.yf])
#how much each joint need to move each step
angles_first_joint = np.linspace(q0[0],qf[0],num=total_steps, endpoint=False)
angles_second_joint = np.linspace(q0[1],qf[1],num=total_steps,  endpoint=False)

pesos_first,pesos_second = calculate_pesos(angles_first_joint,angles_second_joint)

print pesos_first
print pesos_second


pyglet.app.run()
