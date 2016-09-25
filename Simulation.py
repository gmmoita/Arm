import numpy as np
import pyglet
import time
import math

import Ball
import Arm

yf = 400
n_passos = 200

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

def update(dt):
    label.text = '(x,y) = (%.3f, %.3f)' % (ball.xf, ball.yf)
    arm.q = arm.inv_kin([ball.xf - window.width / 2, ball.yf])  # get new arm angles
    window.jps = get_joint_positions()  # get new joint (x,y) positions

class Simulation(pyglet.window.Window):
    def __init__(self):

        #push the handlers
        #self.push_handlers(EventHandler(self))

        pyglet.window.Window.__init__(self,1000, 600)

        pyglet.clock.schedule_interval(update, 1 / 60.0)

    def set_jps(self):
        self.jps = get_joint_positions()

    def on_draw(self):
        self.clear()
        label.draw()
        ball.update(n_passos)
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
ball = Ball.Ball(window.width / 2, window.height, math.radians(60), yf)

distance = math.hypot((ball.xf - (window.width // 2)), (ball.yf - 0))

if distance > arm.reach:
    print "NAO VAI ALCANCAR!"
    raise

label = pyglet.text.Label('Mouse (x,y)', font_name='Times New Roman',
                                  font_size=36, x=window.width // 2, y=window.height // 2,
                                  anchor_x='center', anchor_y='center')

pyglet.app.run()
