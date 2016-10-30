import math
import sys
import time

import numpy as np
import pyglet
from sklearn import linear_model
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import random

import Arm
import Ball

angle = 60
yf = 400.0
total_steps = 200
friction = 0.3 #value between 0 and 1
pi = math.pi
mode = ""

def bezier(p, n):
    t = np.linspace(0, 1, n)
    return p[0] * (1 - t)**3 + p[1] * (1 - t)**2 * t*3 + p[2] * (1 - t) * (t**2) * 3 + p[3] * (t**3)

def calc_steps_linear(q0,qf):
    trajectory_theta1 = np.linspace(q0[0], qf[0], num=total_steps, endpoint=False)
    trajectory_theta2 = np.linspace(q0[1], qf[1], num=total_steps, endpoint=False)

    return trajectory_theta1,trajectory_theta2

def calc_steps_exponential(q0,qf):
    base_traj = 2.0**(np.linspace(0, 1, num=total_steps, endpoint=False))

    #print base_traj

    normalized_traj = (base_traj - min(base_traj)) / (max(base_traj) - min(base_traj))
    trajectory_theta1 = normalized_traj * (qf[0] - q0[0]) + q0[0]
    trajectory_theta2 = normalized_traj * (qf[1] - q0[1]) + q0[1]

    return trajectory_theta1,trajectory_theta2

def calc_steps_cubic(q0,qf):
    base_traj = np.linspace(-1, 0, num=total_steps, endpoint=False) ** 3

    normalized_traj = (base_traj - min(base_traj)) / (max(base_traj) - min(base_traj))
    trajectory_theta1 = normalized_traj * (qf[0] - q0[0]) + q0[0]
    trajectory_theta2 = normalized_traj * (qf[1] - q0[1]) + q0[1]

    return trajectory_theta1,trajectory_theta2


def calc_steps_sigmoid(q0, qf):

    temperature = 5.0

    tmp1 = math.e ** (np.linspace(-1, 1, total_steps))
    tmp2 = tmp1 ** (-temperature)
    base_traj = 1. / (1 + tmp2)

    normalized_traj = (base_traj - min(base_traj)) / (max(base_traj) - min(base_traj))

    trajectory_theta1 = normalized_traj * (qf[0]- q0[0]) + q0[0]
    trajectory_theta2 = normalized_traj * (qf[1] - q0[1]) + q0[1]

    return trajectory_theta1, trajectory_theta2

def calc_steps_bezier(q0, qf):
    normalized_traj = bezier([0,0.7,0.3,1], total_steps)
    trajectory_theta1 = normalized_traj * (qf[0] - q0[0]) + q0[0]
    trajectory_theta2 = normalized_traj * (qf[1] - q0[1]) + q0[1]

    return trajectory_theta1, trajectory_theta2

def calc_steps_mixed(q0,qf):
    trajectory_theta1 = calc_steps_bezier(q0,qf)[0]
    trajectory_theta2 = calc_steps_sigmoid(q0,qf)[1]

    return trajectory_theta1,trajectory_theta2

def normalize(value, oldmin, oldmax, newmin, newmax):
    newvalue = (((float(value) - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
    return newvalue

def calculate_pesos_angles(first,second):
    angles = [round(x * 0.1, 1) for x in range(-650, 651)]
    regr_first = linear_model.LinearRegression(fit_intercept=False)
    regr_second = linear_model.LinearRegression(fit_intercept=False)

    steps_matrix = np.matrix([
                              [1 for angle in angles],
                              [np.sin(pi * normalize(angle, -65, 65, 0, 1)) for angle in angles],
                              [np.cos(pi * normalize(angle, -65, 65, 0, 1)) for angle in angles],
                              [np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) for angle in angles],
                              [np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) for angle in angles],
                              [np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) for angle in angles],
                              [np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3) for angle in angles]#,
                              #[np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) for angle in angles],
                              #[np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) for angle in angles],
                              #[np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) for angle in angles],
                              #[np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5) for angle in angles],
                             ]).T

    regr_first.fit(steps_matrix,first)
    regr_second.fit(steps_matrix,second)

    return regr_first.coef_,regr_second.coef_

def convert_deltas(v):
    deltas = []
    deltas.append(v[0])

    for i in range(1,len(v)):
        deltas.append(v[i]-v[i-1])

    return deltas

def convert_trajectory(v,init):
    trajectory = []
    trajectory.append(init)

    for i in range(0,len(v)):
        trajectory.append(v[i]+trajectory[-1])

    return trajectory


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

    steps_matrix = np.matrix([[1 for t in range(total_steps-1)],
                              [np.sin(pi*t/200) for t in range(total_steps-1)],[np.cos(pi*t/200) for t in range(total_steps-1)],
                              [np.sin(pi*t*2/200) for t in range(total_steps-1)],[np.cos(pi*t*2/200) for t in range(total_steps-1)],
                              [np.sin(pi*t*3/200) for t in range(total_steps-1)],[np.cos(pi*t*3/200) for t in range(total_steps-1)]]).T

    regr_first.fit(steps_matrix,first)
    regr_second.fit(steps_matrix,second)

    return regr_first.coef_,regr_second.coef_


def angles_function(t,mode):

    if(mode == "array_angles"):
        #array of angles
        first_joint = pesos_first[t]
        second_joint = pesos_second[t]
        return [first_joint,second_joint,0.0]

    elif(mode == "array_deltas"):
        #array of deltas
        window.last_first = window.last_first + pesos_first[t]
        window.last_second = window.last_second + pesos_second[t]
        return [window.last_first,window.last_second,0.0]

    elif(mode == "weights_deltas"):
        #with regression and deltas
        window.last_first = window.last_first + pesos_first[0] + \
                            pesos_first[1] * np.sin(pi*t/200) + pesos_first[2] * np.cos(pi*t/200) + \
                            pesos_first[3] * np.sin(pi*t*2/200) + pesos_first[4] * np.cos(pi*t*2/200) + \
                            pesos_first[5] * np.sin(pi*t*3/200) + pesos_first[6] * np.cos(pi*t*3/200)
        window.last_second =  window.last_second + pesos_second[0] + \
                              pesos_second[1] * np.sin(pi*t/200) + pesos_second[1] * np.cos(pi*t/200) + \
                              pesos_second[2] * np.sin(pi*t*2/200) + pesos_second[2] * np.cos(pi*t*2/200) + \
                              pesos_second[3] * np.sin(pi*t*3/200) + pesos_second[3] * np.cos(pi*t*3/200)
        return [window.last_first, window.last_second, 0.0]

    elif(mode == "weights_angles"):
        # with regression and angles
        first_joint = pesos_first[0] + \
                            pesos_first[1] * np.sin(pi * t / 200) + pesos_first[2] * np.cos(pi * t / 200) + \
                            pesos_first[3] * np.sin(pi * t * 2 / 200) + pesos_first[4] * np.cos(pi * t * 2 / 200) + \
                            pesos_first[5] * np.sin(pi * t * 3 / 200) + pesos_first[6] * np.cos(pi * t * 3 / 200)
        second_joint = pesos_second[0] + \
                             pesos_second[1] * np.sin(pi * t / 200) + pesos_second[2] * np.cos(pi * t / 200) + \
                             pesos_second[3] * np.sin(pi * t * 2 / 200) + pesos_second[4] * np.cos(pi * t * 2 / 200) + \
                             pesos_second[5] * np.sin(pi * t * 3 / 200) + pesos_second[6] * np.cos(pi * t * 3 / 200)
        return [first_joint, second_joint, 0.0]


def update(dt):
    label.text = '(x,y) = (%.3f, %.3f)' % (ball.x, ball.y)
    window.traj_theta1.append(arm.q[0])
    window.traj_theta2.append(arm.q[1])
    arm.q = angles_function(window.step,mode)
    window.step = window.step + 1
    window.jps = get_joint_positions()  # get new joint (x,y) positions
    label.text = 'ball = (%.3f, %.3f)' % (ball.x, ball.y)
    ball.update(total_steps)

class Simulation():
    def __init__(self):

        #push the handlers
        #self.push_handlers(EventHandler(self))

        #pyglet.window.Window.__init__(self, 1000, 600)
        self.width = 1000
        self.height = 600

        pyglet.clock.schedule_interval(update, 1/60.0)

        self.step = 0

        self.last_first = 0.0
        self.last_second = 0.0

        self.traj_theta1 = []
        self.traj_theta2 = []

    def set_jps(self):
        self.jps = get_joint_positions()  # get new joint (x,y) positions

    def on_draw(self):
        self.clear()
        if(self.step == 199):
            arm_xy = arm.get_xy(arm.q)
            arm_xy = (arm_xy[0] + (window.width / 2), arm_xy[1])
            ball_xy = [ball.xf,ball.yf]
            error = np.sqrt((np.array(ball_xy) - np.array(arm_xy)) ** 2)
            print error
        label.draw()
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
        for i in range(3):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i',
                                                         (self.jps[0][i], self.jps[1][i],
                                                          self.jps[0][i + 1], self.jps[1][i + 1])))

        if(self.step == 199):
            '''plt.figure(1)
            plt.subplot(211)
            line1, = plt.plot(trajectory_theta1,color='green',label='Expected')
            line2, = plt.plot(self.traj_theta1,color='blue',label='Real')
            plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc=4)
            plt.title("Real x Expected Theta1")
            plt.ylabel("Angle")

            plt.subplot(212)
            line1, = plt.plot(trajectory_theta2, color='green', label='Expected')
            line2, = plt.plot(self.traj_theta2, color='blue', label='Real')
            plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc=4)
            plt.title("Real x Expected Theta2")
            plt.xlabel("Step")
            plt.ylabel("Angle")

            #plt.show()'''

            sys.exit()

    '''def on_mouse_motion(self,x, y, dx, dy):
        # call the inverse kinematics function of the arm
        # to find the joint angles optimal for pointing at
        # this position of the mouse
        label.text = '(x,y) = (%.3f, %.3f)' % (x, y)
        arm.q = arm.inv_kin([x - self.width / 2, y])  # get new arm angles
        window.jps = get_joint_positions()  # get new joint (x,y) positions'''

trajectory_normal_joint1 = [[],[],[],[],[],[],[]]
trajectory_normal_joint2 = [[],[],[],[],[],[],[]]
trajectory_friction_joint1 = [[],[],[],[],[],[],[]]
trajectory_friction_joint2 = [[],[],[],[],[],[],[]]
deltas_normal_joint1 = [[],[],[],[],[],[],[]]
deltas_normal_joint2 =[[],[],[],[],[],[],[]]
deltas_friction_joint1 = [[],[],[],[],[],[],[]]
deltas_friction_joint2 = [[],[],[],[],[],[],[]]

angles = [round(x * 0.1,1) for x in range(-650, 651)]


#with open("trajectory_normal.csv","wb") as tn:
    #with open("trajectory_friction.csv","wb") as tf:
        #with open("deltas_normal.csv","wb") as dn:
            #with open("deltas_friction.csv", "wb") as df:
                #tn.write("angle" + "," + "pesos_first" + "," + "pesos_second" + "\n")
                #tf.write("angle" + "," + "pesos_first" + "," + "pesos_second" + "\n")
                #dn.write("angle" + "," + "pesos_first" + "," + "pesos_second" + "\n")
                #df.write("angle" + "," + "pesos_first" + "," + "pesos_second" + "\n")

for angle in [round(x * 0.1,1) for x in range(-650, 651)]:
    # create an instance of the arm
    arm = Arm.Arm3Link(L=np.array([400, 200, 0]))

    window = Simulation()
    window.set_jps()

    # create an instance of the ball
    ball = Ball.Ball(float(window.width / 2), float(window.height), math.radians(angle), yf)

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
    qf = arm.inv_kin([ball.xf - (window.width / 2), ball.yf])
    #how much each joint need to move each step
    #trajectory_theta1,trajectory_theta2 = calc_steps_linear(q0,qf)
    #trajectory_theta1,trajectory_theta2 = calc_steps_exponential(q0,qf)
    #trajectory_theta1,trajectory_theta2 = calc_steps_cubic(q0,qf)
    #trajectory_theta1,trajectory_theta2 = calc_steps_sigmoid(q0,qf)
    #trajectory_theta1,trajectory_theta2 = calc_steps_bezier(q0,qf)
    trajectory_theta1,trajectory_theta2 = calc_steps_mixed(q0,qf)

    # first position
    window.last_first = trajectory_theta1[0]
    window.last_second = trajectory_theta2[0]

    # calculate weights that predict the trajectory without friction
    pesos_trajectory_without_friction_first, pesos_trajectory_without_friction_second = calculate_pesos(trajectory_theta1[1:],trajectory_theta2[1:])

    # convert the trajectory to deltas
    deltas_theta1, deltas_theta2 = convert_deltas(trajectory_theta1), convert_deltas(trajectory_theta2)

    # calculate weights that predict the deltas without friction
    pesos_without_friction_first, pesos_without_friction_second = calculate_pesos(deltas_theta1[1:], deltas_theta2[1:])

    # friction applied in the deltas
    deltas_friction_theta1 = [p * (1.0 - friction) for p in deltas_theta1[1:]]
    deltas_friction_theta2 = [p * (1.0 - friction) for p in deltas_theta2[1:]]

    # calculate weights that predict the deltas with friction
    pesos_with_friction_first, pesos_with_friction_second = calculate_pesos(deltas_friction_theta1, deltas_friction_theta2)

    # convert the frictioned deltas to a frictioned trajectory
    trajectory_friction_theta1, trajectory_friction_theta2 = convert_trajectory(deltas_friction_theta1,trajectory_theta1[0]), convert_trajectory(deltas_friction_theta2, trajectory_theta2[0])

    # calculate weights that predict the trajectory with friction
    pesos_trajectory_with_friction_first, pesos_trajectory_with_friction_second = calculate_pesos(trajectory_friction_theta1[1:], trajectory_friction_theta2[1:])

    for i in range(0,7):
        trajectory_normal_joint1[i].append(pesos_trajectory_without_friction_first[i])
        trajectory_normal_joint2[i].append(pesos_trajectory_without_friction_second[i])
        trajectory_friction_joint1[i].append(pesos_trajectory_with_friction_first[i])
        trajectory_friction_joint2[i].append(pesos_trajectory_with_friction_second[i])
        deltas_normal_joint1[i].append(pesos_without_friction_first[i])
        deltas_normal_joint2[i].append(pesos_without_friction_second[i])
        deltas_friction_joint1[i].append(pesos_with_friction_first[i])
        deltas_friction_joint2[i].append(pesos_with_friction_second[i])

    if angle % 1 == 0:
        print angle

'''#generation of angulo -> wi_primeira_regressao
predicted_weights_trajectory_normal_joint1 = []
predicted_weights_trajectory_normal_joint2 = []
predicted_weights_trajectory_friction_joint1 = []
predicted_weights_trajectory_friction_joint2 = []
predicted_weights_deltas_normal_joint1 = []
predicted_weights_deltas_normal_joint2 =[]
predicted_weights_deltas_friction_joint1 = []
predicted_weights_deltas_friction_joint2 = []

for i in range(0,7):
    a, b = calculate_pesos_angles(deltas_normal_joint1[i],deltas_normal_joint2[i])
    predicted_weights_deltas_normal_joint1.append(a)
    predicted_weights_deltas_normal_joint2.append(b)

    a, b = calculate_pesos_angles(deltas_friction_joint1[i], deltas_friction_joint2[i])
    predicted_weights_deltas_friction_joint1.append(a)
    predicted_weights_deltas_friction_joint2.append(b)

    a, b = calculate_pesos_angles(trajectory_normal_joint1[i], trajectory_normal_joint2[i])
    predicted_weights_trajectory_normal_joint1.append(a)
    predicted_weights_trajectory_normal_joint2.append(b)

    a, b = calculate_pesos_angles(trajectory_friction_joint1[i], trajectory_friction_joint2[i])
    predicted_weights_trajectory_friction_joint1.append(a)
    predicted_weights_trajectory_friction_joint2.append(b)

predicted_values_trajectory_normal_joint1 = []
predicted_values_trajectory_normal_joint2 = []
predicted_values_trajectory_friction_joint1 = []
predicted_values_trajectory_friction_joint2 = []
predicted_values_deltas_normal_joint1 = []
predicted_values_deltas_normal_joint2 =[]
predicted_values_deltas_friction_joint1 = []
predicted_values_deltas_friction_joint2 = []

for i in range(0,7):
    b = []
    for angle in angles:
        a = predicted_weights_deltas_normal_joint1[i][0] + \
            predicted_weights_deltas_normal_joint1[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_normal_joint1[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_normal_joint1[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_normal_joint1[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_normal_joint1[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_deltas_normal_joint1[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_deltas_normal_joint1[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_normal_joint1[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_normal_joint1[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_deltas_normal_joint1[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_deltas_normal_joint1.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_deltas_normal_joint2[i][0] + \
            predicted_weights_deltas_normal_joint2[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_normal_joint2[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_normal_joint2[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_normal_joint2[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_normal_joint2[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_deltas_normal_joint2[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_deltas_normal_joint2[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_normal_joint2[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_normal_joint2[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_deltas_normal_joint2[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_deltas_normal_joint2.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_deltas_friction_joint1[i][0] + \
            predicted_weights_deltas_friction_joint1[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_friction_joint1[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_friction_joint1[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_friction_joint1[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_friction_joint1[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_deltas_friction_joint1[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_friction_joint1[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_friction_joint1[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_deltas_friction_joint1[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_deltas_friction_joint1.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_deltas_friction_joint2[i][0] + \
            predicted_weights_deltas_friction_joint2[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_friction_joint2[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_deltas_friction_joint2[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_friction_joint2[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_deltas_friction_joint2[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_deltas_friction_joint2[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_deltas_friction_joint2[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_friction_joint2[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_deltas_friction_joint2[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_deltas_friction_joint2[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_deltas_friction_joint2.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_trajectory_normal_joint1[i][0] + \
            predicted_weights_trajectory_normal_joint1[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_normal_joint1[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_normal_joint1[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_normal_joint1[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_normal_joint1[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_trajectory_normal_joint1[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_trajectory_normal_joint1[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_normal_joint1[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_normal_joint1[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_trajectory_normal_joint1[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_trajectory_normal_joint1.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_trajectory_normal_joint2[i][0] + \
            predicted_weights_trajectory_normal_joint2[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_normal_joint2[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_normal_joint2[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_normal_joint2[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_normal_joint2[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_trajectory_normal_joint2[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_trajectory_normal_joint2[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_normal_joint2[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_normal_joint2[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_trajectory_normal_joint2[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_trajectory_normal_joint2.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_trajectory_friction_joint1[i][0] + \
            predicted_weights_trajectory_friction_joint1[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_friction_joint1[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_friction_joint1[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_friction_joint1[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_friction_joint1[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_trajectory_friction_joint1[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_trajectory_friction_joint1[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_friction_joint1[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_friction_joint1[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_trajectory_friction_joint1[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_trajectory_friction_joint1.append(b)

    b = []
    for angle in angles:
        a = predicted_weights_trajectory_friction_joint2[i][0] + \
            predicted_weights_trajectory_friction_joint2[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_friction_joint2[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
            predicted_weights_trajectory_friction_joint2[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_friction_joint2[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
            predicted_weights_trajectory_friction_joint2[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
            predicted_weights_trajectory_friction_joint2[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)# + \
            #predicted_weights_trajectory_friction_joint2[i][7] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_friction_joint2[i][8] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 4) + \
            #predicted_weights_trajectory_friction_joint2[i][9] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 5) + \
            #predicted_weights_trajectory_friction_joint2[i][10] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 5)
        b.append(a)
    predicted_values_trajectory_friction_joint2.append(b)


for i in range(0,7):
    plt.plot(angles,deltas_normal_joint1[i],label='values')
    plt.plot(angles,predicted_values_deltas_normal_joint1[i],label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_deltas_normal_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_deltas_normal_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,deltas_normal_joint2[i],label='values')
    plt.plot(angles,predicted_values_deltas_normal_joint2[i],label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_deltas_normal_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_deltas_normal_joint2_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,deltas_friction_joint1[i],label='values')
    plt.plot(angles,predicted_values_deltas_friction_joint1[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_deltas_friction_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_deltas_friction_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,deltas_friction_joint2[i],label='values')
    plt.plot(angles,predicted_values_deltas_friction_joint2[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_deltas_friction_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_deltas_friction_joint2_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,trajectory_normal_joint1[i],label='values')
    plt.plot(angles,predicted_values_trajectory_normal_joint1[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_trajectory_normal_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_trajectory_normal_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,trajectory_normal_joint2[i],label='values')
    plt.plot(angles,predicted_values_trajectory_normal_joint2[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_trajectory_normal_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_trajectory_normal_joint2_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,trajectory_friction_joint1[i],label='values')
    plt.plot(angles,predicted_values_trajectory_friction_joint1[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_trajectory_friction_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_trajectory_friction_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles,trajectory_friction_joint2[i],label='values')
    plt.plot(angles,predicted_values_trajectory_friction_joint2[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('w' + str(i))
    plt.title("compar_trajectory_friction_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_trajectory_friction_joint2_w" + str(i) + ".png")
    plt.clf()'''

#generation of angulo -> wi_semFriccao-wi_comFriccao
#generation of angulo -> wi_semFriccao/wi_comFriccao
#ONLY WORKING WITH TRAJECTORY

'''for i in range(0,7):
    nome_arq_delta = "w" + str(i) + "-delta_training_set.csv"
    nome_arq_ratio = "w" + str(i) + "-ratio_training_set.csv"
    with open(nome_arq_delta,"wb") as arq_delta:
        header = "angle,w"+str(i)+"_semFriccao-w"+str(i)+"_comFriccao_joint1,w"+str(i)+"_semFriccao-w"+str(i)+"_comFriccao_joint2\n"
        arq_delta.write(header)
        with open(nome_arq_ratio,"wb") as arq_ratio:
            header = "angle,w"+str(i)+"_semFriccao/w"+str(i)+"_comFriccao_joint1,w"+str(i)+"_semFriccao/w"+str(i)+"_comFriccao_joint2\n"
            arq_ratio.write(header)
            for angle in angles:
                arq_delta.write(str(angle)+",")
                arq_ratio.write(str(angle)+",")

                angle_index = angles.index(angle)

                wi_semFriccao_joint1 = trajectory_normal_joint1[i][angle_index]
                wi_comFriccao_joint1 = trajectory_friction_joint1[i][angle_index]
                wi_semFriccao_joint2 = trajectory_normal_joint2[i][angle_index]
                wi_comFriccao_joint2 = trajectory_friction_joint2[i][angle_index]

                joint1_delta = wi_semFriccao_joint1-wi_comFriccao_joint1
                joint1_ratio = wi_semFriccao_joint1/wi_comFriccao_joint1
                joint2_delta = wi_semFriccao_joint2-wi_comFriccao_joint2
                joint2_ratio = wi_semFriccao_joint2/wi_comFriccao_joint2

                arq_delta.write(str(joint1_delta) + "," + str(joint2_delta) + "\n")
                arq_ratio.write(str(joint1_ratio) + "," + str(joint2_ratio) + "\n")

                if angle % 1 == 0:
                    print str(angle) + "(w" + str(i) + ")"
        arq_ratio.close()
    arq_delta.close()'''


#performing regression2
joint1_deltas = [[],[],[],[],[],[],[]]
joint2_deltas = [[],[],[],[],[],[],[]]
joint1_ratios = [[],[],[],[],[],[],[]]
joint2_ratios = [[],[],[],[],[],[],[]]

#creating ratios and deltas storage
for i in range(0,7):
    for angle in angles:

        angle_index = angles.index(angle)

        wi_semFriccao_joint1 = trajectory_normal_joint1[i][angle_index]
        wi_comFriccao_joint1 = trajectory_friction_joint1[i][angle_index]
        wi_semFriccao_joint2 = trajectory_normal_joint2[i][angle_index]
        wi_comFriccao_joint2 = trajectory_friction_joint2[i][angle_index]

        joint1_delta = wi_semFriccao_joint1 - wi_comFriccao_joint1
        joint1_ratio = wi_semFriccao_joint1 / wi_comFriccao_joint1
        joint2_delta = wi_semFriccao_joint2 - wi_comFriccao_joint2
        joint2_ratio = wi_semFriccao_joint2 / wi_comFriccao_joint2

        joint1_deltas[i].append(joint1_delta)
        joint1_ratios[i].append(joint1_ratio)
        joint2_deltas[i].append(joint2_delta)
        joint2_ratios[i].append(joint2_ratio)

        if angle % 1 == 0:
            print str(angle) + "(w" + str(i) + ")"


weights_joint1_deltas = []
weights_joint1_ratios = []
weights_joint2_deltas = []
weights_joint2_ratios = []

#performing regressions
for i in range(0,7):
    weights_joint1_delta_i, weights_joint2_delta_i = calculate_pesos_angles(joint1_deltas[i],joint2_deltas[i])
    weights_joint1_deltas.append(weights_joint1_delta_i)
    weights_joint2_deltas.append(weights_joint2_delta_i)

    weights_joint1_ratio_i, weights_joint2_ratio_i = calculate_pesos_angles(joint1_ratios[i], joint2_ratios[i])
    weights_joint1_ratios.append(weights_joint1_ratio_i)
    weights_joint2_ratios.append(weights_joint2_ratio_i)

#generating predicted for each angle
predicted_joint1_deltas = [[],[],[],[],[],[],[]]
predicted_joint1_ratios = [[],[],[],[],[],[],[]]
predicted_joint2_deltas = [[],[],[],[],[],[],[]]
predicted_joint2_ratios = [[],[],[],[],[],[],[]]

for i in range(0,7):
    for angle in angles:
        predicted_joint1_delta_i = weights_joint1_deltas[i][0] + \
                                   weights_joint1_deltas[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
                                   weights_joint1_deltas[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
                                   weights_joint1_deltas[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
                                   weights_joint1_deltas[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2) + \
                                   weights_joint1_deltas[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3) + \
                                   weights_joint1_deltas[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3)
        predicted_joint1_deltas[i].append(predicted_joint1_delta_i)

        predicted_joint2_delta_i = weights_joint2_deltas[i][0] + \
                                   weights_joint2_deltas[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1)) + \
                                   weights_joint2_deltas[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1)) + \
                                   weights_joint2_deltas[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint2_deltas[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint2_deltas[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3 ) + \
                                   weights_joint2_deltas[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3 )
        predicted_joint2_deltas[i].append(predicted_joint2_delta_i)

        predicted_joint1_ratio_i = weights_joint1_ratios[i][0] + \
                                   weights_joint1_ratios[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1) ) + \
                                   weights_joint1_ratios[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1) ) + \
                                   weights_joint1_ratios[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint1_ratios[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint1_ratios[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3 ) + \
                                   weights_joint1_ratios[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3 )
        predicted_joint1_ratios[i].append(predicted_joint1_ratio_i)

        predicted_joint2_ratio_i = weights_joint2_ratios[i][0] + \
                                   weights_joint2_ratios[i][1] * np.sin(pi * normalize(angle, -65, 65, 0, 1) ) + \
                                   weights_joint2_ratios[i][2] * np.cos(pi * normalize(angle, -65, 65, 0, 1) ) + \
                                   weights_joint2_ratios[i][3] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint2_ratios[i][4] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 2 ) + \
                                   weights_joint2_ratios[i][5] * np.sin(pi * normalize(angle, -65, 65, 0, 1) * 3 ) + \
                                   weights_joint2_ratios[i][6] * np.cos(pi * normalize(angle, -65, 65, 0, 1) * 3 )
        predicted_joint2_ratios[i].append(predicted_joint2_ratio_i)


#generating graphs for comparing
for i in range(0,7):
    plt.plot(angles, joint1_ratios[i], label='values')
    plt.plot(angles, predicted_joint1_ratios[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('joint1_w' + str(i))
    plt.title("compar_ratio_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_ratio_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles, joint2_ratios[i], label='values')
    plt.plot(angles, predicted_joint2_ratios[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('joint2_w' + str(i))
    plt.title("compar_ratio_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_ratio_joint2_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles, joint1_deltas[i], label='values')
    plt.plot(angles, predicted_joint1_deltas[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('joint1_w' + str(i))
    plt.title("compar_delta_joint1_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_delta_joint1_w" + str(i) + ".png")
    plt.clf()

    plt.plot(angles, joint2_deltas[i], label='values')
    plt.plot(angles, predicted_joint2_deltas[i], label='regression')
    plt.xlabel('angle')
    plt.ylabel('joint2_w' + str(i))
    plt.title("compar_delta_joint2_w" + str(i))
    plt.grid(True)
    plt.savefig("compar_delta_joint2_w" + str(i) + ".png")
    plt.clf()


