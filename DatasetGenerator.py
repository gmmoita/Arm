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

def calculate_pesos_angles(first,second, angles):
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

class PseudoSimulation():
    def __init__(self):

        #push the handlers
        #self.push_handlers(EventHandler(self))

        #pyglet.window.Window.__init__(self, 1000, 600)
        self.width = 1000
        self.height = 600

        #pyglet.clock.schedule_interval(update, 1/60.0)

        self.step = 0

        self.last_first = 0.0
        self.last_second = 0.0

        self.traj_theta1 = []
        self.traj_theta2 = []


'''
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

    window = PseudoSimulation()
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
        print angle'''

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

'''
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
        predicted_joint2_ratios[i].append(predicted_joint2_ratio_i)'''


#generating graphs for comparing
'''for i in range(0,7):
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

#anotate in an txt the weights
with open("regression2_weights.csv",'wb') as file:
    file.write("Type;w0;w1;w2;w3;w4,w5,w6\n")

    file.write("Joint1_ratios;")
    for weight in weights_joint1_ratios:
        file.write(str(weight) + ';')
    file.write("\n")

    file.write("Joint2_ratios;")
    for weight in weights_joint2_ratios:
        file.write(str(weight) + ';')
    file.write("\n")

    file.write("Joint1_deltas;")
    for weight in weights_joint1_deltas:
        file.write(str(weight) + ';')
    file.write("\n")

    file.write("Joint2_deltas;")
    for weight in weights_joint2_deltas:
        file.write(str(weight) + ';')

    file.close()'''


def second_recursion(type, angles, friction_type):

    trajectory_normal_joint1 = [[], [], [], [], [], [], []]
    trajectory_normal_joint2 = [[], [], [], [], [], [], []]
    trajectory_friction_joint1 = [[], [], [], [], [], [], []]
    trajectory_friction_joint2 = [[], [], [], [], [], [], []]
    deltas_normal_joint1 = [[], [], [], [], [], [], []]
    deltas_normal_joint2 = [[], [], [], [], [], [], []]
    deltas_friction_joint1 = [[], [], [], [], [], [], []]
    deltas_friction_joint2 = [[], [], [], [], [], [], []]

    for angle in angles:
        # create an instance of the arm
        arm = Arm.Arm3Link(L=np.array([400, 200, 0]))

        window = PseudoSimulation()

        # create an instance of the ball
        ball = Ball.Ball(float(window.width / 2), float(window.height), math.radians(angle), yf)

        distance = math.hypot((ball.xf - (window.width / 2)), (ball.yf - 0))

        if distance > arm.reach:
            print "NAO VAI ALCANCAR!"
            raise

        #label = pyglet.text.Label('Mouse (x,y)', font_name='Times New Roman',
        #                          font_size=36, x=window.width // 2, y=window.height // 2,
         #                         anchor_x='center', anchor_y='center')

        # initial configuration of the arm
        q0 = arm.q
        # calculate final angles
        qf = arm.inv_kin([ball.xf - (window.width / 2), ball.yf])
        # how much each joint need to move each step
        # trajectory_theta1,trajectory_theta2 = calc_steps_linear(q0,qf)
        # trajectory_theta1,trajectory_theta2 = calc_steps_exponential(q0,qf)
        # trajectory_theta1,trajectory_theta2 = calc_steps_cubic(q0,qf)
        # trajectory_theta1,trajectory_theta2 = calc_steps_sigmoid(q0,qf)
        # trajectory_theta1,trajectory_theta2 = calc_steps_bezier(q0,qf)
        trajectory_theta1, trajectory_theta2 = calc_steps_mixed(q0, qf)

        # first position
        window.last_first = trajectory_theta1[0]
        window.last_second = trajectory_theta2[0]

        # calculate weights that predict the trajectory without friction
        pesos_trajectory_without_friction_first, pesos_trajectory_without_friction_second = calculate_pesos(
            trajectory_theta1[1:], trajectory_theta2[1:])

        # convert the trajectory to deltas
        deltas_theta1, deltas_theta2 = convert_deltas(trajectory_theta1), convert_deltas(trajectory_theta2)

        # calculate weights that predict the deltas without friction
        pesos_without_friction_first, pesos_without_friction_second = calculate_pesos(deltas_theta1[1:],
                                                                                      deltas_theta2[1:])

        # friction applied in the deltas (simple)
        if friction_type == "simple":
            factors = [1] * len(deltas_theta1[1:])
        # friction applied in the deltas
        # 1 - Linear
        if friction_type == "linear":
            factors = list(np.linspace(0.0, 1.0, len(deltas_theta1[1:])))
        # 2 - Sigmoid
        elif friction_type == "sigmoid":
            factors = calc_steps_sigmoid([0.0, 0.0], [1.0, 1.0])[0]
        # 3 - Inverse Sigmoid
        elif friction_type == "inv_sigmoid":
            factors = calc_steps_sigmoid([0.0, 0.0], [1.0, 1.0])[0]
            factors = list(reversed(factors))
        # no friction
        else:
            factors = [0] * len(deltas_theta1[1:])
        # applying friction
        deltas_friction_theta1 = [deltas_theta1[1:][i] * (1.0 - (friction * factors[i])) for i in
                                  range(len(deltas_theta1[1:]))]
        deltas_friction_theta2 = [deltas_theta2[1:][i] * (1.0 - (friction * factors[i])) for i in
                                  range(len(deltas_theta2[1:]))]

        # calculate weights that predict the deltas with friction
        pesos_with_friction_first, pesos_with_friction_second = calculate_pesos(deltas_friction_theta1,
                                                                                deltas_friction_theta2)

        # convert the frictioned deltas to a frictioned trajectory
        trajectory_friction_theta1, trajectory_friction_theta2 = convert_trajectory(deltas_friction_theta1,
                                                                                    trajectory_theta1[
                                                                                        0]), convert_trajectory(
            deltas_friction_theta2, trajectory_theta2[0])

        # calculate weights that predict the trajectory with friction
        pesos_trajectory_with_friction_first, pesos_trajectory_with_friction_second = calculate_pesos(
            trajectory_friction_theta1[1:], trajectory_friction_theta2[1:])

        for i in range(0, 7):
            trajectory_normal_joint1[i].append(pesos_trajectory_without_friction_first[i])
            trajectory_normal_joint2[i].append(pesos_trajectory_without_friction_second[i])
            trajectory_friction_joint1[i].append(pesos_trajectory_with_friction_first[i])
            trajectory_friction_joint2[i].append(pesos_trajectory_with_friction_second[i])
            deltas_normal_joint1[i].append(pesos_without_friction_first[i])
            deltas_normal_joint2[i].append(pesos_without_friction_second[i])
            deltas_friction_joint1[i].append(pesos_with_friction_first[i])
            deltas_friction_joint2[i].append(pesos_with_friction_second[i])

        #if angle % 1 == 0:
        #    print angle

    # performing regression2
    joint1_deltas = [[], [], [], [], [], [], []]
    joint2_deltas = [[], [], [], [], [], [], []]
    joint1_ratios = [[], [], [], [], [], [], []]
    joint2_ratios = [[], [], [], [], [], [], []]

    # creating ratios and deltas storage
    for i in range(0, 7):
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

            #if angle % 1 == 0:
            #    print str(angle) + "(w" + str(i) + ")"


    if type == "deltas":
        weights_joint1_deltas = []
        weights_joint2_deltas = []

        for i in range(0, 7):
            weights_joint1_delta_i, weights_joint2_delta_i = calculate_pesos_angles(joint1_deltas[i], joint2_deltas[i], angles)
            weights_joint1_deltas.append(weights_joint1_delta_i)
            weights_joint2_deltas.append(weights_joint2_delta_i)

        return weights_joint1_deltas, weights_joint2_deltas

    elif type == "ratios":
        weights_joint1_ratios = []
        weights_joint2_ratios = []

        for i in range(0, 7):
            weights_joint1_ratio_i, weights_joint2_ratio_i = calculate_pesos_angles(joint1_ratios[i], joint2_ratios[i], angles)
            weights_joint1_ratios.append(weights_joint1_ratio_i)
            weights_joint2_ratios.append(weights_joint2_ratio_i)

        return weights_joint1_ratios, weights_joint2_ratios

    else:
        raise #error in type



