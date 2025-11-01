import numpy as np
import random
from scipy.spatial.transform import Rotation as R

grav_vec = np.array([0,0,-1])

def gaussian_rbf(x, y, c):
    distance_squared = np.sum((x - y)**2)
    return np.exp(distance_squared*c)

def normalizer(x):
    var = np.exp(x)/(1+np.exp(x))
    return np.interp(var, [0, 1], [-1, 1])

def quaternion_randomizer():
    x0 = random.uniform(0,1)
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1)
    theta1 = 2*np.pi*x1
    theta2 = 2*np.pi*x2
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    r1 = np.sqrt(1-x0)
    r2 = np.sqrt(x0)
    return [s1*r1, c1*r1, s2*r2, c2*r2]


# ----- State extraction helpers -----

def get_joint_states(client, robot, joints):
    pos, vel, torque = [], [], []
    for j in joints:
        js = client.getJointState(robot, j)
        pos.append(js[0]); vel.append(js[1]); torque.append(js[3])
    return np.array(pos), np.array(vel), np.array(torque)


def get_base_state(client, robot):
    pos, quat = client.getBasePositionAndOrientation(robot)
    lin, ang = client.getBaseVelocity(robot)
    return (np.array(pos, dtype=np.float32),
            np.array(quat, dtype=np.float32),
            np.array(lin, dtype=np.float32),
            np.array(ang, dtype=np.float32))
