import numpy as np
from .utils import gaussian_rbf, grav_vec

def compute_reward(env, action):
    pos_reward = sum(env.joint_error[i]**2 for i in range(len(env.jointIds)))
    vel_reward = sum(abs(env.vel_obs[i]) for i in range(len(env.jointIds)))
    action_diff = sum(((abs(action[i] - env.action_obs[i]) / 2)**2) for i in range(len(env.jointIds)))

    try:
        grav_reward = np.linalg.norm(grav_vec - env.rot_mat, ord=2)
        grav_reward = np.interp(grav_reward + np.random.normal(0, 0.05), [0, 2], [1, 0])
    except:
        grav_reward = 0

    # Base height
    height_reward = env.base_pos_obs[2] / 0.31 if env.base_pos_obs[2] < 0.31 else 1

    base_vel_reward = gaussian_rbf(env.base_lin_vel + np.random.normal(0, 0.5, 3), [0]*3, -2)

    foot_contact = sum(env.leg_contact_obs) / 4
    kc = 1 if grav_reward > 0.7 else 0

    reward = (
        0.3*(grav_reward) +
        0.3*height_reward +
        kc*(0.3*height_reward + 0.6*pos_reward/12 - 0.1*(1-foot_contact)) +
        0.05*(1 - action_diff/12) +
        0.005*(1 - vel_reward/12) +
        0.05*base_vel_reward
    )

    return float(reward)
