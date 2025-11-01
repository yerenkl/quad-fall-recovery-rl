import pybullet as p
import numpy as np
import time
import pybullet_utils.bullet_client as bc
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R
import random

from .utils import (
    gaussian_rbf, quaternion_randomizer, grav_vec,
    get_joint_states, get_base_state
)
from .reward import compute_reward

SIM_TIME = 1/100
default_pos = [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148, 0.048225,0.690008,-1.254787,-0.050525,0.661355,-1.243304]
TOTAL_TIMESTEP = 1024

class CustomQuadEnv(Env):
    metadata = {"render_modes": ["human", "headless"], "render_fps": 30}

    def __init__(self, render_mode=None, episodes=1, isTrain=True):
        super().__init__()

        self.isTrain = isTrain
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client = bc.BulletClient(connection_mode=mode)

        self.client.setGravity(0,0,-9.81)
        self.client.setTimeStep(SIM_TIME)

        self.plane = self.client.loadURDF("./data/plane.urdf")
        self.client.changeDynamics(self.plane, -1, lateralFriction=1.0)

        self.robot = self.client.loadURDF("./data/a1/urdf/a1.urdf", [0,0,0.5], [0,0,0,1],
                                          flags=p.URDF_USE_SELF_COLLISION)

        self.jointIds = [
            j for j in range(self.client.getNumJoints(self.robot))
            if self.client.getJointInfo(self.robot,j)[2] in [p.JOINT_REVOLUTE,p.JOINT_PRISMATIC]
        ]

        self.observation_space = spaces.Box(low=np.array([-5]*30), high=np.array([5]*30), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1]*12), high=np.array([1]*12), dtype=np.float32)

        self._action_repeat = 10
        self.last_action = np.zeros(12)
        self.action_obs = np.zeros(12, dtype=np.float32)
        self.iteration = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        quat = quaternion_randomizer()
        self.client.resetBasePositionAndOrientation(self.robot, [0,0,0.5], quat)

        for j in self.jointIds:
            info = self.client.getJointInfo(self.robot,j)
            self.client.resetJointState(self.robot, j, random.uniform(info[8], info[9]))

        for _ in range(33):
            self.client.stepSimulation()
            time.sleep(SIM_TIME)

        self.get_observation()
        return self.arr_obs, {}

    def step(self, action):
        reward = compute_reward(self, action)
        for i in range(self._action_repeat):
            if not self.isTrain and (reward >= 0.8 * 1.55):
                action = self.last_action
            proc_action=self.ProcessAction(action,i)
            for idx, j in enumerate(self.jointIds):
                info = self.client.getJointInfo(self.robot,j)
                tgt = np.interp(proc_action[idx], [-1,1], [info[8],info[9]])
                self.client.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                                  targetPosition=tgt, force=info[10], maxVelocity=info[11])
            self.client.stepSimulation()
            if not self.isTrain:
                time.sleep(SIM_TIME)
        
        self.last_action = action
        self.action_obs = self.last_action.astype(np.float32)
        self.iteration += 1
        done = (self.iteration % TOTAL_TIMESTEP == 0)

        self.get_observation()
        return self.arr_obs, compute_reward(self, action), done, False, {}

    def get_observation(self):
        self.pos_obs = np.zeros(12, dtype=np.float32)
        self.vel_obs = np.zeros(12, dtype=np.float32)
        self.joint_error = np.zeros(12, dtype=np.float32)
        self.leg_contact_obs = np.zeros((4,), dtype=np.float32)

        joint_pos, joint_vel, joint_torque = get_joint_states(self.client, self.robot, self.jointIds)

        for i, j in enumerate(self.jointIds):
            info = self.client.getJointInfo(self.robot, j)
            lo, hi = info[8], info[9]
            max_vel = info[11]

            # noisy_pos = joint_pos[i] + np.random.normal(0, 0.1)
            # noisy_vel = joint_vel[i] + np.random.normal(0, 1)

            self.pos_obs[i] = np.interp(joint_pos[i], [lo, hi], [-1, 1])
            self.vel_obs[i] = np.interp(joint_vel[i], [-max_vel, max_vel], [-1, 1])

            dist = abs(joint_pos[i] - default_pos[i])
            span = max(abs(hi - default_pos[i]), abs(lo - default_pos[i]))
            self.joint_error[i] = np.interp(dist, [0, span], [1, 0])

        self.base_pos_obs, quat, self.base_lin_vel, self.base_ang_vel = get_base_state(self.client, self.robot)
        self.base_orientation_obs = quat

        r = R.from_quat(self.base_orientation_obs)
        self.rot_mat = np.linalg.inv(r.as_matrix()).dot(grav_vec)

        self.arr_obs = np.concatenate([
            self.pos_obs,
            self.vel_obs,
            self.rot_mat.astype(np.float32),
            self.base_ang_vel.astype(np.float32)
        ]).astype(np.float32)

        # foot contact
        for idx, link in enumerate([5,9,13,17]):
            self.leg_contact_obs[idx] = 1 if self.client.getContactPoints(self.robot,self.plane,link) else 0

    def ProcessAction(self, action, substep_count):
        if np.all(self.last_action != np.zeros(12)):
            lerp = float(substep_count + 1) / self._action_repeat
            proc_action = self.last_action + lerp * (action - self.last_action)
        else:
            proc_action = action

        return proc_action