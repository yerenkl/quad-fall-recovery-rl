import time
import gymnasium as gym
import quad 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

from config import *

def make_env(seed, render_mode):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env.reset(seed=seed)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i, RENDER_MODE) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LR_RATE,
        gamma=GAMMA,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        n_epochs=N_EPOCHS,
        verbose=1,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=LOGDIR
    )

    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=LOGDIR,
        name_prefix="rl_model",
    )

    event_callback = EveryNTimesteps(N_STEPS * N_ENVS * 10, callback=checkpoint)

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=event_callback
    )
    print(f"Training time: {time.time() - start:.2f}s")
