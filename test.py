import quad   
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import torch as th

MODEL_PATH = "./best_weights.zip"
RENDER_MODE = "human"    
SEED = 15


def make_env(seed=0, rank=0, render_mode="headless"):
    def _init():
        env = gymnasium.make(
            "CustomQuad",
            render_mode=render_mode,
            isTrain=False,        
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(seed=SEED, rank=0, render_mode=RENDER_MODE)])
    model = PPO.load(MODEL_PATH, env=env, device="cuda" if th.cuda.is_available() else "cpu")
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()
