import torch as th

# Environment
ENV_ID = "CustomQuad"
N_ENVS = 8
RENDER_MODE = "headless"

# RL Hyperparameters
LR_RATE = 3e-4
GAMMA = 0.97
N_STEPS = 1024
BATCH_SIZE = 32 * 8
MAX_GRAD_NORM = 0.7
VF_COEF = 0.3
ENT_COEF = 0.0
N_EPOCHS = 10

TOTAL_TIMESTEPS = N_STEPS * 600 * N_ENVS  

# Policy Architecture
POLICY_KWARGS = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(
        pi=[256,256,16],
        vf=[256,256,16]
    )
)

# Logging & Checkpoints
LOGDIR = "./saves"
CHECKPOINT_FREQ = 1
