from gymnasium.envs.registration import register

register(
    id="CustomQuad",                       
    entry_point="quad.envs:CustomQuadEnv",  
)
