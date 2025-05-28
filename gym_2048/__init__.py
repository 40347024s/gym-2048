from gymnasium.envs.registration import register

register(
    id='Game2048-v0',
    entry_point='gym_2048.envs:Game2048Env',
    max_episode_steps=10000,
)

register(
    id='Norm2048-v0',
    entry_point='gym_2048.envs:NormGame2048Env_ver0',
    max_episode_steps=10000,
)

register(
    id='Norm2048-v1',
    entry_point='gym_2048.envs:NormGame2048Env_ver1',
    max_episode_steps=10000,
)