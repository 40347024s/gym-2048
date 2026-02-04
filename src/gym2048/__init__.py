from gymnasium.envs.registration import register

register(
    id="Gym2048-v0",
    entry_point="gym2048.env:Gym2048Env",
)

from .env import Gym2048Env

__all__ = ["Gym2048Env"]
