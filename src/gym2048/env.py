from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from .core import Board


class Gym2048Env(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, copy_observation: bool = False):
        self.render_mode = render_mode
        self.copy_observation = copy_observation
        self._board = Board()

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.iinfo(np.uint16).max,
            shape=(4, 4),
            dtype=np.uint16,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._board.reset(0 if seed is None else int(seed))
        obs = self._observe()
        info = {"score": 0}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        reward, terminated, moved = self._board.step(int(action))
        obs = self._observe()
        info = {"moved": bool(moved)}
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), False, info

    def render(self):
        board = self._board.get_board_view()
        if self.render_mode == "ansi":
            return self._render_ansi(board)
        if self.render_mode == "human":
            print(self._render_ansi(board))
            return None
        return None

    def _render_ansi(self, board: np.ndarray) -> str:
        lines = []
        for row in board:
            line = " ".join(f"{int(v):4d}" for v in row)
            lines.append(line)
        return "\n".join(lines)

    def _observe(self) -> np.ndarray:
        board = self._board.get_board_view()
        if self.copy_observation:
            return np.array(board, copy=True)
        return board
