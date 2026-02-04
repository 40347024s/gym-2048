import numpy as np
import gymnasium as gym

import gym2048


def _set_board(env, grid):
    # direct access to internal board for deterministic tests
    flat = np.array(grid, dtype=np.uint16).reshape(16)
    env.unwrapped._board.board[:] = flat


def _get_board(env):
    return env.unwrapped._board.get_board_view().copy()


def test_reset_has_two_tiles():
    env = gym.make("Gym2048-v0")
    obs, info = env.reset(seed=123)
    assert obs.shape == (4, 4)
    non_zero = np.count_nonzero(obs)
    assert non_zero == 2
    env.close()


def test_left_merge_single():
    env = gym.make("Gym2048-v0", copy_observation=True)
    env.reset(seed=1)
    _set_board(env, [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    obs, reward, terminated, truncated, info = env.step(2)  # left
    # There will be a random tile added; check the merge result in first row
    assert reward == 4.0
    board = _get_board(env)
    assert board[0, 0] == 4
    assert board[0, 1] == 0
    env.close()


def test_right_merge_chain():
    env = gym.make("Gym2048-v0")
    env.reset(seed=2)
    _set_board(env, [
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    obs, reward, terminated, truncated, info = env.step(3)  # right
    # 2+2 and 2+2 => reward 8
    assert reward == 8.0
    board = _get_board(env)
    assert board[0, 3] == 4
    assert board[0, 2] == 4
    env.close()


def test_terminal_state():
    env = gym.make("Gym2048-v0")
    env.reset(seed=3)
    _set_board(env, [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    env.close()


def test_copy_observation_true_returns_copy():
    env = gym.make("Gym2048-v0", copy_observation=True)
    obs, info = env.reset(seed=4)
    obs[0, 0] = 999
    obs2, info2 = env.reset(seed=4)
    # If copy_observation is True, external mutation should not affect internal board
    assert obs2[0, 0] != 999
    env.close()


def test_copy_observation_false_shares_view():
    env = gym.make("Gym2048-v0", copy_observation=False)
    obs, info = env.reset(seed=5)
    # write into view and expect internal board to reflect
    obs[0, 0] = 888
    board = _get_board(env)
    assert board[0, 0] == 888
    env.close()
