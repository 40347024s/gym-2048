import numpy as np
import gymnasium as gym

import gym2048


def _set_board(env, grid):
    # direct access to internal board for deterministic tests
    flat = np.array(grid, dtype=np.uint16).reshape(16)
    env.unwrapped._board.set_board(flat)


def _get_board(env):
    return env.unwrapped._board.get_board_view().copy()


def _non_zero_values(board):
    return board[board != 0]


def test_reset_has_two_tiles_and_values():
    env = gym.make("Gym2048-v0")
    obs, info = env.reset(seed=123)
    assert obs.shape == (4, 4)
    assert obs.dtype == np.uint16
    non_zero = _non_zero_values(obs)
    assert non_zero.size == 2
    assert set(non_zero.tolist()).issubset({2, 4})
    env.close()


def test_reset_same_seed_is_deterministic():
    env = gym.make("Gym2048-v0")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.array_equal(obs1, obs2)
    env.close()


def test_action_and_observation_space():
    env = gym.make("Gym2048-v0")
    obs, _ = env.reset(seed=1)
    assert env.action_space.n == 4
    assert env.action_space.contains(0)
    assert env.action_space.contains(3)
    assert env.observation_space.contains(obs)
    env.close()


def test_invalid_action_raises():
    env = gym.make("Gym2048-v0")
    env.reset(seed=1)
    try:
        env.step(5)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
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
    assert reward == 4.0
    board = _get_board(env)
    assert board[0, 0] == 4
    assert board[0, 1] == 0
    assert info["moved"] is True
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
    assert reward == 8.0
    board = _get_board(env)
    assert board[0, 3] == 4
    assert board[0, 2] == 4
    assert info["moved"] is True
    env.close()


def test_up_merge_single():
    env = gym.make("Gym2048-v0")
    env.reset(seed=3)
    _set_board(env, [
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    obs, reward, terminated, truncated, info = env.step(0)  # up
    board = _get_board(env)
    assert reward == 4.0
    assert board[0, 0] == 4
    assert board[1, 0] == 0
    env.close()


def test_down_merge_single():
    env = gym.make("Gym2048-v0")
    env.reset(seed=4)
    _set_board(env, [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [2, 0, 0, 0],
    ])
    obs, reward, terminated, truncated, info = env.step(1)  # down
    board = _get_board(env)
    assert reward == 4.0
    assert board[3, 0] == 4
    assert board[2, 0] == 0
    env.close()


def test_no_move_does_not_add_tile():
    env = gym.make("Gym2048-v0")
    env.reset(seed=5)
    _set_board(env, [
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    board_before = _get_board(env)
    obs, reward, terminated, truncated, info = env.step(2)  # left (no change)
    board_after = _get_board(env)
    assert reward == 0.0
    assert info["moved"] is False
    assert np.array_equal(board_before, board_after)
    env.close()


def test_terminal_state():
    env = gym.make("Gym2048-v0")
    env.reset(seed=6)
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
    obs, info = env.reset(seed=7)
    obs[0, 0] = 999
    obs2, info2 = env.reset(seed=7)
    assert obs2[0, 0] != 999
    env.close()


def test_copy_observation_false_shares_view():
    env = gym.make("Gym2048-v0", copy_observation=False)
    obs, info = env.reset(seed=8)
    obs[0, 0] = 888
    board = _get_board(env)
    assert board[0, 0] == 888
    env.close()


def test_render_ansi_returns_string():
    env = gym.make("Gym2048-v0", render_mode="ansi")
    env.reset(seed=9)
    output = env.render()
    assert isinstance(output, str)
    lines = output.splitlines()
    assert len(lines) == 4
    env.close()


def test_render_human_returns_none():
    env = gym.make("Gym2048-v0", render_mode="human")
    env.reset(seed=10)
    result = env.render()
    assert result is None
    env.close()
