import gymnasium as gym
from itertools import count
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pygame
import typing as tp


@nb.jit(nopython=True, nogil=False)
def slide2right(input_board: np.ndarray):
    reward = 0
    board = input_board.copy()
    for row in board:
        ids = np.where(row > 0)
        if ids[0].shape[0] == 0.0:
            continue
        selected_row = row[ids]
        cur_tail = len(selected_row) - 1
        new_ids = []
        while cur_tail > 0:
            if selected_row[cur_tail] == selected_row[cur_tail - 1]:
                row[ids[0][cur_tail]] *= 2
                reward += row[ids[0][cur_tail]]
                row[ids[0][cur_tail - 1]] = 0
                new_ids.append(ids[0][cur_tail])
                cur_tail -= 2
            else:
                cur_tail -= 1
        ids = np.where(row > 0)
        if ids[0].shape[0] == 0.0:
            continue
        row[-len(ids[0]):] = row[ids]
        row[:-len(ids[0])] = 0.0

    return reward, board

@nb.jit(nopython=True, nogil=False)
def get_valid_action_mask(input_board: np.ndarray):
    ava_action_mask = np.array([1, 1, 1, 1])
    for i in range(4):
        tmp_state = np.rot90(input_board, i)
        _, tmp_state = slide2right(tmp_state)
        tmp_state = np.rot90(tmp_state, -i)
        if (tmp_state==input_board).all():
            ava_action_mask[i] = 0

    return ava_action_mask

@nb.jit(nopython=True, nogil=False)
def is_game_over(input_board: np.ndarray):
    if len(np.where(input_board==0)[0]) > 0:
        return False
    for i in range(2):
        board = input_board.copy()
        board = np.rot90(board, i)
        tmp_board = slide2right(board)[1]
        if (tmp_board!=board).any():
            return False
    
    return True

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 8}

    def __init__(self, render_mode=None, size=4) -> None:
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('monospace', 24)

        self.size = size
        self.window_size = 450
        self.cmap = plt.get_cmap('Spectral')
        self.val2cmap_key: tp.Dict[float, float] = {2.0**i: i/20 for i in range(1, 20)}
        self.cur_iteration = 0
        self.max_iteration = 10000

        self.observation_space = gym.spaces.Box(low=0, high=2**20, shape=(4, 4), dtype=float)

        # corresponding to "right", "down", "left", "up"
        self.action_space = gym.spaces.Discrete(4)
        self.action_meaning = ["right", "down", "left", "up"]

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cur_iteration = 0
        self.state = np.zeros(self.size**2)
        self.state[:2] = np.random.choice([1]*9+[2]*1, 2)
        self.state = 2**self.state
        np.random.shuffle(self.state)
        self.state[np.where(self.state==1)] = 0
        self.state = self.state.reshape(self.size, self.size)

        if self.render_mode == "human":
            self._render_frame()

        return self.encode_state(self.state), {'max_val': np.max(self.state)}
    
    def step(self, action):
        reward = 0.0
        prev_state = self.state.copy()
        self.state = np.rot90(self.state, action)
        truncated = False
        terminated = False
        for row in self.state:
            ids = np.where(row > 0)
            if ids[0].shape[0] == 0.0:
                continue
            selected_row = row[ids]
            cur_tail = len(selected_row) - 1
            new_ids = []
            while cur_tail > 0:
                if selected_row[cur_tail] == selected_row[cur_tail - 1]:
                    row[ids[0][cur_tail]] *= 2
                    reward += row[ids[0][cur_tail]]
                    row[ids[0][cur_tail - 1]] = 0
                    new_ids.append(ids[0][cur_tail])
                    cur_tail -= 2
                else:
                    cur_tail -= 1
            ids = np.where(row > 0)
            if ids[0].shape[0] == 0.0:
                continue
            row[-len(ids[0]):] = row[ids]
            row[:-len(ids[0])] = 0.0
        self.state = np.rot90(self.state, (4 - action) % 4)

        ids = np.where(self.state == 0.0)
        if len(ids[0]) == 0 and ((self.state[:, 1:] - self.state[:, :-1])!=0).all() and ((self.state[1:, :] - self.state[:-1, :])!=0).all():
            terminated = True
        else:
            if (prev_state!=self.state).any():
                rand_id = np.random.randint(len(ids[0]))
                self.state[ids[0][rand_id]][ids[1][rand_id]] = np.random.choice([2, 4], p=[0.9, 0.1])
            else:
                truncated = True

        if self.render_mode == "human":
            self._render_frame()

        self.cur_iteration += 1
        terminated = terminated or (self.cur_iteration == self.max_iteration)

        return self.encode_state(self.state), reward, terminated, truncated, {'max_val': np.max(self.state), 'original_reward': reward}
    
    def encode_state(self, input_board):
        return input_board.copy()

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = 100
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j] == 0.0:
                    color = (225, 225, 225, 1.0)
                else:
                    color = np.array(self.cmap(self.val2cmap_key[self.state[i][j]]))
                    color[:3] *= 255
                    
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (10*(j+1)+j*pix_square_size, 10*(i+1)+i*pix_square_size),
                        (pix_square_size, pix_square_size)
                    )
                )
                if self.state[i][j] > 0:
                    text = self.font.render(str(int(self.state[i][j])), False, (0, 0, 0))
                    text_rect = text.get_rect(center=(10*(j+1)+j*pix_square_size + pix_square_size / 2, 
                                                      10*(i+1)+i*pix_square_size + pix_square_size / 2))
                    canvas.blit(text, text_rect)
                

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def valid_action_mask(self):
        return get_valid_action_mask(self.state)
    
class NormGame2048Env_ver0(Game2048Env):
    def __init__(self, render_mode=None, size=4) -> None:
        super().__init__(render_mode, size)
        self.max_pow_of_two = 20
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, self.size, self.size), dtype=float)

    def encode_state(self, input_board):
        tmp_state = input_board.copy()
        tmp_state[np.where(tmp_state==0.0)] = 1.0
        tmp_state = np.log2(tmp_state)
        return (tmp_state / self.max_pow_of_two)[np.newaxis]
    
class NormGame2048Env_ver1(Game2048Env):
    def __init__(self, render_mode=None, size=4) -> None:
        super().__init__(render_mode, size)
        self.max_pow_of_two = 20
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_pow_of_two, self.size, self.size), dtype=float)
        self.state_mask = np.ones(self.observation_space.shape).astype(np.int32)
        self.state_mask[0] *= 0
        for i in range(1, self.max_pow_of_two):
            self.state_mask[i] *= 2**i

    def encode_state(self, input_board):
        return (input_board == self.state_mask).astype(float)

if __name__ == '__main__':
    env = Game2048Env(render_mode='human')
    obs, _ = env.reset()

    for i in count():
        print('-'*50)
        act = env.action_space.sample()
        # act = int(input("Action:"))
        obs, r, done, _, _ = env.step(act)
        env.render()
        print(env.action_meaning[act])
        print(obs)
        if done:
            break
        
        






