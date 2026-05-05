import gym
import torch
from gym.spaces import Box
from torchvision import transforms as T
import numpy as np

# Return only every skip-th frame
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


# Semantic encoding values
EMPTY   = np.array([255, 255, 255], dtype=np.uint8)  # white background
SOLID   = np.array([139, 90,  43],  dtype=np.uint8)  # brown ground/bricks
PIPE    = np.array([34,  139, 34],  dtype=np.uint8)  # green pipes
ENEMY   = np.array([220, 50,  50],  dtype=np.uint8)  # red enemies
MARIO   = np.array([30,  100, 220], dtype=np.uint8)  # blue Mario


# RAM constants
_ENEMY_DRAWN            = 0x0F
_ENEMY_TYPE             = 0x16
_ENEMY_X_IN_LEVEL       = 0x6E
_ENEMY_X_ON_SCREEN      = 0x87
_ENEMY_Y_ON_SCREEN      = 0xCF

_MARIO_X_IN_LEVEL       = 0x6D
_MARIO_X_ON_SCREEN      = 0x86
_MARIO_X_SCREEN_OFFSET  = 0x3AD
_MARIO_Y_SCREEN_OFFSET  = 0x3B8
_MARIO_Y_ON_SCREEN      = 0xCE
_MARIO_VERT_SCREEN_POS  = 0xB5

_TILEMAP_BASE   = 0x500
_PAGE_STRIDE    = 208
_TILEMAP_ROWS   = 13
_TILEMAP_COLS   = 16

_PIPE_TILES  = {0x12, 0x13, 0x14, 0x15}
_EMPTY_TILES = {0x00}

_SCREEN_W     = 256
_SCREEN_H     = 240
_TILE_PX      = 16
_GRID_COLS    = 16
_GRID_ROWS    = 15
_SPRITE_H     = 16
_STATUS_BAR_H = 32


class TileObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        ram = self.env.unwrapped.env.ram
        canvas = np.full((_SCREEN_H, _SCREEN_W, 3), 255, dtype=np.uint8)
        self._draw_static_tiles(canvas, ram)
        self._draw_enemies(canvas, ram)
        self._draw_mario(canvas, ram)
        return canvas

    def _fill_tile(self, canvas, row, col, value):
        if 0 <= row < _GRID_ROWS and 0 <= col < _GRID_COLS:
            y0 = row * _TILE_PX
            x0 = col * _TILE_PX
            canvas[y0:y0 + _TILE_PX, x0:x0 + _TILE_PX] = value

    def _get_tile_id(self, ram, x_level, y_screen):
        page  = (x_level // 256) % 2
        sub_x = (x_level % 256) // 16
        sub_y = (y_screen - _STATUS_BAR_H) // 16
        if sub_y not in range(_TILEMAP_ROWS):
            return 0x00
        addr = _TILEMAP_BASE + page * _PAGE_STRIDE + sub_y * _TILEMAP_COLS + sub_x
        return int(ram[addr])

    def _draw_static_tiles(self, canvas, ram):
        mario_level_x  = int(ram[_MARIO_X_IN_LEVEL]) * 256 + int(ram[_MARIO_X_ON_SCREEN])
        mario_screen_x = int(ram[_MARIO_X_SCREEN_OFFSET])
        x_start        = mario_level_x - mario_screen_x

        for row in range(_GRID_ROWS):
            if row < 2:
                continue
            y_screen = row * _TILE_PX
            for col in range(_GRID_COLS):
                x_level = x_start + col * _TILE_PX
                tile_id = self._get_tile_id(ram, x_level, y_screen)
                if tile_id == 0x00:
                    continue
                elif tile_id in _PIPE_TILES:
                    self._fill_tile(canvas, row, col, PIPE)
                else:
                    self._fill_tile(canvas, row, col, SOLID)

    def _draw_enemies(self, canvas, ram):
        mario_level_x  = int(ram[_MARIO_X_IN_LEVEL]) * 256 + int(ram[_MARIO_X_ON_SCREEN])
        mario_screen_x = int(ram[_MARIO_X_SCREEN_OFFSET])
        x_start        = mario_level_x - mario_screen_x

        for slot in range(5):
            if ram[_ENEMY_DRAWN + slot] == 0:
                continue
            x_level  = int(ram[_ENEMY_X_IN_LEVEL + slot]) * 0x100 + int(ram[_ENEMY_X_ON_SCREEN + slot])
            y_screen = int(ram[_ENEMY_Y_ON_SCREEN + slot]) + 8
            screen_x = x_level - x_start
            col = int(np.clip(screen_x, 0, _SCREEN_W - 1)) // _TILE_PX
            row = int(np.clip(y_screen, 0, _SCREEN_H - 1)) // _TILE_PX
            self._fill_tile(canvas, row, col, ENEMY)

    def _draw_mario(self, canvas, ram):
        x = int(ram[_MARIO_X_SCREEN_OFFSET]) + 12
        y = int(ram[_MARIO_Y_SCREEN_OFFSET]) + _SPRITE_H
        col = x // _TILE_PX
        row = y // _TILE_PX
        self._fill_tile(canvas, row, col, MARIO)


# Crop an image by slicing from each side
# Input:  (H, W, 3) uint8
# Output: (H', W', 3) uint8
class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, left=0, right=0, top=0, bottom=0):
        super().__init__(env)
        self.left   = left
        self.right  = right
        self.top    = top
        self.bottom = bottom

        old_shape = self.observation_space.shape
        new_h = old_shape[0] - top - bottom
        new_w = old_shape[1] - left - right
        self.observation_space = Box(
            low=0, high=255, shape=(new_h, new_w, old_shape[2]), dtype=np.uint8
        )

    def observation(self, observation):
        h, w, c = observation.shape
        return observation[
            self.top : h - self.bottom,
            self.left : w - self.right,
            :
        ]


# Convert (H, W, 3) uint8 -> (H, W) uint8 grayscale
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

    def observation(self, observation):
        gray = np.zeros(observation.shape[:2], dtype=np.uint8)
        
        gray[np.all(observation == [255, 255, 255], axis=-1)] = 255  # EMPTY  -> white
        gray[np.all(observation == [139, 90,  43],  axis=-1)] = 128  # SOLID  -> gray
        gray[np.all(observation == [34,  139, 34],  axis=-1)] = 128  # PIPE   -> gray
        gray[np.all(observation == [220, 50,  50],  axis=-1)] = 64   # ENEMY  -> dark gray
        gray[np.all(observation == [30,  100, 220], axis=-1)] = 0    # MARIO  -> black
        
        return gray.astype(np.uint8)


# Resize (H, W) uint8 -> (shape, shape) float32 normalized to [0, 1]
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        self.observation_space = Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        obs = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        obs = T.Resize(self.shape, antialias=False)(obs)
        return obs.squeeze(0).numpy().astype(np.uint8)