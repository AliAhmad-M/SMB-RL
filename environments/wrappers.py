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
        # Repeat action and sum reward
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info
    
# Convert image into grayscale
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

# Crop an image by slicing from each side
class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, left=0, right=0, top=0, bottom=0):
        super().__init__(env)
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        
        old_shape = self.observation_space.shape # (240, 256, 3)
        # Calculate H and W specifically from the first two indices
        new_h = old_shape[0] - top - bottom
        new_w = old_shape[1] - left - right
        
        # Preserve the channel dimension (index 2)
        self.observation_space = Box(
            low=0, high=255, shape=(new_h, new_w, old_shape[2]), dtype=np.uint8
        )

    def observation(self, observation):
        # Explicitly slice H and W for a (Height, Width, Channel) input
        h, w, c = observation.shape
        return observation[
            self.top : h - self.bottom, 
            self.left : w - self.right, 
            :
        ]

# Resize an image into a square shape
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation