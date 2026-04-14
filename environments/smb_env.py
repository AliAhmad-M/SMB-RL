import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from environments.wrappers import SkipFrame, GrayScaleObservation, CropObservation, ResizeObservation

def make_smb_env():
    # Initialize SMB environment
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v2", render_mode='rgb_array', apply_api_compatibility=True)

    # Limit the action-space
    #   0. Sprint right
    #   1. Sprint jump right
    env = JoypadSpace(env, [["right", "B"], ["right", "A", "B"]])

    # Apply wrappers to environment
    env = SkipFrame(env, skip=4)
    env = CropObservation(env, top=64, bottom=16)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    return env