import warnings
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from environments.wrappers import SkipFrame, TileObservation, GrayScaleObservation, CropObservation, ResizeObservation

def make_smb_env(render_mode="rgb_array"):
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    
    # Initialize SMB environment
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)

    # Limit the action-space
    #   0. Sprint right
    #   1. Sprint jump right
    env = JoypadSpace(env, [["right", "B"], ["right", "A", "B"]])

    # Apply wrappers to environment
    env = SkipFrame(env, skip=4)
    env = TileObservation(env)
    env = CropObservation(env, top=48, bottom=16, left=32, right=32)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=12)
    env = FrameStack(env, num_stack=4)
    
    return env