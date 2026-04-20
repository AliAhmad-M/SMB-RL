import torch
from pathlib import Path
from environments.smb_env import make_smb_env
from agents.mario import Mario

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
env = make_smb_env(render_mode="human")

# Re-initialize agent
save_dir = Path("output") / "2026-04-20T14-12-52"
checkpoint_path = save_dir / "mario_final.chkpt"

mario = Mario(
    state_dim=(4, 84, 84), 
    action_dim=env.action_space.n, 
    save_dir=save_dir, 
    device=device
)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
mario.net.load_state_dict(checkpoint["model"])

# Evaluation Loop
mario.exploration_rate = 0 

state = env.reset()
done = False

while not done:
    # Use the agent's act method
    action = mario.act(state)
    
    # Take the action
    next_state, reward, done, trunc, info = env.step(action)
    
    # Visualize the environment
    env.render()
    
    state = next_state
    
    if info["flag_get"]:
        print("Mario reached the flag!")
        break

env.close()