import torch
import datetime
from pathlib import Path

from environments.smb_env import make_smb_env
from agents.mario import Mario
from metrics.logger import MetricLogger

# Initialize environment
env = make_smb_env()

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using CUDA: {use_cuda}\n")

save_dir = Path("output") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# Initialize agent
mario = Mario(
    state_dim=(4, 84, 84), 
    action_dim=env.action_space.n, 
    save_dir=save_dir, 
    device=device
)

logger = MetricLogger(save_dir)

episodes = 1000
save_every = 20 # Save every 20 episodes

for e in range(episodes):
    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    # Save logs and model at checkpoint
    if e % save_every == 0 and e != 0:
        checkpoint_path = save_dir / f"mario_net_{e}.chkpt"
        torch.save({
            "model": mario.net.state_dict(),
            "exploration_rate": mario.exploration_rate,
            "step": mario.curr_step
        }, checkpoint_path)

# Save final metrics and model
final_checkpoint_path = save_dir / "mario_final.chkpt"
torch.save({
    "model": mario.net.state_dict(),
    "exploration_rate": mario.exploration_rate,
    "step": mario.curr_step
}, final_checkpoint_path)

mario.save()
print("Training Complete. Final model saved.")