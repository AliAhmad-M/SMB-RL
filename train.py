import torch
import datetime
import argparse
from pathlib import Path
from environments.smb_env import make_smb_env
from agents.mario import Mario
from metrics.logger import MetricLogger

# Helper function to load a checkpoint
def _load_checkpoint(checkpoint_path, mario_agent):
    checkpoint = torch.load(checkpoint_path)
    mario_agent.net.load_state_dict(checkpoint["model"])
    mario_agent.exploration_rate = checkpoint["exploration_rate"]
    mario_agent.curr_step = checkpoint["step"]

    # Restore optimizer state
    if "optimizer" in checkpoint:
        mario_agent.optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("  [warn] No optimizer state found in checkpoint, LR will reset.")

    # Restore scheduler state
    if "scheduler" in checkpoint and hasattr(mario_agent, "scheduler"):
        mario_agent.scheduler.load_state_dict(checkpoint["scheduler"])
    elif "scheduler" in checkpoint:
        print("  [warn] Checkpoint has scheduler state but agent has no scheduler attribute.")

    start_episode = checkpoint.get("episode", 0) + 1
    print(f"Resumed from {checkpoint_path} at step {checkpoint['step']}, episode {start_episode}")
    return (
        start_episode,
        checkpoint.get("best_mean_reward", float("-inf")),
        checkpoint.get("episode_rewards", []),
        checkpoint.get("flags_captured", 0),
        checkpoint.get("total_steps", 0),
    )

# Helper function to save a checkpoint
def _save_checkpoint(tag, episode):
    path = save_dir / f"mario_{tag}.chkpt"

    payload = {
        "model":             mario.net.state_dict(),
        "optimizer":         mario.optimizer.state_dict(),
        "exploration_rate":  mario.exploration_rate,
        "step":              mario.curr_step,
        "episode":           episode,
        # Resume-critical training state
        "best_mean_reward":  best_mean_reward,
        "episode_rewards":   episode_rewards,
        "flags_captured":    flags_captured,
        "total_steps":       total_steps,
    }

    # Save scheduler state if the agent has one
    if hasattr(mario, "scheduler"):
        payload["scheduler"] = mario.scheduler.state_dict()

    torch.save(payload, path)
    return path

# CUDA Setup
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using CUDA: {use_cuda}\n")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="Path to .chkpt file to resume from")
parser.add_argument("--transfer", type=str, default=None, help="Path to .chkpt file to transfer weights from")
args = parser.parse_args()

# Output directory
save_dir = Path("output") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# Initialize env
env = make_smb_env()

# Initialize agent
mario = Mario(
    state_dim=(4, 12, 12),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    device=device,
)

# Hyperparameters
EPISODES        = 4000
SAVE_EVERY      = 20    # periodic checkpoint interval
LOG_EVERY       = 20    # how often record() is called
WARMUP_EPISODES = 80    # episodes before we start checking for "best" model

# State tracking 
best_mean_reward  = float("-inf")
episode_rewards   = []
flags_captured    = 0
total_steps       = 0

# Resume
if args.resume:
    (
        start_episode,
        best_mean_reward,
        episode_rewards,
        flags_captured,
        total_steps,
    ) = _load_checkpoint(args.resume, mario)

# Transfer
elif args.transfer:
    ckpt = torch.load(args.transfer, map_location=device)
    mario.net.load_state_dict(ckpt["model"])
    start_episode = 0

else:
    start_episode = 0

# Logger
logger = MetricLogger(
    save_dir,
    checkpoint_every=20,
    window=5,
)

# Training loop
for e in range(start_episode, EPISODES):
    state = env.reset()
    ep_reward   = 0.0
    ep_flag_get = False

    while True:
        # Act
        action = mario.act(state)
        logger.update_epsilon(mario.exploration_rate)

        # Step
        next_state, reward, done, trunc, info = env.step(action)
        ep_reward  += reward
        total_steps += 1

        flag_get = bool(info.get("flag_get", False))
        if flag_get:
            ep_flag_get = True

        # Remember
        mario.cache(state, next_state, action, reward, done or trunc)

        # Learn
        q, loss = mario.learn()

        # Log step
        grad_norm = None
        if loss is not None:
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in mario.net.parameters()
                if p.grad is not None
            ) ** 0.5

        logger.log_step(reward, loss, q, grad_norm=grad_norm)

        state = next_state

        # Finish
        if done or trunc or flag_get:
            break

    # End of episode
    flags_captured += int(ep_flag_get)
    episode_rewards.append(ep_reward)
    logger.log_episode()

    mario.on_episode_end()

    # Best-model tracking after warmup
    # Use absolute episode index so warmup threshold is consistent across resumes
    if e >= WARMUP_EPISODES:
        window      = min(100, len(episode_rewards))
        mean_reward = sum(episode_rewards[-window:]) / window
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            path = _save_checkpoint("best", e)
            print(
                f"\tNew best mean reward: {best_mean_reward:.3f} "
                f"(ep {e + 1}) -> saved to {path}"
            )

    # Periodic logging
    if e % LOG_EVERY == 0 or e == EPISODES - 1:
        flag_rate = flags_captured / (e + 1) * 100
        print(
            f"\tEpisode {e + 1} | "
            f"Flag capture rate: {flag_rate:.1f}%  ({flags_captured}/{e + 1})"
        )
        logger.record_checkpoint(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step,
        )
        print(f"\tCurrent agent learning rate: {mario.get_current_lr():.6f}")

    # Periodic checkpoint
    if e % SAVE_EVERY == 0 and e != start_episode:
        path = _save_checkpoint(f"ep{e + 1}", e)
        print(f"\tCheckpoint saved -> {path}")

# Final save
_save_checkpoint("final", EPISODES - 1)
mario.save()

print(
    f"\nTraining complete.\n"
    f"  Total steps   : {total_steps}\n"
    f"  Flags captured: {flags_captured}/{EPISODES} "
    f"({flags_captured / EPISODES * 100:.1f}%)\n"
    f"  Best mean reward (100-ep window): {best_mean_reward:.3f}\n"
    f"  Outputs saved to: {save_dir}"
)