import torch
import datetime
from pathlib import Path
from environments.smb_env import make_smb_env
from agents.mario import Mario
from metrics.logger import MetricLogger

# Setup
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using CUDA: {use_cuda}\n")

save_dir = Path("output") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

env = make_smb_env()

mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    device=device,
)
logger = MetricLogger(save_dir)

# Hyperparameters
EPISODES        = 4000
SAVE_EVERY      = 20    # periodic checkpoint interval
LOG_EVERY       = 20    # how often record() is called
WARMUP_EPISODES = 80    # episodes before we start checking for "best" model

# State tracking
best_mean_reward  = float("-inf")
episode_rewards   = []   # raw per-episode totals for computing the running mean
flags_captured    = 0
total_steps       = 0

def _save_checkpoint(tag: str):
    # Save a checkpoint of the current model
    path = save_dir / f"mario_{tag}.chkpt"
    torch.save(
        {
            "model": mario.net.state_dict(),
            "exploration_rate": mario.exploration_rate,
            "step": mario.curr_step,
        },
        path,
    )
    return path


# Training loop
for e in range(EPISODES):
    state = env.reset()
    ep_reward   = 0.0
    ep_flag_get = False

    while True:
        # Act
        action = mario.act(state)

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

    # Best-model tracking after warmup
    if e >= WARMUP_EPISODES:
        window       = min(100, len(episode_rewards))
        mean_reward  = sum(episode_rewards[-window:]) / window
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            path = _save_checkpoint("best")
            print(
                f"\tNew best mean reward: {best_mean_reward:.3f} "
                f"(ep {e+1}) -> saved to {path}"
            )

    # Periodic logging
    if e % LOG_EVERY == 0 or e == EPISODES - 1:
        flag_rate = flags_captured / (e + 1) * 100
        print(f"\tFlag capture rate: {flag_rate:.1f}%  ({flags_captured}/{e+1})")
        logger.record_checkpoint(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step,
        )

    # Periodic checkpoint
    if e % SAVE_EVERY == 0 and e != 0:
        path = _save_checkpoint(f"ep{e}")
        print(f"\tCheckpoint saved -> {path}")

# Final save
_save_checkpoint("final")
mario.save()

print(
    f"\nTraining complete.\n"
    f"  Total steps   : {total_steps}\n"
    f"  Flags captured: {flags_captured}/{EPISODES} "
    f"({flags_captured/EPISODES*100:.1f}%)\n"
    f"  Best mean reward (100-ep window): {best_mean_reward:.3f}\n"
    f"  Outputs saved to: {save_dir}"
)