import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from collections import deque

class MetricLogger:
    def __init__(self, save_dir, checkpoint_every=100, window=100):
        self.save_dir = save_dir
        self.checkpoint_every = checkpoint_every  # Log every N episodes
        self.window = window
        self.save_log = save_dir / "log"

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Checkpoint':>10}{'Episode':>10}{'Step':>12}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'BestReward':>15}{'TimeDelta':>15}{'Time':>20}\n"
            )

        # Per-episode history
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_epsilon = []
        self.ep_timestamps = []
        self.ep_checkpoint_num = []  # Which checkpoint each episode belongs to

        # Checkpoint summaries
        self.checkpoint_rewards = []
        self.checkpoint_lengths = []
        self.checkpoint_losses = []
        self.checkpoint_qs = []
        self.checkpoint_epsilons = []
        self.checkpoint_timestamps = []
        self.checkpoint_episodes = []  # Episode number at checkpoint
        self.checkpoint_steps = []     # Step number at checkpoint

        # Moving averages (by checkpoint)
        self.moving_avg_rewards = []
        self.moving_avg_lengths = []
        self.moving_avg_losses = []
        self.moving_avg_qs = []

        # Best reward tracking
        self.best_reward = -np.inf
        self.best_reward_episode = 0
        self.best_reward_checkpoint = 0

        # Reward distribution tracking
        self.recent_rewards = deque(maxlen=window)

        # Checkpoint counter
        self.current_checkpoint = 0
        self.episodes_since_checkpoint = 0
        
        # Accumulators for current checkpoint
        self.checkpoint_reward_sum = 0
        self.checkpoint_length_sum = 0
        self.checkpoint_loss_sum = 0
        self.checkpoint_q_sum = 0
        self.checkpoint_epsilon_sum = 0

        # Current episode state
        self.current_epsilon = 1.0
        
        self.init_episode()
        self.record_time = time.time()
        self.start_time = time.time()

    # Per-step logging
    def log_step(self, reward, loss, q, grad_norm=None):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
        if grad_norm is not None:
            self.curr_ep_grad_norm += grad_norm
            self.curr_ep_grad_length += 1
    
    # Method to update epsilon
    def update_epsilon(self, epsilon):
        self.current_epsilon = epsilon

    # Per-episode logging
    def log_episode(self):
        # Record episode metrics
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.recent_rewards.append(self.curr_ep_reward)
        self.ep_timestamps.append(time.time() - self.start_time)
        
        # Record epsilon for this episode
        self.ep_epsilon.append(self.current_epsilon)
        
        # Accumulate for checkpoint
        self.checkpoint_reward_sum += self.curr_ep_reward
        self.checkpoint_length_sum += self.curr_ep_length
        self.checkpoint_epsilon_sum += self.current_epsilon

        if self.curr_ep_reward > self.best_reward:
            self.best_reward = self.curr_ep_reward
            self.best_reward_episode = len(self.ep_rewards) - 1
            self.best_reward_checkpoint = self.current_checkpoint

        # Calculate episode averages
        ep_avg_loss = (
            np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            if self.curr_ep_loss_length > 0 else 0
        )
        ep_avg_q = (
            np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            if self.curr_ep_loss_length > 0 else 0
        )
        ep_avg_grad = (
            np.round(self.curr_ep_grad_norm / self.curr_ep_grad_length, 5)
            if self.curr_ep_grad_length > 0 else 0
        )

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        
        # Accumulate checkpoint averages
        self.checkpoint_loss_sum += ep_avg_loss
        self.checkpoint_q_sum += ep_avg_q

        self.episodes_since_checkpoint += 1
        
        # Check if we should log checkpoint
        if self.episodes_since_checkpoint >= self.checkpoint_every:
            self._log_checkpoint()
        
        self.init_episode()

    def _log_checkpoint(self):
        self.current_checkpoint += 1
        
        # Calculate checkpoint averages
        num_eps = self.episodes_since_checkpoint
        avg_reward = self.checkpoint_reward_sum / num_eps
        avg_length = self.checkpoint_length_sum / num_eps
        avg_loss = self.checkpoint_loss_sum / num_eps
        avg_q = self.checkpoint_q_sum / num_eps
        avg_epsilon = self.checkpoint_epsilon_sum / num_eps
        
        # Store checkpoint data
        self.checkpoint_rewards.append(avg_reward)
        self.checkpoint_lengths.append(avg_length)
        self.checkpoint_losses.append(avg_loss)
        self.checkpoint_qs.append(avg_q)
        self.checkpoint_epsilons.append(avg_epsilon)
        
        # Store episode number (last episode in checkpoint)
        last_episode = len(self.ep_rewards) - 1
        self.checkpoint_episodes.append(last_episode)
        
        # Calculate moving averages (over last N checkpoints)
        if len(self.checkpoint_rewards) >= self.window:
            self.moving_avg_rewards.append(np.mean(self.checkpoint_rewards[-self.window:]))
            self.moving_avg_lengths.append(np.mean(self.checkpoint_lengths[-self.window:]))
            self.moving_avg_losses.append(np.mean(self.checkpoint_losses[-self.window:]))
            self.moving_avg_qs.append(np.mean(self.checkpoint_qs[-self.window:]))
        else:
            self.moving_avg_rewards.append(avg_reward)
            self.moving_avg_lengths.append(avg_length)
            self.moving_avg_losses.append(avg_loss)
            self.moving_avg_qs.append(avg_q)
        
        # Reset checkpoint accumulators
        self.checkpoint_reward_sum = 0
        self.checkpoint_length_sum = 0
        self.checkpoint_loss_sum = 0
        self.checkpoint_q_sum = 0
        self.checkpoint_epsilon_sum = 0
        self.episodes_since_checkpoint = 0

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_grad_norm = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_grad_length = 0

    # Record checkpoint
    def record_checkpoint(self, episode, epsilon, step):
        
        # If we have pending episodes, log checkpoint first
        if self.episodes_since_checkpoint > 0:
            self._log_checkpoint()
        
        # Get latest checkpoint data
        if len(self.checkpoint_rewards) > 0:
            mean_reward = self.checkpoint_rewards[-1]
            mean_length = self.checkpoint_lengths[-1]
            mean_loss = self.checkpoint_losses[-1]
            mean_q = self.checkpoint_qs[-1]
        else:
            mean_reward = 0
            mean_length = 0
            mean_loss = 0
            mean_q = 0
        
        self.checkpoint_steps.append(step)
        
        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)
        now_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        # Get moving average over last N checkpoints
        moving_avg_reward = self.moving_avg_rewards[-1] if self.moving_avg_rewards else mean_reward
        
        print(
            f"Checkpoint {self.current_checkpoint} - Episode {episode} - Step {step} - Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_reward:.3f} (MA: {moving_avg_reward:.3f}) - "
            f"Mean Length {mean_length:.1f} - Mean Loss {mean_loss:.5f} - Mean Q {mean_q:.3f} - "
            f"Best Reward {self.best_reward:.3f} (ep {self.best_reward_episode}, ckpt {self.best_reward_checkpoint}) - "
            f"Time Delta {time_delta}s - {now_str}"
        )
        
        with open(self.save_log, "a") as f:
            f.write(
                f"{self.current_checkpoint:10d}{episode:10d}{step:12d}{epsilon:10.3f}"
                f"{mean_reward:15.3f}{mean_length:15.1f}{mean_loss:15.5f}{mean_q:15.3f}"
                f"{self.best_reward:15.3f}{time_delta:15.3f}{now_str:>20}\n"
            )
        
        self._plot_dashboard()
        self._save_json_snapshot(episode, epsilon, step)

    # Plotting
    def _plot_dashboard(self):
        if len(self.checkpoint_rewards) < 1:
            return
            
        n = len(self.checkpoint_rewards)
        checkpoints = np.arange(1, n + 1)

        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#1a1a2e")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        # Reward panel (spans full width of top row)
        ax1 = fig.add_subplot(gs[0, :])
        self._style_ax(ax1)
        ax1.plot(checkpoints, self.checkpoint_rewards, 'o-', alpha=0.5, 
                linewidth=1.2, markersize=4, color="#4cc9f0", label="Checkpoint Avg")
        if len(self.moving_avg_rewards) > 0:
            ax1.plot(checkpoints[:len(self.moving_avg_rewards)], self.moving_avg_rewards, 
                    linewidth=2.5, color="#f72585", label=f"MA({self.window})")
        ax1.set_title(f"Reward (Checkpoint every {self.checkpoint_every} episodes)", 
                     color="white", fontsize=12, pad=6)
        ax1.legend(fontsize=9, loc="upper left", facecolor="#2a2a4a", 
                  labelcolor="white", framealpha=0.6)
        
        # Loss panel
        ax2 = fig.add_subplot(gs[1, 0])
        self._style_ax(ax2)
        ax2.plot(checkpoints, self.checkpoint_losses, 'o-', alpha=0.5, 
                linewidth=1.2, markersize=4, color="#ff9e00", label="Loss")
        if len(self.moving_avg_losses) > 0:
            ax2.plot(checkpoints[:len(self.moving_avg_losses)], self.moving_avg_losses, 
                    linewidth=2, color="#06d6a0", label=f"MA({self.window})")
        ax2.set_title("Loss", color="white", fontsize=11, pad=6)
        ax2.legend(fontsize=8, loc="upper right", facecolor="#2a2a4a", 
                  labelcolor="white", framealpha=0.6)
        
        # Q Value panel
        ax3 = fig.add_subplot(gs[1, 1])
        self._style_ax(ax3)
        ax3.plot(checkpoints, self.checkpoint_qs, 'o-', alpha=0.5, 
                linewidth=1.2, markersize=4, color="#7209b7", label="Q Value")
        if len(self.moving_avg_qs) > 0:
            ax3.plot(checkpoints[:len(self.moving_avg_qs)], self.moving_avg_qs, 
                    linewidth=2, color="#06d6a0", label=f"MA({self.window})")
        ax3.set_title("Q Value", color="white", fontsize=11, pad=6)
        ax3.legend(fontsize=8, loc="upper left", facecolor="#2a2a4a", 
                  labelcolor="white", framealpha=0.6)
        
        # Epsilon decay panel
        ax4 = fig.add_subplot(gs[1, 2])
        self._style_ax(ax4)
        
        # Plot epsilon
        if len(self.checkpoint_epsilons) > 0:
            ax4.plot(checkpoints, self.checkpoint_epsilons, 
                    'o-', linewidth=2, markersize=4, color="#4cc9f0", label="Epsilon")
        ax4.set_xlabel("Checkpoint", color="gray", fontsize=9)
        ax4.set_ylabel("Epsilon", color="#4cc9f0", fontsize=9)
        ax4.tick_params(axis='y', labelcolor="#4cc9f0")
        ax4.set_title("Epsilon Decay", color="white", fontsize=11, pad=6)
        ax4.legend(fontsize=8, loc="upper right", facecolor="#2a2a4a", 
                  labelcolor="white", framealpha=0.6)
        
        # Add training progress summary as text on the figure
        total_time = np.round(time.time() - self.start_time, 1)
        current_eps = self.checkpoint_epsilons[-1] if self.checkpoint_epsilons else 0
        
        summary_text = (
            f"Best Reward: {self.best_reward:.1f} @ ep {self.best_reward_episode} | "
            f"Checkpoints: {self.current_checkpoint} | "
            f"Total Episodes: {len(self.ep_rewards)} | "
            f"Epsilon: {current_eps:.4f} | "
            f"Elapsed: {total_time}s"
        )
        
        fig.suptitle(
            f"RL Training Dashboard\n{summary_text}",
            color="white", fontsize=12, fontweight="bold", y=0.98
        )

        fig.savefig(self.save_dir / "dashboard.jpg", dpi=120,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, linestyle="--", linewidth=0.4, color="#333355", alpha=0.6)

    # JSON snapshot
    def _save_json_snapshot(self, episode, epsilon, step):
        snapshot = {
            "checkpoint": self.current_checkpoint,
            "episode": episode,
            "step": step,
            "epsilon": epsilon,
            "best_reward": self.best_reward,
            "best_reward_episode": self.best_reward_episode,
            "best_reward_checkpoint": self.best_reward_checkpoint,
            "checkpoint_every": self.checkpoint_every,
            "recent_checkpoint_rewards": self.checkpoint_rewards[-50:],
            "recent_checkpoint_losses": self.checkpoint_losses[-50:],
            "recent_checkpoint_qs": self.checkpoint_qs[-50:],
            "recent_checkpoint_epsilons": self.checkpoint_epsilons[-50:],
            "moving_avg_rewards": self.moving_avg_rewards[-50:],
            "total_episodes": len(self.ep_rewards),
            "total_checkpoints": self.current_checkpoint
        }
        with open(self.save_dir / "snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)