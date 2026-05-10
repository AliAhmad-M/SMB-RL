import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from models.mario_nn import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, device):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.save_dir   = save_dir
        self.device     = device

        # Initialize neural network
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # Exploration
        self.exploration_rate       = 1.0
        self.exploration_rate_decay = 0.9993
        self.exploration_rate_min   = 0.1
        self.curr_episode           = 0

        # Steps
        self.curr_step  = 0
        self.save_every = 5e5
        self.burnin      = 1e4
        self.learn_every = 3
        self.sync_every  = 1e4

        # Learning rate
        self.lr0 = 2.5e-4
        self.lrf = 1.0e-5
        self.discount = 0.95

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr0)
        self.loss_fn   = torch.nn.SmoothL1Loss()

        # ExponentialLR per episode decay
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.9991,
            last_epoch=-1
        )
        
        # Initialize memory parameters
        self.batch_size = 64
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(75000, device=torch.device("cpu")),
        )

    # Update parameters at episode end
    def on_episode_end(self):
        self.curr_episode += 1

        # Epsilon decay
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # LR decay
        if self.curr_episode > 1:
            self.scheduler.step()
            for pg in self.optimizer.param_groups:
                if pg["lr"] < self.lrf:
                    pg["lr"] = self.lrf

    # Agent act function
    def act(self, state):
        # Exploration
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # Exploitation
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.curr_step += 1
        return action_idx

    # Store memories in replay buffers
    def cache(self, state, next_state, action, reward, done):
        # Extract data from observation tuples
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state      = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        # Convert data into pytorch tensors
        state      = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action     = torch.tensor([action])

        # Apply reward scaling
        if reward > 10:
            reward = reward / 100.0
        else:
            reward = reward / 10.0

        reward = torch.tensor([reward])
        done   = torch.tensor([done])

        # Store transition data
        self.memory.add(TensorDict(
            {"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done},
            batch_size=[]
        ))

    # Sample a random batch of experiences from memory
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state.float(), next_state.float(), action.squeeze(), reward.squeeze(), done.squeeze()

    # Estimate Q values for the specific ation taken
    def td_estimate(self, state, action):
        return self.net(state, model="online")[np.arange(0, self.batch_size), action]

    # Implement the Double DQN target calculation
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action  = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.discount * next_Q).float()

    # Calculate loss and update online network
    def update_Q_online(self, td_estimate, td_target):
        loss = torch.nn.functional.huber_loss(td_estimate, td_target, delta=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # Copy weights from online network to target network
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # Save model weights and state checkpoint to the disk
    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    # Agent training function
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            # Copy weights to target network
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            # Export the current model state
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Fetch a sample of past experiences from memory
        state, next_state, action, reward, done = self.recall()

        # Estimate and calculate Q-values and update network
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss   = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    # Retrieve the current learning rate
    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]