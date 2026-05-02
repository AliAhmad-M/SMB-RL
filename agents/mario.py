import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from models.mario_nn import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, device):
        # Initialize dimensions
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.save_dir   = save_dir

        # Initialize GPU device
        self.device = device

        # Initialize neural net
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # Initialize learning parameters
        self.exploration_rate       = 1
        self.exploration_rate_decay = 0.99999333
        self.exploration_rate_min   = 0.1

        self.discount   = 0.95  # Gamma
        self.curr_step  = 0     # The current step
        self.save_every = 5e5   # Save after every n experiences

        self.burnin      = 1e4  # Minimum number of experiences before training
        self.learn_every = 3    # Number of experiences between updates to Q_online
        self.sync_every  = 1e4  # Number of experiences between Q_target & Q_online sync

        self.lr0 = 2.5e-4
        self.lrf = 1.0e-5
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr0)
        self.loss_fn   = torch.nn.SmoothL1Loss()

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=0.99997425
        )

        # Initialize memory parameters
        self.batch_size = 64
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(75000, device=torch.device("cpu")),
        )

    # Given a state, choose an action and update step
    def act(self, state):
        # Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # Exploit
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate  = max(self.exploration_rate_min, self.exploration_rate)

        # Increment step
        self.curr_step += 1
        return action_idx

    # Store experience to memory
    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state      = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state      = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action     = torch.tensor([action])

        # Reward scaling
        if reward > 10:
            reward = reward / 100.0
        else:
            reward = reward / 10.0

        reward     = torch.tensor([reward])
        done       = torch.tensor([done])

        self.memory.add(TensorDict(
            {"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done},
            batch_size=[]
        ))

    # Retrieve batch of experiences
    def recall(self):
        batch = self.memory.sample(self.batch_size)
        batch = batch.to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # Q_online = Q(s, a) -> Expected reward
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    # Q_target = R + gamma * max(Q(s', a'))  [Double DQN]
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action  = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.discount * next_Q).float()

    # Gradient descent with IS weights from PER
    def update_Q_online(self, td_estimate, td_target):
        loss = torch.nn.functional.huber_loss(td_estimate, td_target, delta=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Enforce minimum LR, scheduler can't go below lr_min
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], self.lrf)

        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < self.lrf:
                param_group["lr"] = self.lrf

        return loss.item()

    # Stabilizer function
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Learn less frequently early on to avoid overfitting sparse buffer
        effective_learn_every = max(self.learn_every, 10 - self.curr_step // 10000)
        if self.curr_step % effective_learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        # Get the current learning rate from the optimizer
        current_lr = self.optimizer.param_groups[0]['lr']

        return (td_est.mean().item(), loss)
    
    # Helper function to get current learning rate
    def get_current_lr(self):
        # Returns the LR of the first parameter group
        return self.optimizer.param_groups[0]['lr']