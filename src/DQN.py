from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):

    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*input_dims, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc = nn.Linear(*input_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        actions = self.fc2(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilson, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilson
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, input_dims, n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def save_model(self, path):
        """Persist the current network and optimiser state to ``path``."""
        checkpoint_path = Path(path)
        if checkpoint_path.parent:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.Q_eval.state_dict(),
            "optimizer_state_dict": self.Q_eval.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, checkpoint_path)

    def load_model(self, path, load_optimizer=True):
        """Load network (and optionally optimiser) state from ``path``."""
        checkpoint = torch.load(path, map_location=self.Q_eval.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.Q_eval.load_state_dict(checkpoint["model_state_dict"])
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.Q_eval.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.Q_eval.load_state_dict(checkpoint)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            print("random action")
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]
        Q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, Q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
