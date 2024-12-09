import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import mse_loss

from connectfour.connect_four_dqn_dense import Connect4DQN
from connectfour.replay_memory import ReplayMemory


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        learning_rate=1e-4,
        memory_capacity=10000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=1e-3,
        temperature=1.0,
    ):
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.temperature = temperature

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = Connect4DQN(state_dim[0]).to(device)
        self.target_net = Connect4DQN(state_dim[0]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.training_steps = 0
        self.memory = ReplayMemory(memory_capacity, device)
        self.loss_window = []
        self.loss_window_size = 100

    def select_action(self, state, valid_moves, deterministic=False):
        if deterministic:
            with torch.no_grad():
                state_tensor = (
                    state
                    if isinstance(state, torch.Tensor)
                    else torch.FloatTensor(state).to(self.device)
                )
                state_tensor = state_tensor.unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                mask = torch.ones(self.action_dim, device=self.device) * float("-inf")
                mask[valid_moves] = 0
                q_values = q_values + mask
                return q_values.max(1)[1].item()

        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = (
                    state
                    if isinstance(state, torch.Tensor)
                    else torch.FloatTensor(state).to(self.device)
                )
                state_tensor = state_tensor.unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                mask = torch.ones(self.action_dim, device=self.device) * float("-inf")
                mask[valid_moves] = 0
                q_values = q_values + mask

                if self.temperature != 1.0:
                    scaled_q_values = q_values / self.temperature
                    probs = F.softmax(scaled_q_values, dim=1)
                    return torch.multinomial(probs, 1).item()
                return q_values.max(1)[1].item()
        else:
            return np.random.choice(valid_moves)

    def train_step(self):
        if not self.memory.can_sample(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        if len(states.shape) != 4:
            raise ValueError(
                f"Expected states shape (batch_size, channels, height, width), got {states.shape}"
            )

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            if self.temperature != 1.0:
                next_q_values = next_q_values / self.temperature
            next_q_values = next_q_values.max(1)[0]
            next_q_values[dones] = 0.0

        target_q_values = rewards + (self.gamma * next_q_values)
        loss = mse_loss(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_value = loss.item()
        self.loss_window.append(loss_value)
        if len(self.loss_window) > self.loss_window_size:
            self.loss_window.pop(0)

        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.soft_update_target_network()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss_value

    def soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "loss_window": self.loss_window,
                "temperature": self.temperature,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.loss_window = checkpoint.get("loss_window", [])
        self.temperature = checkpoint.get("temperature", 1.0)
