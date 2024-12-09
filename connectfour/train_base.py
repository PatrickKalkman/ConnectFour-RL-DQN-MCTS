import json
import os
import time
from collections import deque
from dataclasses import dataclass
import numpy as np

import torch
from pettingzoo.classic import connect_four_v3
from torch.optim.lr_scheduler import CyclicLR

from connectfour.dqn_agent import DQNAgent


@dataclass
class TrainingConfig:
    episodes: int = 300_000
    log_interval: int = 250
    save_interval: int = 10_000
    render_interval: int = 400
    render_delay: float = 0.1
    # DQN specific parameters
    batch_size: int = 64
    memory_capacity: int = 750_000
    learning_rate: float = 1e-4
    gamma: float = 0.98
    epsilon_start: float = 0.3
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.9999975
    opponent_update_freq: int = 10000
    temperature: float = 1.0  # Temperature parameter for action selection
    # Learning rate cycle parameters
    lr_cycle_length: int = 50000
    lr_max: float = 3e-4
    lr_min: float = 5e-5
    # Paths for loading and saving
    pretrained_model_path: str = "models/dqn_agent_self_play_mps"
    opponent_model_path: str = (
        "models/dqn_agent_random_first_player.pth"  # Separate path for opponent
    )
    model_path: str = "models/dqn_agent_self_play.pth"
    metrics_path: str = "metrics/dqn_training_metrics_self_play"


class DQNTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_metrics()
        self.setup_device()
        self.setup_environment()
        self.setup_agents()
        self.setup_scheduler()

    def setup_metrics(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.moves_history = []
        self.recent_results = deque(maxlen=1000)
        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "epsilon": [],
            "draws_ratio": [],
            "loss": [],
            "fps": [],
            "avg_moves": [],
            "max_streak": [],
            "win_rate_improvement": [],
            "learning_rate": [],
            "avg_q_value": [],
            "exploration_ratio": [],
            "mcts_visit_diversity": [],
            "memory_usage": [],
            "recent_draws": [],
            "recent_losses": [],
            "visit_entropy": [],
        }

    def setup_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def setup_environment(self):
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        temp_env = connect_four_v3.env()
        temp_env.reset()
        obs, _, _, _, _ = temp_env.last()
        self.state_dim = (3, 6, 7)
        self.action_dim = temp_env.action_space("player_1").n
        temp_env.close()

    def setup_agents(self):
        # Main agent setup
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            learning_rate=self.config.learning_rate,
            memory_capacity=self.config.memory_capacity,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=self.config.epsilon_decay,
            temperature=self.config.temperature,
        )

        # Load main agent checkpoint
        if os.path.exists(self.config.pretrained_model_path):
            checkpoint = torch.load(
                self.config.pretrained_model_path, map_location=self.device
            )
            self.agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])

        # Opponent agent setup with separate checkpoint
        self.opponent_agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            learning_rate=self.config.learning_rate,
            memory_capacity=self.config.memory_capacity,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay=0.99999,
            temperature=self.config.temperature
            * 1.5,  # Slightly higher temperature for opponent
        )

        self.opponent_agent.policy_net.load_state_dict(
            self.agent.policy_net.state_dict()
        )
        self.opponent_agent.target_net.load_state_dict(
            self.agent.target_net.state_dict()
        )

    def setup_scheduler(self):
        self.scheduler = CyclicLR(
            self.agent.optimizer,
            base_lr=self.config.lr_min,
            max_lr=self.config.lr_max,
            step_size_up=self.config.lr_cycle_length // 2,
            mode="triangular2",
        )

    def _preprocess_observation(self, obs):
        board = torch.from_numpy(obs["observation"][:, :, 0]).to(self.device)
        if not hasattr(self, "valid_moves_tensor"):
            self.valid_moves_tensor = torch.zeros((6, 7), device=self.device)
        self.valid_moves_tensor.zero_()
        self.valid_moves_tensor[
            0, [i for i, valid in enumerate(obs["action_mask"]) if valid]
        ] = 1
        return torch.stack(
            [
                (board == 1).float(),
                (board == -1).float(),
                self.valid_moves_tensor,
            ]
        )

    def _get_valid_moves(self, obs):
        return [i for i, valid in enumerate(obs["action_mask"]) if valid]

    def _update_stats(self, reward: float):
        if reward > 0:
            self.wins += 1
            self.recent_results.append("W")
        elif reward < 0:
            self.losses += 1
            self.recent_results.append("L")
        else:
            self.draws += 1
            self.recent_results.append("D")

    def train(self):
        training_interval = 4
        steps_since_train = 0
        episode_times = []

        for episode in range(self.config.episodes):
            move_count = 0
            episode_start = time.time()
            render_mode = (
                "human"
                if episode % self.config.render_interval == 0 and episode > 0
                else None
            )
            self.env = connect_four_v3.env(render_mode=render_mode)
            self.env.reset()

            episode_loss = 0
            training_steps = 0
            previous_state = None
            previous_action = None
            game_done = False

            # Update opponent network and save checkpoint
            if episode % self.config.opponent_update_freq == 0 and episode > 0:
                print(f"\nUpdating opponent network at episode {episode}")
                self.opponent_agent.policy_net.load_state_dict(
                    self.agent.policy_net.state_dict()
                )
                torch.save(
                    {
                        "policy_net_state_dict": self.opponent_agent.policy_net.state_dict(),
                        "target_net_state_dict": self.opponent_agent.target_net.state_dict(),
                    },
                    self.config.opponent_model_path,
                )

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, _ = self.env.last()

                if (
                    (termination or truncation)
                    and not game_done
                    and agent == "player_0"
                ):
                    self._update_stats(reward)
                    self.moves_history.append(move_count)
                    game_done = True

                if termination or truncation:
                    action = None
                    if previous_state is not None and agent == "player_0":
                        current_state = self._preprocess_observation(observation)
                        self.agent.memory.push(
                            previous_state, previous_action, reward, current_state, True
                        )
                        steps_since_train += 1
                        if steps_since_train >= training_interval:
                            loss = self.agent.train_step()
                            if loss is not None:
                                episode_loss += loss
                                training_steps += 1
                                self.scheduler.step()
                            steps_since_train = 0
                else:
                    current_state = self._preprocess_observation(observation)

                    if previous_state is not None and agent == "player_0":
                        self.agent.memory.push(
                            previous_state,
                            previous_action,
                            reward,
                            current_state,
                            False,
                        )
                        steps_since_train += 1
                        if steps_since_train >= training_interval:
                            loss = self.agent.train_step()
                            if loss is not None:
                                episode_loss += loss
                                training_steps += 1
                                self.scheduler.step()
                            steps_since_train = 0

                    if agent == "player_0":
                        valid_moves = self._get_valid_moves(observation)
                        action = self.agent.select_action(current_state, valid_moves)
                        previous_state = current_state
                        previous_action = action
                    else:
                        valid_moves = self._get_valid_moves(observation)
                        action = self.opponent_agent.select_action(
                            current_state, valid_moves
                        )
                    move_count += 1

                self.env.step(action)

            episode_end = time.time()
            episode_times.append(episode_end - episode_start)

            if episode % self.config.log_interval == 0 and episode > 0:
                recent_episodes = min(1000, episode)
                recent_times = episode_times[-recent_episodes:]
                avg_time = sum(recent_times) / len(recent_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0

                avg_loss = episode_loss / training_steps if training_steps > 0 else 0
                self._log_progress(episode, avg_loss, fps)

            if episode % self.config.save_interval == 0:
                self.agent.save(self.config.model_path)
                self.save_metrics(episode)

            self.env.close()

        self.save_metrics("last")

    def _log_progress(self, episode: int, loss: float, fps: float):
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games if total_games > 0 else 0
        win_percentage = win_rate * 100

        # Convert deque to list for slicing
        recent_results_list = list(self.recent_results)
        recent_wins = recent_results_list.count("W")
        recent_losses = recent_results_list.count("L")
        recent_draws = recent_results_list.count("D")
        recent_total = len(recent_results_list)
        recent_percentage = (recent_wins / recent_total * 100) if recent_total > 0 else 0
        draws_ratio = (self.draws / total_games * 100) if total_games > 0 else 0

        avg_moves = sum(self.moves_history[-1000:]) / len(self.moves_history[-1000:]) if self.moves_history else 0
        
        # Calculate average Q-value from policy net's predictions
        avg_q_value = 0
        try:
            if (hasattr(self, 'agent') and hasattr(self.agent, 'policy_net') and 
                hasattr(self, 'env') and self.env is not None):
                
                # Create a dummy state for Q-value calculation
                dummy_board = np.zeros((6, 7, 1), dtype=np.float32)
                dummy_observation = {
                    "observation": dummy_board,
                    "action_mask": [1] * 7  # All moves valid for dummy state
                }
                current_state = self._preprocess_observation(dummy_observation)
                
                if current_state is not None:
                    with torch.no_grad():
                        q_values = self.agent.policy_net(current_state.unsqueeze(0))
                        avg_q_value = q_values.mean().item()
        except Exception as e:
            print(f"Warning: Q-value calculation failed: {e}")

        # Calculate MCTS statistics
        mcts_visit_diversity = 0
        visit_entropy = 0
        if hasattr(self, 'mcts') and hasattr(self.mcts, 'root') and self.mcts.root is not None:
            visits = [child.visit_count for child in self.mcts.root.children.values()]
            if visits:
                # Calculate visit diversity as normalized entropy
                total_visits = sum(visits)
                if total_visits > 0:
                    probs = [v/total_visits for v in visits]
                    # Calculate entropy
                    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
                    # Normalize by log(n) where n is number of moves
                    max_entropy = np.log(len(visits))
                    visit_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    mcts_visit_diversity = len(set(visits))

        # Calculate exploration ratio (based on epsilon value instead of move history)
        exploration_ratio = self.agent.epsilon if hasattr(self, 'agent') else 0

        memory_usage = (len(self.agent.memory) / self.config.memory_capacity * 100 
                    if hasattr(self, 'agent') else 0)

        current_streak = 0
        max_streak = 0
        for result in reversed(recent_results_list):
            if result == "W":
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                break

        win_rate_improvement = recent_percentage - getattr(self, "previous_win_rate", recent_percentage)
        self.previous_win_rate = recent_percentage

        current_lr = self.scheduler.get_last_lr()[0]

        # Update metrics history
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["overall_win_rate"].append(win_percentage)
        self.metrics_history["recent_win_rate"].append(recent_percentage)
        self.metrics_history["epsilon"].append(self.agent.epsilon)
        self.metrics_history["draws_ratio"].append(draws_ratio)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["fps"].append(fps)
        self.metrics_history["avg_moves"].append(avg_moves)
        self.metrics_history["max_streak"].append(max_streak)
        self.metrics_history["win_rate_improvement"].append(win_rate_improvement)
        self.metrics_history["learning_rate"].append(current_lr)
        self.metrics_history["avg_q_value"].append(avg_q_value)
        self.metrics_history["exploration_ratio"].append(exploration_ratio)
        self.metrics_history["mcts_visit_diversity"].append(mcts_visit_diversity)
        self.metrics_history["visit_entropy"].append(visit_entropy)
        self.metrics_history["memory_usage"].append(memory_usage)
        self.metrics_history["recent_draws"].append(recent_draws / recent_total * 100 if recent_total > 0 else 0)
        self.metrics_history["recent_losses"].append(recent_losses / recent_total * 100 if recent_total > 0 else 0)

        # Print enhanced progress
        print(f"\nEpisode: {episode} | FPS: {fps:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Overall Win Rate: {win_percentage:.1f}% | Recent: {recent_percentage:.1f}%")
        print(f"Recent W/D/L: {recent_wins/recent_total*100:.1f}%/{recent_draws/recent_total*100:.1f}%/{recent_losses/recent_total*100:.1f}%")
        print(f"Win Rate Change: {win_rate_improvement:+.1f}% | Current Streak: {current_streak}")
        print(f"Avg Moves: {avg_moves:.1f} | Loss: {loss:.6f}")
        print(f"Avg Q-Value: {avg_q_value:.3f} | Visit Entropy: {visit_entropy:.3f}")
        print(f"MCTS Diversity: {mcts_visit_diversity} | Memory Usage: {memory_usage:.1f}%")
        print(f"Main ε: {self.agent.epsilon:.3f} | Opp ε: {self.opponent_agent.epsilon:.3f}")
        print(f"Temp: {self.config.temperature:.2f} | Opp Temp: {self.config.temperature * 1.5:.2f}")
        print("-" * 70)

    def save_metrics(self, episode):
        with open(f"{self.config.metrics_path}_{episode}.json", "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
