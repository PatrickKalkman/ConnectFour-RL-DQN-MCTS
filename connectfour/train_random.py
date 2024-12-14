import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from dqn_agent import DQNAgent
from pettingzoo.classic import connect_four_v3


@dataclass
class TrainingConfig:
    episodes: int = 300_000
    log_interval: int = 250
    save_interval: int = 10_000
    render_interval: int = 600_000
    render_delay: float = 0.1
    model_path: str = "models/dqn_agent_random_first_player.pth"
    metrics_path: str = "metrics/dqn_training_metrics_random_first_player"
    # DQN specific parameters
    batch_size: int = 128
    memory_capacity: int = 750_000
    learning_rate: float = 5e-5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.999995


class DQNTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create directories
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        # Initialize metrics tracking
        self.recent_results = deque(maxlen=1000)
        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "epsilon": [],
            "draws_ratio": [],
            "loss": [],
            "fps": [],
        }

        # Initialize environment dimensions
        temp_env = connect_four_v3.env()
        temp_env.reset()
        obs, _, _, _, _ = temp_env.last()
        self.state_dim = (3, 6, 7)  # Channels, Height, Width
        self.action_dim = temp_env.action_space("player_1").n
        temp_env.close()

        # Initialize DQN agent
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            learning_rate=config.learning_rate,
            memory_capacity=config.memory_capacity,
            batch_size=config.batch_size,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
        )

    def _preprocess_observation(self, obs):
        """Faster preprocessing with less memory allocation"""
        board = torch.from_numpy(obs["observation"][:, :, 0]).to(self.device)
        if not hasattr(self, "valid_moves_tensor"):
            self.valid_moves_tensor = torch.zeros((6, 7), device=self.device)

        self.valid_moves_tensor.zero_()
        self.valid_moves_tensor[
            0, [i for i, valid in enumerate(obs["action_mask"]) if valid]
        ] = 1

        return torch.stack(
            [
                (board == 1).float(),  # Player pieces
                (board == -1).float(),  # Opponent pieces
                self.valid_moves_tensor,
            ]
        )

    def _get_valid_moves(self, obs):
        """Get list of valid moves from observation"""
        return [i for i, valid in enumerate(obs["action_mask"]) if valid]

    def _check_three_in_a_row(self, board, player):
        """Check if player has three in a row with an open fourth position"""
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]  # horizontal, vertical, diagonals
        rows, cols = board.shape

        for r in range(rows):
            for c in range(cols):
                if board[r, c] != player:
                    continue

                for dr, dc in directions:
                    count = 1
                    open_pos = None

                    # Check next 3 positions
                    for i in range(1, 4):
                        new_r, new_c = r + i * dr, c + i * dc
                        if not (0 <= new_r < rows and 0 <= new_c < cols):
                            break

                        if board[new_r, new_c] == player:
                            count += 1
                        elif board[new_r, new_c] == 0:  # Empty position
                            if open_pos is None:  # Only count first empty position
                                open_pos = (new_r, new_c)
                        else:
                            break

                    if count == 3 and open_pos is not None:
                        return True
        return False

    def _check_opponent_win_possible(self, board, action):
        """Check if the opponent can win in the next move"""
        # Create a copy of the board with our move
        test_board = board.copy()
        row = self._get_landing_row(test_board, action)
        if row is None:
            return False
        test_board[row, action] = 1  # Our move

        # Try each possible opponent move
        for col in range(test_board.shape[1]):
            row = self._get_landing_row(test_board, col)
            if row is not None:
                test_board[row, col] = -1  # Opponent move
                if self._check_win(test_board, -1):
                    return True
                test_board[row, col] = 0  # Undo move
        return False

    def _get_landing_row(self, board, col):
        """Get the row where a piece would land in the given column"""
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None

    def _check_win(self, board, player):
        """Check if player has won"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        rows, cols = board.shape

        for r in range(rows):
            for c in range(cols):
                if board[r, c] != player:
                    continue

                for dr, dc in directions:
                    count = 1
                    for i in range(1, 4):
                        new_r, new_c = r + i * dr, c + i * dc
                        if not (0 <= new_r < rows and 0 <= new_c < cols):
                            break
                        if board[new_r, new_c] != player:
                            break
                        count += 1
                    if count == 4:
                        return True
        return False

    def _calculate_auxiliary_reward(self, observation, action):
        """Calculate additional rewards based on game state"""
        board = observation["observation"][:, :, 0]
        reward = 0

        # Reward for creating three in a row with open fourth position
        if self._check_three_in_a_row(board, 1):  # 1 is our player
            reward += 0.2

        # Penalize allowing opponent to win next turn
        if self._check_opponent_win_possible(board, action):
            reward -= 0.5

        return reward

    def _horizontal_flip(self, state, action=None):
        """Flip state horizontally and adjust action if provided"""
        if isinstance(state, torch.Tensor):
            flipped_state = torch.flip(state, dims=[-1])
        else:
            flipped_state = np.flip(state, axis=-1)

        if action is not None:
            flipped_action = 6 - action  # For 7 columns (0-6)
            return flipped_state, flipped_action
        return flipped_state

    def _update_stats(self, reward: float):
        """Update win/loss/draw statistics"""
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
            episode_start = time.time()
            self.env = connect_four_v3.env(render_mode=None)
            self.env.reset()

            episode_loss = 0
            training_steps = 0
            previous_state = None
            previous_action = None
            game_done = False

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, _ = self.env.last()

                if (
                    (termination or truncation)
                    and not game_done
                    and agent == "player_0"
                ):
                    self._update_stats(reward)
                    game_done = True

                if agent == "player_0":  # Our agent
                    current_state = self._preprocess_observation(observation)

                    if termination or truncation:
                        action = None
                        if previous_state is not None:
                            # Store original experience
                            self.agent.memory.push(
                                previous_state,
                                previous_action,
                                reward,
                                current_state,
                                True,
                            )
                            # Store flipped experience
                            flipped_prev_state = self._horizontal_flip(previous_state)
                            flipped_curr_state = self._horizontal_flip(current_state)
                            flipped_action = 6 - previous_action
                            self.agent.memory.push(
                                flipped_prev_state,
                                flipped_action,
                                reward,
                                flipped_curr_state,
                                True,
                            )
                    else:
                        if previous_state is not None:
                            # Calculate auxiliary reward
                            aux_reward = self._calculate_auxiliary_reward(
                                observation, previous_action
                            )
                            total_reward = reward + aux_reward

                            # Store original experience
                            self.agent.memory.push(
                                previous_state,
                                previous_action,
                                total_reward,
                                current_state,
                                False,
                            )
                            # Store flipped experience
                            flipped_prev_state = self._horizontal_flip(previous_state)
                            flipped_curr_state = self._horizontal_flip(current_state)
                            flipped_action = 6 - previous_action
                            self.agent.memory.push(
                                flipped_prev_state,
                                flipped_action,
                                total_reward,
                                flipped_curr_state,
                                False,
                            )

                        valid_moves = self._get_valid_moves(observation)
                        action = self.agent.select_action(current_state, valid_moves)
                        previous_state = current_state
                        previous_action = action

                        steps_since_train += 1
                        if steps_since_train >= training_interval:
                            loss = self.agent.train_step()
                            if loss is not None:
                                episode_loss += loss
                                training_steps += 1
                            steps_since_train = 0
                else:  # Random opponent
                    if not (termination or truncation):
                        valid_moves = self._get_valid_moves(observation)
                        action = np.random.choice(valid_moves)
                    else:
                        action = None

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
        # Calculate metrics
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games if total_games > 0 else 0
        win_percentage = win_rate * 100

        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_percentage = (
            (recent_wins / recent_total * 100) if recent_total > 0 else 0
        )
        draws_ratio = (self.draws / total_games * 100) if total_games > 0 else 0

        # Store metrics
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["overall_win_rate"].append(win_percentage)
        self.metrics_history["recent_win_rate"].append(recent_percentage)
        self.metrics_history["epsilon"].append(self.agent.epsilon)
        self.metrics_history["draws_ratio"].append(draws_ratio)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["fps"].append(fps)

        # Print metrics
        print(f"FPS: {fps:.2f}")
        print(f"Episode: {episode}")
        print(f"Overall Win Rate: {win_rate:.2f} ({win_percentage:.1f}%)")
        print(f"Last {recent_total} games: {recent_percentage:.1f}%")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
        print(f"Epsilon: {self.agent.epsilon:.3f}")

        if loss > 0:
            if len(self.metrics_history["loss"]) > 1:
                last_100_losses = self.metrics_history["loss"][-100:]
                print(f"Current Loss: {loss:.6f}")
                print(
                    f"Avg Loss (last 100): {sum(last_100_losses) / len(last_100_losses):.6f}"
                )
                print(
                    f"Min/Max Loss (last 100): {min(last_100_losses):.6f}/{max(last_100_losses):.6f}"
                )
            else:
                print(f"Initial Loss: {loss:.6f}")

        print("-" * 50)

    def save_metrics(self, episode):
        with open(f"{self.config.metrics_path}_{episode}.json", "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
