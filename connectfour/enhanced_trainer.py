import time

import numpy as np
import torch
from pettingzoo.classic import connect_four_v3
from train_random import DQNTrainer


class EnhancedDQNTrainer(DQNTrainer):
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

            # Rest of the training loop remains the same...
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
