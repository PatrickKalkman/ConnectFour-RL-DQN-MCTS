import time
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import connect_four_v3

from connectfour.mcts import MCTS
from connectfour.train_base import DQNTrainer


@dataclass
class MCTSConfig:
    num_simulations: int = 40
    c1: float = 1.5
    c2: float = 19652
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


@dataclass
class TrainingConfig:
    episodes: int = 300_000
    log_interval: int = 200
    save_interval: int = 10_000
    render_interval: int = 600
    render_delay: float = 0.1
    # DQN specific parameters
    batch_size: int = 128
    memory_capacity: int = 1_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.98
    epsilon_start: float = 0.3
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.9999975
    opponent_update_freq: int = 7500
    temperature: float = 1.0
    # Learning rate cycle parameters
    lr_cycle_length: int = 50000
    lr_max: float = 5e-4
    lr_min: float = 1e-4
    # Paths for loading and saving
    pretrained_model_path: str = "models/dqn_agent_random_first_player.pth"
    opponent_model_path: str = "models/dqn_agent_random_first_player.pth"
    model_path: str = "models/dqn_agent_self_play.pth"
    metrics_path: str = "metrics/dqn_training_metrics_self_play"
    # MCTS parameters
    mcts: MCTSConfig = MCTSConfig()


class DQNMCTSTrainer(DQNTrainer):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.mcts_args = {
            "c1": config.mcts.c1,
            "c2": config.mcts.c2,
            "num_simulations": config.mcts.num_simulations,
            "temperature": config.mcts.temperature,
        }
        # We'll initialize MCTS instances in the train method
        self.mcts = None
        self.opponent_mcts = None

    def _create_mcts_instances(self):
        """Create MCTS instances with the current environment"""
        temp_env = connect_four_v3.env()
        self.mcts = MCTS(temp_env, self.agent.policy_net, self.mcts_args)
        self.opponent_mcts = MCTS(
            temp_env, self.opponent_agent.policy_net, self.mcts_args
        )
        temp_env.close()

    def _add_exploration_noise(self, node):
        """Add Dirichlet noise to the prior probabilities in the root node."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.mcts.dirichlet_alpha] * len(actions))

        for action, n in zip(actions, noise):
            node.children[action].prior = (
                1 - self.config.mcts.dirichlet_epsilon
            ) * node.children[action].prior + self.config.mcts.dirichlet_epsilon * n

    def _update_mcts_environments(self):
        """Update MCTS instances with the current environment"""
        if self.env is not None:
            self.mcts.game = self.env
            self.opponent_mcts.game = self.env

    def train(self):
        print("Initializing MCTS instances...")
        self._create_mcts_instances()

        print("Starting training...")
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

            # Update MCTS instances with the new environment
            self._update_mcts_environments()

            episode_loss = 0
            training_steps = 0
            previous_observation = None
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
                    self.moves_history.append(move_count)
                    game_done = True

                if termination or truncation:
                    action = None
                    if previous_observation is not None and agent == "player_0":
                        current_state = self._preprocess_observation(observation)
                        self.agent.memory.push(
                            previous_observation,
                            previous_action,
                            reward,
                            current_state,
                            True,
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
                    valid_moves = self._get_valid_moves(observation)

                    if previous_observation is not None and agent == "player_0":
                        current_state = self._preprocess_observation(observation)
                        self.agent.memory.push(
                            previous_observation,
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
                        # Use raw observation for MCTS
                        action = self.mcts.run(observation, valid_moves, agent)
                        previous_observation = self._preprocess_observation(observation)
                        previous_action = action
                    else:
                        action = self.opponent_mcts.run(observation, valid_moves, agent)

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


def main():
    print("Initializing DQN-MCTS trainer...")
    config = TrainingConfig()
    trainer = DQNMCTSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
