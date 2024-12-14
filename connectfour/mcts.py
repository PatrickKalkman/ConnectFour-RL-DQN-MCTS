import math

import numpy as np
import torch
import torch.nn.functional as F


class Node:
    def __init__(self, prior=0, parent=None):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.parent = parent
        self.state = None
        self.player = None
        self.valid_moves = None

    def expanded(self):
        return bool(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.c1 = args.get("c1", 1.25)
        self.c2 = args.get("c2", 19652)
        self.num_simulations = args.get("num_simulations", 50)
        self.temperature = args.get("temperature", 1.0)
        self.device = next(model.parameters()).device
        self.root = None
        self.valid_moves_tensor = torch.zeros((6, 7), device=self.device)

    def _get_observation_tensor(self, observation):
        """Convert PettingZoo observation to tensor format."""
        if observation is None:
            return None

        board = torch.from_numpy(observation["observation"][:, :, 0]).to(self.device)
        self.valid_moves_tensor.zero_()
        self.valid_moves_tensor[
            0, [i for i, valid in enumerate(observation["action_mask"]) if valid]
        ] = 1

        return torch.stack(
            [
                (board == 1).float(),
                (board == -1).float(),
                self.valid_moves_tensor,
            ]
        )

    def run(self, observation, valid_moves, player):
        # Create new root node for the current position
        self.root = Node()
        self.root.state = self._get_observation_tensor(observation)
        self.root.player = player
        self.root.valid_moves = valid_moves

        if not valid_moves:
            return None

        self._expand(self.root)

        for _ in range(self.num_simulations):
            node = self.root
            search_path = [node]

            # Selection
            current_depth = 0
            while node.expanded() and current_depth < 15:  # Limit search depth
                action, node = self._select_child(node)
                search_path.append(node)
                current_depth += 1

            value = self._evaluate(node)
            self._backpropagate(search_path, value, player)

        # Action selection
        if self.temperature == 0:
            action = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            visits = np.array(
                [child.visit_count for action, child in self.root.children.items()]
            )
            actions = list(self.root.children.keys())
            probs = visits ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            action = np.random.choice(actions, p=probs)

        return action

    def _select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())

        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            if child.visit_count > 0:
                q_value = -child.value()
                p_score = (
                    (self.c1 + math.log((total_visits + self.c2 + 1) / self.c2))
                    * child.prior
                    * math.sqrt(total_visits)
                    / (child.visit_count + 1)
                )
                score = q_value + p_score
            else:
                score = float("inf")

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node):
        if node.state is None:
            return

        with torch.no_grad():
            state_tensor = node.state.unsqueeze(0)
            policy_logits = self.model(state_tensor)
            policy = F.softmax(policy_logits / self.temperature, dim=1).squeeze(0)

        for move in node.valid_moves:
            node.children[move] = Node(prior=policy[move].item(), parent=node)

    def _evaluate(self, node):
        if node.state is None:
            return 0.0

        with torch.no_grad():
            state_tensor = node.state.unsqueeze(0)
            return self.model(state_tensor).max().item()

    def _backpropagate(self, search_path, value, player):
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1
            value *= -1

    def get_policy(self):
        """Return the MCTS policy (visit counts) for all moves."""
        if not self.root:
            return {}

        total_visits = sum(child.visit_count for child in self.root.children.values())
        if total_visits == 0:
            return {}

        return {
            action: child.visit_count / total_visits
            for action, child in self.root.children.items()
        }
