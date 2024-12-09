import math

import numpy as np
import torch
import torch.nn.functional as F


class Node:
    __slots__ = [
        "visit_count",
        "prior",
        "value_sum",
        "children",
        "parent",
        "state",
        "player",
        "valid_moves",
    ]

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
        return self.value_sum / self.visit_count if self.visit_count else 0


class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.c1 = args.get("c1", 1.25)
        self.c2 = args.get("c2", 19652)
        self.num_simulations = args.get("num_simulations", 20)  # Reduced from 50
        self.temperature = args.get("temperature", 1.0)
        self.device = next(model.parameters()).device

        # Pre-allocate tensors
        self.valid_moves_tensor = torch.zeros((6, 7), device=self.device)

    @torch.no_grad()  # Disable gradient computation
    def _get_observation_tensor(self, observation):
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
        root = Node()
        root.state = self._get_observation_tensor(observation)
        root.player = player
        root.valid_moves = valid_moves

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_depth = 0

            # Selection
            while node.expanded() and current_depth < 15:  # Reduced max depth
                action, node = self._select_child(node)
                search_path.append(node)
                current_depth += 1

            value = self._evaluate(node)
            self._backpropagate(search_path, value, player)

        # Action selection
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())

        if self.temperature == 0:
            return actions[np.argmax(visit_counts)]

        visit_count_distribution = visit_counts ** (1 / self.temperature)
        visit_count_distribution /= visit_count_distribution.sum()
        return np.random.choice(actions, p=visit_count_distribution)

    def _select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())

        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            if child.visit_count:
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

    @torch.no_grad()  # Disable gradient computation
    def _expand(self, node):
        if node.state is None:
            return

        state_tensor = node.state.unsqueeze(0)
        policy_logits = self.model(state_tensor)
        policy = F.softmax(policy_logits / self.temperature, dim=1).squeeze(0)

        for move in node.valid_moves:
            node.children[move] = Node(prior=policy[move].item(), parent=node)

    @torch.no_grad()  # Disable gradient computation
    def _evaluate(self, node):
        if node.state is None:
            return 0.0

        state_tensor = node.state.unsqueeze(0)
        return self.model(state_tensor).max().item()

    def _backpropagate(self, search_path, value, player):
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1
            value *= -1
