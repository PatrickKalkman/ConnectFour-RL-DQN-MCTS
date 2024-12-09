import os
import time

import torch
from pettingzoo.classic import connect_four_v3

from connectfour.deep_q_network.dqn_agent import DQNAgent


class GameState:
    def __init__(self):
        self.move_history = []
        self.state_history = []
        self.show_analysis = True


def play_against_agent(model_path: str = None):
    if model_path is None:
        potential_paths = ["./models/dqn_agent_random_first_player_dualing.pth"]
        model_path = next(
            (path for path in potential_paths if os.path.exists(path)), None
        )
        if not model_path:
            print("Error: Model not found. Available files:", os.listdir("."))
            model_path = input("Enter model path: ")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    state_dim = (3, 6, 7)
    action_dim = 7
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.policy_net.eval()  # Set to evaluation mode
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    game_state = GameState()
    valid_moves_tensor = torch.zeros((6, 7), device=device)

    def preprocess_observation(obs):
        board = torch.from_numpy(obs["observation"][:, :, 0]).to(device)
        valid_moves_tensor.zero_()
        valid_moves_tensor[
            0, [i for i, valid in enumerate(obs["action_mask"]) if valid]
        ] = 1
        return torch.stack(
            [
                (board == 1).float(),
                (board == -1).float(),
                valid_moves_tensor,
            ]
        )

    def undo_move():
        if not game_state.move_history:
            print("No moves to undo")
            return False
        game_state.move_history.pop()
        game_state.state_history.pop()
        env.reset()
        for move in game_state.move_history:
            env.step(move)
        return True

    def handle_human_turn(observation, valid_actions):
        if game_state.show_analysis:
            print("\nAgent's move analysis:")
            state = preprocess_observation(observation)
            with torch.no_grad():
                q_values = agent.policy_net(state.unsqueeze(0))[0]
                for action in valid_actions:
                    print(f"Column {action}: {q_values[action].item():.3f}")

        while True:
            print(f"\nValid moves: {valid_actions}")
            action = input("Enter move (0-6), 'u' to undo, 't' to toggle analysis: ")

            if action == "u":
                if undo_move():
                    return None
                continue
            elif action == "t":
                game_state.show_analysis = not game_state.show_analysis
                continue

            try:
                action = int(action)
                if action in valid_actions:
                    return action
                print("Invalid move")
            except ValueError:
                print("Enter a number between 0-6")

    env = connect_four_v3.env(render_mode="human")
    env.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()

        if termination or truncation:
            if reward != 0:
                winner = "Agent" if agent_name == "player_0" else "Human"
                print(f"\n{winner} wins!")
            else:
                print("\nDraw!")
            break

        valid_moves = [i for i, valid in enumerate(observation["action_mask"]) if valid]

        if agent_name == "player_1":  # Human
            action = handle_human_turn(observation, valid_moves)
            if action is None:  # Undo was performed
                continue
        else:  # Agent
            print("\nAgent thinking...")
            time.sleep(0.5)
            state = preprocess_observation(observation)
            action = agent.select_action(state, valid_moves, deterministic=True)

            if game_state.show_analysis:
                with torch.no_grad():
                    q_values = agent.policy_net(state.unsqueeze(0))[0]
                    print(
                        f"Agent chose {action} (value: {q_values[action].item():.3f})"
                    )

        game_state.move_history.append(action)
        game_state.state_history.append(observation)
        env.step(action)

    env.close()

    if input("\nPlay again? (y/n): ").lower() == "y":
        play_against_agent(model_path)


if __name__ == "__main__":
    print("Connect Four - You are yellow (player 2)")
    play_against_agent()
