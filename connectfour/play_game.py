import os
import time

import torch
from pettingzoo.classic import connect_four_v3

from connectfour.dqn_agent import DQNAgent
from connectfour.mcts import MCTS  # Import the MCTS class we created earlier


class GameState:
    def __init__(self):
        self.move_history = []
        self.state_history = []
        self.show_analysis = True


def play_against_agent(model_path: str = None):
    if model_path is None:
        potential_paths = ["./models/dqn_agent_self_play.pth"]
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

    # Initialize MCTS
    mcts_args = {
        "num_simulations": 2800,  # More simulations for stronger play
        "c1": 1.8,
        "c2": 19652,
        "temperature": 0.3,  # Lower temperature for more focused play
    }
    env = connect_four_v3.env(render_mode="human")
    mcts = MCTS(env, agent.policy_net, mcts_args)

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

    def analyze_position(observation, valid_actions):
        """Analyze position using both MCTS and DQN"""
        state = preprocess_observation(observation)
        print("\nPosition Analysis:")

        # DQN Analysis
        with torch.no_grad():
            q_values = agent.policy_net(state.unsqueeze(0))[0]
            print("\nDQN Values:")
            for action in valid_actions:
                print(f"Column {action}: {q_values[action].item():.3f}")

        # MCTS Analysis
        if game_state.show_analysis:
            print("\nRunning MCTS analysis...")
            action = mcts.run(observation, valid_actions, "player_0")
            visits = [
                (action, child.visit_count)
                for action, child in mcts.root.children.items()
            ]
            visits.sort(key=lambda x: x[1], reverse=True)

            print("\nMCTS Visit Counts:")
            for action, visits in visits:
                if action in valid_actions:
                    print(f"Column {action}: {visits} visits")

    def handle_human_turn(observation, valid_actions):
        if game_state.show_analysis:
            analyze_position(observation, valid_actions)

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

    env.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()

        if termination or truncation:
            if reward != 0:
                winner = "Human" if agent_name == "player_0" else "Agent"
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

            action = mcts.run(observation, valid_moves, agent_name)

            if game_state.show_analysis:
                state = preprocess_observation(observation)
                with torch.no_grad():
                    q_values = agent.policy_net(state.unsqueeze(0))[0]
                    print(f"\nAgent chose column {action}")
                    print(f"DQN value: {q_values[action].item():.3f}")

                    # Get MCTS statistics
                    policy = mcts.get_policy()
                    if policy:
                        print("\nMCTS Analysis:")
                        for move, prob in sorted(
                            policy.items(), key=lambda x: x[1], reverse=True
                        ):
                            if move in valid_moves:
                                visits = mcts.root.children[move].visit_count
                                value = mcts.root.children[move].value()
                                print(
                                    f"Column {move}: {prob:.3f} ({visits} visits, value: {value:.3f})"
                                )

        game_state.move_history.append(action)
        game_state.state_history.append(observation)
        env.step(action)

    env.close()

    if input("\nPlay again? (y/n): ").lower() == "y":
        play_against_agent(model_path)


if __name__ == "__main__":
    print("Connect Four - You are yellow (player 2)")
    print("The agent will use both DQN and MCTS for move selection")
    print("Analysis will show both DQN values and MCTS visit counts")
    play_against_agent()
