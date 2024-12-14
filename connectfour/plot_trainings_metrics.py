import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MetricsVisualizer:
    def __init__(self, metrics_path):
        self.metrics_path = Path(metrics_path)
        print(f"Loading metrics from: {self.metrics_path}")
        with open(self.metrics_path, "r") as f:
            self.metrics = json.load(f)

        print("\nAvailable metrics keys:", self.metrics.keys())

    def debug_print_metrics(self):
        """Print detailed information about the metrics"""
        print("\nDetailed Metrics Information:")
        for key, value in self.metrics.items():
            if isinstance(value, list):
                print(f"{key}:")
                print(f"  Length: {len(value)}")
                print(f"  First few values: {value[:5]}")
                print(f"  Last few values: {value[-5:]}")
                print("-" * 50)

    def calculate_rolling_average(self, data, window=100):
        """Calculate rolling average with the specified window size"""
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def plot_metrics(self, save_dir="metrics/training_plots"):
        """Plot metrics in separate subplots using seaborn style"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Set the style
        sns.set_style("whitegrid")
        plt.rcParams["figure.facecolor"] = "white"

        # Create figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(15, 20))

        # Get episode numbers for x-axis
        episodes = np.array(self.metrics["episodes"])

        # 1. Win Rates Plot
        axs[0].plot(
            episodes,
            self.metrics["overall_win_rate"],
            label="overall_win_rate",
            color="#1f77b4",
            alpha=0.7,
        )
        axs[0].plot(
            episodes,
            self.metrics["recent_win_rate"],
            label="recent_win_rate",
            color="#ff7f0e",
            alpha=0.7,
        )
        axs[0].set_ylabel("Win Rate (%)")

        # 2. Epsilon Plot
        axs[1].plot(
            episodes,
            self.metrics["epsilon"],
            label="epsilon",
            color="#2ca02c",
            alpha=0.7,
        )
        axs[1].set_ylabel("Epsilon Value")

        # 3. Loss Plot with Rolling Average
        window_size = 100
        raw_loss = np.array(self.metrics["loss"])
        smoothed_loss = self.calculate_rolling_average(raw_loss, window=window_size)

        # Plot raw loss with low opacity
        axs[2].plot(episodes, raw_loss, label="raw loss", color="#d62728", alpha=0.2)

        # Plot smoothed loss
        axs[2].plot(
            episodes[window_size - 1 :],
            smoothed_loss,
            label=f"smoothed loss ({window_size}-episode avg)",
            color="#2c3e50",
            linewidth=2,
            alpha=1.0,
        )
        axs[2].set_ylabel("Loss Value")

        # 4. Draws Ratio Plot
        axs[3].plot(
            episodes,
            self.metrics["draws_ratio"],
            label="draws_ratio",
            color="#9467bd",
            alpha=0.7,
        )
        axs[3].set_xlabel("Episodes")
        axs[3].set_ylabel("Ratio Value")

        # Common styling for all subplots
        for ax in axs:
            ax.legend(loc="upper right")
            ax.set_xlim(episodes[0], episodes[-1])
            ax.grid(True, linestyle="--", alpha=0.7)
            sns.despine(ax=ax)

        # Set title for the entire figure
        plt.suptitle("Training Metrics", y=0.95, fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plot_path = save_dir / "training_metrics_seaborn_batch.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\nPlot saved to: {plot_path}")


def main():
    metrics_path = "./metrics/dqn_training_metrics_random_first_player_140000.json"
    visualizer = MetricsVisualizer(metrics_path)

    # Print detailed debug information
    visualizer.debug_print_metrics()

    # Create the multi-panel plot
    visualizer.plot_metrics()


if __name__ == "__main__":
    main()
