import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_activity(history, mode="lines", out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    losses = np.array(history["loss"])
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Meta-training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    if mode == "lines":
        outs = np.array(history["outputs"])
        targs = np.array(history["targets"])
        plt.figure(figsize=(6,4))
        plt.plot(outs.mean(axis=1), label="outputs (mean)")
        plt.plot(targs.mean(axis=1), label="targets (mean)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "outputs_targets.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=None)
    parser.add_argument("--mode", type=str, default="lines")
    args = parser.parse_args()
    # If run standalone without history, just notify
    print("Use visualize_activity(history) from training scripts.")
