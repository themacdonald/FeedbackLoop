from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="feedbackloop", description="FeedbackLoop CLI (v0.1)")
    parser.add_argument("--demo", action="store_true", help="Run a demo preference collection and reward model training")
    args = parser.parse_args()

    if not args.demo:
        raise SystemExit("v0.1 supports --demo only. Next step: add full RLHF pipeline.")

    # generate simple synthetic preference data
    preferences = []
    for i in range(3):
        prompt = f"Prompt {i}"
        preferred = random.choice(["A", "B"])
        preferences.append((prompt, preferred))

    # save a stub 'reward model'
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "reward_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"preferences": preferences}, f)

    print(f"Demo reward model saved to {model_path}")
