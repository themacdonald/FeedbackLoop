# FeedbackLoop

**Goal:** Develop a full reinforcement learning from human feedback (RLHF) pipeline tailored for recruiting chatbots and scoring models. This system collects human feedback from recruiters, trains a reward model and fine‑tunes a language model using algorithms such as PPO or Direct Preference Optimization (DPO). RLHF aligns AI with human values by training on human feedback rather than hand‑crafted reward functions【165012901649046†L48-L57】, and typically involves stages from data generation to reward modeling and final training【165012901649046†L59-L65】.

## Features

- **Feedback Collection Interface:** A web‑based annotation tool where recruiters compare two responses and record which one is preferred. Data is stored as `(prompt, response_A, response_B, preferred)` pairs.
- **Reward Model Trainer:** A module that trains a lightweight reward model to assign scalar scores to responses, reflecting recruiter preferences.
- **Policy Optimization:** Scripts to fine‑tune a base model (e.g. Llama‑2 or open‑source 7B models) using RLHF techniques. Supports PPO with KL‑penalty or DPO for efficiency.
- **Evaluation Suite:** Tools to compare the fine‑tuned model against the base model using automatic metrics and human‑style tests (win‑rates), plus fairness metrics via BiasGuard to ensure improvements don’t exacerbate bias.
- **Colab Integration:** Provide Jupyter notebooks for each stage (feedback collection simulation, reward model training, RL optimization) that run on Colab's free GPUs.

## Quick Start

1. Clone the repository and install dependencies (Python 3.8, `transformers`, `trl`, `streamlit`):
   ```bash
   git clone https://github.com/themacdonald/FeedbackLoop.git
   cd FeedbackLoop
   pip install -r requirements.txt
   ```
2. Run `feedback_app.py` to launch the annotation interface and collect sample preference data.
3. Train the reward model by running `python train_reward_model.py`.
4. Fine‑tune the policy via PPO or DPO using `python train_rl_agent.py --method ppo` or `--method dpo`.
5. Evaluate the resulting model with `python evaluate_model.py`, including fairness checks with BiasGuard.

## Implementation Plan

Two‑week sprint (~30 hours):

- **Week 1:** Build the annotation interface and collect a small preference dataset; implement reward model training using Hugging Face’s libraries.
- **Week 2:** Implement PPO/DPO training loops, monitor reward and KL metrics, evaluate the fine‑tuned model, and create a high‑level dashboard.

## Contributing

Contributions are welcome! Please open issues or submit pull requests to add features, fix bugs, or improve documentation. See `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License (see `LICENSE` for details).
