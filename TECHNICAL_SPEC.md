# Technical Specification: FeedbackLoop RLHF Pipeline

This technical specification details the architecture, modules, dependencies and data flow for the FeedbackLoop project.

## System Architecture

- **Data Ingestion & Preprocessing:** Scripts to load and clean candidate text data (resumes, interview transcripts). Includes tokenization and anonymization of sensitive attributes.
- **Annotation Interface:** A web-based application (Streamlit or Flask) where recruiters compare two model responses and select their preferred one. Stores (prompt, response_a, response_b, preferred) tuples in JSON/CSV.
- **Reward Model Training:** Functions to train a lightweight reward model (e.g. DistilBERT) on preference data using a pairwise or Bradley‑Terry loss.
- **Policy Optimization:** Trainers implementing PPO and DPO algorithms to fine‑tune a base LLM using the reward model scores【165012901649046†L48-L57】【165012901649046†L59-L65】. Allows configurable hyperparameters like learning rate, KL penalty and number of epochs.
- **Evaluation & Logging:** Utilities to compute average reward, win‑rate against the base model, KL divergence, and fairness metrics (via BiasGuard). Generates plots and tables.
- **Configuration & Storage:** YAML/JSON config files to specify dataset paths, model checkpoints and hyperparameters. Scripts to save intermediate checkpoints, logs and final models.

## Dependencies

- Python ≥3.8.
- **Libraries:**
  - `pandas` and `numpy` for data manipulation.
  - `transformers` and `datasets` for LLMs and tokenization.
  - `trl` or other RLHF frameworks for PPO/DPO training.
  - `scikit-learn` for preprocessing and evaluation metrics.
  - `flask` or `streamlit` for the annotation interface.
  - `matplotlib` or `plotly` for plotting results.
- Optional: `torch` with GPU support for efficient training; `fairlearn` for fairness metrics.

## Data Flow

1. A dataset of prompts (e.g. interview questions) is loaded. A base model generates candidate responses.
2. Recruiters (or simulated labelers) use the annotation interface to compare responses and record preferences, producing a preference dataset.
3. The reward model is trained on the preference dataset. It takes (prompt, response) pairs and outputs scalar scores.
4. The policy optimization module fine‑tunes the base model using PPO or DPO, sampling new responses, scoring them with the reward model and updating the policy. Checkpoints are saved periodically.
5. After training, evaluation scripts compute metrics (mean reward, win‑rate) and fairness measures, and produce plots/reports.
6. All artifacts (datasets, trained models, logs) are stored in a structured directory defined in the config.

## Implementation Notes

- **Modularity:** Each module (annotation, reward training, policy optimization, evaluation) is implemented as a separate Python script or class to facilitate reuse and testing.
- **Configurability:** Hyperparameters, model names and dataset paths are defined in config files. Users can switch between PPO and DPO by changing one flag.
- **Efficiency:** Use mixed‑precision and gradient accumulation to train large models on free GPUs. Sample small batches to fit into limited memory.
- **Extensibility:** Additional RL algorithms (e.g. TRPO or QLoRA-based preference fine‑tuning) can be added by following the existing interface. More sophisticated annotation UIs can be plugged in.

## Scalability and Deployment

FeedbackLoop is designed to run on free‑tier GPU environments (Google Colab, Kaggle) with small models. For larger datasets or production use, containerize the modules with Docker and orchestrate training on cloud GPUs or clusters. Provide instructions for running notebooks locally or in the cloud. For hosting the annotation interface, use a simple deployment (e.g. Heroku or GitHub Pages) and restrict access to authorized recruiters. Use persistent storage (e.g. GCS, S3) for data and model artifacts.

