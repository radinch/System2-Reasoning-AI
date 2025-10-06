# System 2 Reasoning AI
**Experiments on deliberate, interpretable reasoning across neuro-symbolic models, symbolic regression, inference-time scaling, LLM agents, RL post-training in LLMs, and graph-based retrieval.**

---

## Overview
This repository gathers six advanced projects exploring how modern AI systems can perform **System 2–style reasoning** — deliberate, structured, and interpretable decision‑making that combines neural and symbolic approaches. Each notebook represents a distinct yet complementary direction toward **explainable, compositional, and multi‑step reasoning**.

The work spans topics from **neuro‑symbolic program induction** and **symbolic regression** to **inference‑time reasoning optimization**, **reinforcement learning post‑training for LLMs**, **vision‑based LLM agents**, and **GraphRAG‑style retrieval**.

---

## Contents

| # | Project | Path | Description |
|---|--------|------|-------------|
| 1 | **Neuro‑Symbolic Reasoning** | `neuro_symbolic_reasoning/neuro_symbolic_reasoning.ipynb` | CLEVR question answering via program induction, symbolic execution, and seq2seq models (LSTM + Transformer). |
| 2 | **Symbolic Regression** | `symbolic_regression/symbolic_regression.ipynb` | Equation discovery with Equation Learner (EQL) layers and a Transformer Seq2Seq model (token vocabulary → SymPy expressions). |
| 3 | **Inference‑Time Scaling** | `inference_time_scaling/inference_time_scaling.ipynb` | Compares Chain‑of‑Thought, Best‑of‑N, Beam Search, Self‑Refine, Tree‑of‑Thoughts, A*, and MCTS on math reasoning. |
| 4 | **RL Post‑Training for LLMs** | `rl_post_training/rl_post_training.ipynb` | Two‑stage fine‑tuning (SFT → GRPO/TRL) with custom rewards for structure (`<think>…</think>` & `<answer>…</answer>`) and correctness. |
| 5 | **Vision LLM Agent** | `vision_llm_agent/vision_llm_agent.ipynb` | Multi‑agent vision reasoning combining OpenCV heuristics with a VLM (Qwen‑VL). Includes ablations and a small 100‑image dataset. |
| 6 | **GraphRAG Pipeline** | `graph_rag/graph_rag.ipynb` | Graph‑based retrieval and community reasoning with entity extraction, Leiden community detection, and community‑scoped answering. |

---

## Core Themes
- **System 2 Reasoning:** deliberate, multi‑step problem solving  
- **Hybrid Neuro‑Symbolic Learning:** combining neural inference with symbolic structure  
- **Reasoning‑Time Optimization:** inference‑time scaling, beam search, and ToT/MCTS exploration  
- **Reinforcement Learning Post‑Training:** reward shaping for structured reasoning in LLMs  
- **LLM Agents & Vision Integration:** combining perception with reasoning chains  
- **Graph‑Based Retrieval:** leveraging community structure for contextual memory  

---

## Repository Structure
```
System2-Reasoning-AI
│   .gitignore
│   LICENSE
│   README.md
│
├── graph_rag
│   └── graph_rag.ipynb
│
├── inference_time_scaling
│   └── inference_time_scaling.ipynb
│
├── neuro_symbolic_reasoning
│   │   neuro_symbolic_reasoning.ipynb
│   │   prompt_example.txt
│   │
│   ├── dataH5Files
│   │   └── (dataset files)
│   │
│   └── utils
│       ├── clevr_executor.py
│       ├── logger.py
│       ├── preprocess.py
│       ├── preprocess_questions.py
│       ├── programs.py
│       ├── utils.py
│       └── __init__.py
│
├── rl_post_training
│   └── rl_post_training.ipynb
│
├── symbolic_regression
│   ├── dataset.csv
│   └── symbolic_regression.ipynb
│
└── vision_llm_agent
    │   vision_llm_agent.ipynb
    │
    └── agent_data
        │   data.csv
        │
        └── images
            ├── 1018.png
            ├── 10461.png
            ├── 10546.png
            ├── 10916.png
            ├── 11286.png
            ├── ...
            └── 9588.png

```

Each folder contains an independent Jupyter notebook and (where applicable) the supporting data included in your upload.

---

## Environment Setup
> These notebooks target Python 3.10+. Install only what you need for the notebook you plan to run.

```bash
pip install torch transformers accelerate datasets tqdm numpy matplotlib pandas sympy scikit-learn             trl peft vllm opencv-python pillow             langchain langchain-community langchain-graphrag networkx cdlib pypdf
```
**Notes**
- Some parts (e.g., RL post‑training and GraphRAG community detection) benefit from a **GPU** runtime.  
- If using verifier or external API calls in the inference‑time scaling notebook, configure your **API keys via environment variables** instead of hard‑coding.

---

## Selected Observations
- **Vision LLM Agent:** In the provided ablations, the best deep‑agent configuration (Agent 1 + Agent 3) outperformed zero‑shot and classic pipelines on the 100‑image set.  
- **Inference‑Time Scaling:** Search‑based and verification‑based strategies (e.g., Best‑of‑N, ToT/MCTS) showed higher accuracy than plain CoT at additional compute cost.  
- **RL Post‑Training for LLMs:** GRPO with structured rewards improved format compliance and answer correctness relative to SFT‑only baselines.  
- **GraphRAG:** Community summaries derived from the entity‑relation graph improved long‑document question answering quality relative to naive chunk retrieval.

*(Exact metrics depend on runtime settings and hardware; see notebook outputs for details.)*

---

## Motivation
System 1 reasoning in LLMs is fast but often shallow. This project explores **System 2 reasoning** — deliberate, symbolic, and interpretable — by experimenting with architectures and training strategies that encourage models to reason, plan, and reflect.

---

## How to Mention This Work
If you reference this repository in your CV or portfolio:
> *“Developed the `system2-reasoning-ai` repository — a collection of experiments integrating neuro‑symbolic learning, inference‑time scaling, reinforcement learning post‑training in LLMs, LLM agents, and graph‑based retrieval for advanced reasoning.”*