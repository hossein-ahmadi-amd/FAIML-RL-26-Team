# FAIML RL Project — Group 12
**Course:** Fundamentals of Artificial Intelligence, Machine and Deep Learning (01VSDWS)  
**A.Y.:** 2025/2026 — Politecnico di Torino

## Team
| Name | Student ID |
|---|---|
| Sreelekshmi Manju | s358798 |
| Merry Bonifaise | s359201 |
| Hossein Ahmadi | s351951 |
| Prince Thekkedath | s358334 |

---

## Project Overview
This project implements and evaluates reinforcement learning algorithms for continuous robotic control, focusing on the sim-to-real transfer problem. It covers:

- **Part 1 (Hopper-v4):** REINFORCE (with and without baseline) and Actor-Critic from scratch in PyTorch
- **Part 2 (PandaPush-v3):** PPO and SAC via Stable-Baselines3, with Uniform Domain Randomization (UDR) and Automatic Domain Randomization (ADR)

---

## Repository Structure

```
FAIML-RL-26-Team/
├── requirements.txt
├── part1/                        # Hopper-v4 — Tasks 1–3
│   ├── agent.py                  # Policy, Agent, REINFORCE, Actor-Critic
│   ├── train.py                  # Training loop
│   ├── test_random_policy.py     # Random policy baseline
│   └── colab_template/
│       └── test_random_policy.ipynb
└── part2/                        # PandaPush-v3 — Tasks 4–6
    ├── train_sb3.py              # PPO/SAC training with UDR/ADR
    ├── eval_sb3.py               # Evaluation script
    ├── rand_wrapper.py           # UDR and ADR wrapper
    ├── test_random_policy.py     # Random policy baseline
    └── panda-gym/                # Custom PandaGym environments
```

---

## Installation

```bash
pip install -r requirements.txt
cd part2/panda-gym && pip install -e .
```

---

## Usage

### Part 1 — Hopper-v4

REINFORCE without baseline:
```bash
cd part1
python train.py --algo reinforce
```

REINFORCE with baseline b=20:
```bash
cd part1
python train.py --algo reinforce --baseline 20
```

Actor-Critic:
```bash
cd part1
python train.py --algo ac
```

---

### Part 2 — PandaPush-v3

Train SAC on source (no DR):
```bash
cd part2
python train_sb3.py --algo sac --env-type source
```

Train SAC with UDR covering range:
```bash
cd part2
python train_sb3.py --algo sac --sampling-strategy udr --mass-range 0.5 6.0 --env-type source
```

Train SAC with ADR:
```bash
cd part2
python train_sb3.py --algo sac --sampling-strategy adr --mass-range 1.0 1.0 --env-type source
```

Evaluate a saved model on the target domain (50 episodes):
```bash
cd part2
python eval_sb3.py --algo sac --model-path sac_push_none_source_500k --env-type target
```
