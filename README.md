# Soft-Actor-Critic-SAC-for-Continuous-Control-Walker2D-v5-
This project implements a Soft Actor–Critic (SAC) reinforcement learning algorithm in PyTorch, designed for training and evaluating agents on continuous control environments such as Walker2d-v5 (MuJoCo).


# 🧠 Soft Actor–Critic (SAC) — Walker2D-v5 (PyTorch)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A clean, GPU-accelerated **Soft Actor–Critic (SAC)** implementation in **PyTorch**, trained and evaluated on the **Walker2D-v5** environment from OpenAI Gymnasium (MuJoCo).  
SAC is a **state-of-the-art reinforcement learning algorithm** that combines off-policy learning and entropy maximization for stable and efficient continuous control.

---

## 🚀 Features

- ✅ Full **Soft Actor–Critic (SAC)** implementation (actor–critic + auto entropy tuning)  
- ✅ **GPU acceleration (CUDA)**  
- ✅ **TensorBoard** logging for training metrics  
- ✅ **Replay buffer** + **Polyak averaging** for target networks  
- ✅ **Automatic checkpoint saving** every N steps  
- ✅ **Evaluation script** with live rendering and video recording  
- ✅ Works with **Walker2D-v5**, **Humanoid-v4**, and other MuJoCo environments  

---

## 📁 Project Structure

├── sac_torch.py # Core SAC implementation (Actor, Critics, Alpha tuning)
├── train_sac_walker2d_tb.py # Training script (with TensorBoard logging)
├── eval_sac_torch_walker.py # Evaluation script (render or record mode)
├── checkpoints_torch_walker/ # Checkpoints created during training
└── runs/sac_walker2d/ # TensorBoard logs


---

## ⚙️ Installation

Create a new environment with PyTorch and MuJoCo:

```bash
conda create -n rl-torch python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate rl-torch
pip install gymnasium[mujoco] tensorboard numpy
```

🏋️ Training

Train the SAC agent on Walker2D-v5:

```bash
python train_sac_walker2d_tb.py
```

Monitor progress in TensorBoard:
```bash
tensorboard --logdir runs/sac_walker2d --port 6006
```

TensorBoard shows:
Episode return and length
Critic, actor, and alpha losses
Replay buffer growth
Temperature (α) adaptation over time

🎮 Evaluation
Evaluate the trained policy:
```bash
python eval_sac_torch_walker.py --render human
```
Or record episodes to video:
```bash
python eval_sac_torch_walker.py --render record --video_dir videos_eval
```


🧰 Hyperparameters
Parameter	Value
| Parameter              | Value        |
| ---------------------- | ------------ |
| Batch size             | 526          |
| Learning rate          | 3e-4         |
| Discount factor (γ)    | 0.99         |
| Target update rate (τ) | 0.005        |
| Replay buffer size     | 1,000,000    |
| Warmup steps           | 10,000       |
| Evaluation interval    | 10,000 steps |
| Steps                  | 2,000,000    |
| Update every x steps   | 10           |

📚 Reference

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
Soft Actor–Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
International Conference on Machine Learning (ICML).
arXiv:1801.01290

👨‍💻 Author
Muhammad Samad
AI Engineer
M.Sc. Artificial Intelligence Engineering — Jönköping University 🇸🇪
📧 muhammadsamad2@gmail.com


