# 🤖 VLA-CBF Bridge  
### Safety-Aware Embodied AI

#### *Grounding Vision-Language-Action Models with Classical Safety Constraints*

---

## ✨ Overview

This repository presents a hierarchical embodied AI framework that combines the semantic reasoning capabilities of **Vision-Language-Action (VLA)** models with the formal safety guarantees of **Vision-based Control Barrier Functions (VOCBF)**.

Modern VLA systems excel at high-level task understanding but often fail under strict hardware, contact, and safety constraints. This project introduces a decoupled architecture where:

- 🧠 A **foundation model** performs semantic task planning from visual observations.
- 🛡️ A **reactive safety layer** guarantees collision-aware robot control in real time.

> **High-level intelligence proposes actions.  
> Low-level control guarantees safety.**

---

## 🧩 System Architecture

### 🧠 Semantic Planner
A quantized **Qwen-2.5-VL-3B** model processes wrist-camera imagery and predicts task-level Cartesian targets.

### 🛡️ Reactive Safety Layer
A **VOCBF-based safety filter** continuously evaluates the proposed trajectory and overrides unsafe robot commands in real time.

### ⚙️ Differential IK Controller
Cartesian end-effector targets are mapped into robot joint-space commands using **Differential Inverse Kinematics (Diff-IK)**.

---

## 📁 Repository Structure

```text
vla-franka/
│
├── assets/
│   └── franka_emika_panda/
│       ├── scene.xml
│       ├── franka.xml
│       ├── meshes/
│       └── textures/
│
├── env/
│   └── camera.py
│
├── kinematics/
│   └── diff_ik.py
│
├── policies/
│   ├── pi0_policy.py
│   └── action_space.py
│
├── experiments/
│   ├── run_vla_reach.py          # Main execution entrypoint
│   └── data_collection/
│       └── collect_vocbf_data.py
│
├── src/
│   ├── config/
│   ├── vocbf/
│   │   └── collect_vocbf_data.py
│   └── utils/
│
└── README.md
```

---

## 🚀 Methodology

| Module | Description |
|---|---|
| 👁️ Perception | Frozen **DINOv2** spatial embeddings for visual grounding |
| 🧠 High-Level Policy | 4-bit quantized VLA for semantic action generation |
| ⚙️ Kinematics | Differential IK for Cartesian-to-joint-space mapping |
| 🛡️ Safety Filter | QP-based VOCBF controller for safe reactive motion |
| 🔄 Runtime | Asynchronous inference pipeline for decoupled control |

---

## ⚡ Control Pipeline

```text
Wrist Camera Image
        ↓
Qwen-2.5-VL Semantic Planner
        ↓
Desired Cartesian Target
        ↓
Differential IK Controller
        ↓
VOCBF Safety QP Filter
        ↓
Safe Joint Commands
        ↓
Franka Panda Robot
```

---

## 📊 Current Status

- [x] MuJoCo simulation environment with **Franka Emika Panda**
- [x] VOCBF safety filter for cylindrical obstacles
- [x] Asynchronous VLA inference pipeline
- [ ] Closed-loop trajectory refinement
- [ ] Benchmarking against heuristic safety baselines

---

## 🧠 Key Research Insight

> **Foundation models can reason semantically,  
> but safety must remain reactive, local, and formally constrained.**

This project explores how modern embodied AI systems can be grounded using classical control-theoretic safety guarantees.

---

## ⚙️ Requirements

- Python ≥ 3.9
- MuJoCo
- PyTorch
- NumPy
- Transformers
- CUDA (recommended)

---

## 🏛️ Institutional Context

Developed at the Indian Institute of Science (IISc), Bengaluru.

**Advisor:** Prof. Shishir N. Y. Kolathaya

---