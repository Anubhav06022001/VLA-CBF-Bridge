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

<div align="center">

# VLA-CBF Bridge  
### Safety-Aware Embodied AI

#### *Grounding Large Foundation Models with Classical Safety Constraints*

</div>

---

## Overview

This repository presents a hierarchical control framework that bridges high-level semantic reasoning from Vision-Language-Action (VLA) models with low-level robotic safety guarantees using **Vision-based Control Barrier Functions (VOCBF)**.

Current VLA systems demonstrate strong semantic capabilities but often lack robustness under strict hardware and safety constraints. This work proposes a decoupled architecture that combines foundation-model intelligence with formally grounded reactive control.

---

## System Architecture

### Semantic Planner
A quantized **Qwen-2.5-VL-3B** model processes wrist-camera observations to predict task-level target coordinates for manipulation.

### Reactive Safety Layer
A **VOCBF-based safety filter** continuously evaluates the VLA-generated trajectory and enforces real-time obstacle avoidance constraints.

---

## Methodology

### Perception
- Frozen **DINOv2** spatial embeddings for visual grounding.

### High-Level Control
- Asynchronous inference using a **4-bit quantized VLA**.
- Generation of 3D end-effector delta commands.

### Kinematics
- **Differential Inverse Kinematics (Diff-IK)** for mapping Cartesian targets to joint-space trajectories.

### Safety Filter
- A **Quadratic Program (QP)** formulation using learned **Vision-based Control Barrier Functions (VOCBF)**.
- Unsafe joint commands are overridden in real time to maintain safety guarantees.

---

## Current Status

- [x] MuJoCo simulation environment with **Franka Emika Panda**
- [x] VOCBF safety filter implementation for cylindrical obstacles
- [x] Asynchronous threading for decoupled VLA inference
- [ ] Real-time closed-loop refinement and benchmarking against heuristic baselines

---