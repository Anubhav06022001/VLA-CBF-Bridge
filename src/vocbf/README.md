# Value-Oriented Control Barrier Function (V-OCBF)

This module contains implementations for learning **Value-Guided Offline Control Barrier Functions** for safe robotic control of the Franka Panda manipulator.

The barrier function is trained using two learning paradigms:

- Transition-data based learning  
- Dynamics-model based learning  

Both approaches learn a safety value function used later in CBF-QP controllers.

---

## 📂 Files

### `barrier_net.py`

Defines the neural network architecture used to approximate the barrier value function.

---

## 🧪 Training Scripts

---


### train_vocbf_dynamics_model2_mj_geom.py

```bash

q, qd, u ,q_next, qd_next

u-->q_des
```

Batch gradient kle rahe hain last mwe trianing loop pda hai  bina batch ke


site ki bajay mj.mj_geom use kar rahe hain


### train_vocbf.py

```bash

q, qd, u ,q_next, qd_next

u-->q_des
```

learn vocbf without using dynamic model