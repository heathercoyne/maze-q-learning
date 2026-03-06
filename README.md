# Reinforcement Learning Maze Solver

This project implements a **maze treasure navigation problem** using reinforcement learning.  
An agent (Laura) learns to move through a grid maze, avoid traps, and reach a treasure.

Two reinforcement learning algorithms are implemented and compared:

- **Q-learning**
- **2-step Q-learning**

The maze environment is visualized using a **Tkinter GUI**, allowing the training process and final learned path to be observed.

---

# Maze Environment

Below is an example of the maze environment used for training.

<img width="1521" height="498" alt="image" src="https://github.com/user-attachments/assets/3e2c03d3-c0d6-48be-aae4-efeb11ce54a5" />

The environment is a **6×6 grid maze** where the agent must reach the treasure while avoiding traps.

### Rewards

| Event | Reward |
|------|------|
Reach treasure | +1 |
Fall into trap | -1 |
Normal movement | 0 |

### Actions

The agent can move in four directions:

- Up
- Down
- Left
- Right

---

# Algorithms

## Q-learning

Standard Q-learning updates the Q-table using the Bellman equation:

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

Where:

- **α** = learning rate  
- **γ** = discount factor  
- **r** = reward  
- **s'** = next state  

An **ε-greedy policy** is used for exploration.

---

## 2-Step Q-learning

The 2-step method incorporates rewards from two future time steps:

```
Q(s_t,a_t) ← Q(s_t,a_t) +
α [ r_{t+1} + γ r_{t+2} + γ² max_a Q(s_{t+2},a) − Q(s_t,a_t) ]
```

Using multi-step updates allows reward information to propagate faster during training.

---

# Project Structure

```
maze-q-learning
│
├── q_learning.py
├── two_step_q_learning.py
│
├── trap.png
├── laura.png
├── treasure.png
│
├── images
│   └── maze_example.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

### File Description

| File | Description |
|----|----|
q_learning.py | Standard Q-learning implementation |
two_step_q_learning.py | 2-step Q-learning implementation |
trap.png | Trap icon used in maze visualization |
laura.png | Agent character |
treasure.png | Treasure icon |
requirements.txt | Python dependencies |

---

# Installation

Clone the repository:

```
git clone https://github.com/heathercoyne/maze-q-learning.git
cd maze-q-learning
```

Install required packages:

```
pip install -r requirements.txt
```

Python's built-in modules (`tkinter`, `sys`, `time`) are included with standard Python installations.

---

# Running the Program

Run the standard Q-learning implementation:

```
python q_learning.py
```

Run the 2-step Q-learning implementation:

```
python two_step_q_learning.py
```

A GUI window will open showing the maze environment.

During training the agent explores the maze and gradually learns an optimal path to the treasure.

After training finishes, a **final demonstration** runs showing the learned optimal route.

---

# Example Behavior

At the beginning of training the agent moves randomly through the maze while exploring.

Over time the Q-table improves and the agent learns a path that reliably reaches the treasure while avoiding traps.

The final demo shows the learned policy navigating efficiently from the start position to the goal.

---

# Acknowledgement

The maze environment and GUI framework were provided as starter code in an Artificial Intelligence course.

The reinforcement learning algorithms, training logic, and experiments were implemented by the author.

---

# Author

Heather Xin Coyne  
Tsinghua University  
Artificial Intelligence Course Project

