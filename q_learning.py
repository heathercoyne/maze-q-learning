import numpy as np
import pandas as pd
import time
import sys
import tkinter as tk
from tkinter import PhotoImage

UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6   # 迷宫的高度（格子数）
MAZE_W = 6   # 迷宫的宽度（格子数）
INIT_POS = [0, 0]  # 劳拉的起始位置
GOAL_POS = [3, 3]  # 宝藏的位置
TRAP_POS = [[2, 3], [3, 2], [2, 4], [4, 3]]  # 陷阱的位置

# speed control (fast for training, slow for final path)
FAST = True  
RESET_SLOW = 0.5
RENDER_SLOW = 0.1
RESET_FAST = 0.01
RENDER_FAST = 0.005


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 决策空间
        self.n_actions = len(self.action_space)
        self.title('Q-learning')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1, fill="black")
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, fill="black")

        origin = np.array([UNIT / 2, UNIT / 2])

        self.bm_trap = PhotoImage(file="trap.png")
        self.trap_list = []
        for i in range(len(TRAP_POS)):
            self.trap_list.append(self.canvas.create_image(
                origin[0] + UNIT * TRAP_POS[i][0], origin[1] + UNIT * TRAP_POS[i][1], image=self.bm_trap))

        self.bm_laura = PhotoImage(file="laura.png")
        self.laura = self.canvas.create_image(
            origin[0] + INIT_POS[0] * UNIT, origin[1] + INIT_POS[1] * UNIT, image=self.bm_laura)

        self.bm_goal = PhotoImage(file="treasure.png")
        self.goal = self.canvas.create_image(
            origin[0] + GOAL_POS[0] * UNIT, origin[1] + GOAL_POS[1] * UNIT, image=self.bm_goal)

        self.canvas.pack()

    def reset(self):
        self.update() # tk.Tk().update 强制画面更新
        
        if not FAST:
            time.sleep(RESET_SLOW)
        else:
            time.sleep(RESET_FAST)
        
        self.canvas.delete(self.laura)
        origin = np.array([UNIT / 2, UNIT / 2])

        self.laura = self.canvas.create_image(
            origin[0], origin[1], image=self.bm_laura)
        # 返回当前laura所在的位置
        return self.canvas.coords(self.laura)

    def step(self, action):
        s = self.canvas.coords(self.laura)
        base_action = np.array([0, 0])
        if action == 0:    # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:    # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:    # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:    # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.laura, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.laura)

        # 回报函数
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(a_trap) for a_trap in self.trap_list]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        if not FAST:
            time.sleep(RENDER_SLOW)
        else:
            time.sleep(RENDER_FAST)
        self.update()

class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.1, # learning rate
        gamma=0.9, # discount factor
        epsilon=0.9, # explore probability
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Q-table (rows are states, columns are actions)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # for converting pixel coords to discete key (x_y)
    def _state_to_key(self, s):
        if s == 'terminal':
            return 'terminal'
        x_pix, y_pix = float(s[0]), float(s[1])
        x = int(round((x_pix - UNIT / 2) / UNIT))
        y = int(round((y_pix - UNIT / 2) / UNIT))
        return f"{x}_{y}"

    def _ensure_state(self, s_key):
        if s_key not in self.q_table.index:
            self.q_table.loc[s_key, :] = 0.0

    def choose_action(self, s):
        s_key = self._state_to_key(s)
        self._ensure_state(s_key)

        if np.random.rand() > self.epsilon: # rand > epsilon is basically 1 - epsilon % exploit
            row = self.q_table.loc[s_key, :]
            max_q = row.max()
            best_actions = row[row == max_q].index.to_list()
            return int(np.random.choice(best_actions))
        else: # explore (epsilon % chance of epxplor )
            return int(np.random.choice(self.actions))

    #  Q(s,a) <- Q(s,a) + alpha * [ r + gamma*max_a' Q(s',a') - Q(s,a) ]
    def learn(self, s, a, r, s_):
        s_key = self._state_to_key(s)
        s_next_key = self._state_to_key(s_)
        self._ensure_state(s_key)
        if s_ != 'terminal':
            self._ensure_state(s_next_key)

        q_predict = self.q_table.loc[s_key, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_next_key, :].max()

        self.q_table.loc[s_key, a] = q_predict + self.alpha * (q_target - q_predict)

    def decay_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # best action without exploration (used for final demo)
    def best_action(self, s):
        s_key = self._state_to_key(s)
        self._ensure_state(s_key)
        row = self.q_table.loc[s_key, :]
        max_q = row.max()
        best_actions = row[row == max_q].index.to_list()
        return int(np.random.choice(best_actions))

def train_q_learning(
    env,
    agent,
    episodes=500,
    max_steps_per_episode=200,
    render_every=50, # visualize every N episodes
    stable_window=50, # check last N episodes
    success_threshold=0.9 # stop if success rate >= threshold
):
    success_hist = []
    step_hist = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        steps = 0
        success = 0

        # visualize some episodes to show path changes
        do_render = (render_every > 0 and (ep == 1 or ep % render_every == 0))

        while not done and steps < max_steps_per_episode:
            if do_render:
                env.render()

            a = agent.choose_action(s)
            s_, r, done = env.step(a)
            agent.learn(s, a, r, s_)
            s = s_
            steps += 1

            if done:
                success = 1 if r == 1 else 0

        agent.decay_eps()
        success_hist.append(success)
        step_hist.append(steps)

        # print training progress periodically
        if ep == 1 or ep % 50 == 0:
            recent = success_hist[-min(len(success_hist), stable_window):]
            recent_rate = float(np.mean(recent)) if len(recent) > 0 else 0.0
            print(f"[Train] episode={ep:4d}  epsilon={agent.epsilon:.3f}  steps={steps:3d}  recent_success={recent_rate:.2f}")

        # early stop when stable
        if len(success_hist) >= stable_window:
            recent = success_hist[-stable_window:]
            recent_rate = float(np.mean(recent)) if len(recent) > 0 else 0.0
            if float(np.mean(recent)) >= success_threshold:
                print(f"[Train] episode={ep:4d}  epsilon={agent.epsilon:.3f}  steps={steps:3d}  recent_success={recent_rate:.2f}")
                print(f"[Stop] success rate >= {success_threshold:.2f} over last {stable_window} episodes")
                break

    return success_hist, step_hist

# final demo to show the learned route clearly
def final_demo(env, agent, max_steps=200):
    print("[Demo] running greedy policy")
    s = env.reset()
    steps = 0
    while steps < max_steps:
        env.render()
        a = agent.best_action(s)
        s, r, done = env.step(a)
        steps += 1
        if done:
            msg = "treasure" if r == 1 else "trap"
            print(f"[Demo] finished in {steps} steps, result={msg}")
            break


def update():
    # create agent
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.995
    )

    train_q_learning(
        env=env,
        agent=agent,
        episodes=800,
        max_steps_per_episode=200,
        render_every=80,
        stable_window=60,
        success_threshold=0.9
    )

    global FAST
    FAST = False  # slow down for final demo
    final_demo(env, agent, max_steps=200)

    print("[Done] close the window to exit")


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
