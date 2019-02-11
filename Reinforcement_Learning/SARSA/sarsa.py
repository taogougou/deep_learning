from __future__ import print_function
import numpy as np
import time
from maze_env import maze_env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 50

np.random.seed(1)

def epsilon_greedy(Q, state):
    if (np.random.uniform() > 1 - EPSILON) or ((Q[state, :] == 0).all()):
        action = np.random.randint(0, 4)  # 0~3
    else:
        action = Q[state, :].argmax()
    return action


e = maze_env()
Q = np.zeros((e.state_num, 4))

for i in range(200):
    e = maze_env()
    action = epsilon_greedy(Q, e.present_state)
    while (e.is_end is False) and (e.step < MAX_STEP):
        state = e.present_state
        reward = e.interact(action)
        new_state = e.present_state
        new_action = epsilon_greedy(Q, e.present_state)
        Q[state, action] = (1 - ALPHA) * Q[state, action] + ALPHA * (reward + GAMMA * Q[new_state, new_action])
        action = new_action
        time.sleep(0.1)
        print('循环次数:', i, '总步数:', e.step, '总奖励数:', e.total_reward)
    time.sleep(2)