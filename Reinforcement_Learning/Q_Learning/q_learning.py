from __future__ import print_function
import numpy as np
import time
from maze_env import maze_env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 30

np.random.seed(0)

def epsilon_greedy(Q, state):

    #如果所有元素都为真，那么返回真；否则返回假
    all = (Q[state, :] == 0).all()
    if (np.random.uniform() > 1 - EPSILON) or all:
        action = np.random.randint(0, 4)  # 0~3
    else:
        #返回每一个state行上的最大值的action下标
        action = Q[state, :].argmax()
    return action


e = maze_env()
Q = np.zeros((e.state_num, 4))
# print(Q.shape)

for i in range(200):
    e = maze_env()
    while (e.is_end is False) and (e.step < MAX_STEP):
        action = epsilon_greedy(Q, e.present_state)
        state = e.present_state
        reward = e.interact(action)
        new_state = e.present_state
        Q[state, action] = (1 - ALPHA) * Q[state, action] + \
            ALPHA * (reward + GAMMA * Q[new_state, :].max())
        print(Q)
        time.sleep(0.1)
    print('循环次数:', i, '总步数:', e.step, '总奖励数:', e.total_reward)
    time.sleep(2)

