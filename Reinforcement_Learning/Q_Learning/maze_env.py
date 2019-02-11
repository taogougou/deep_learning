from __future__ import print_function
import copy
'''
设置迷宫的环境
'''

#迷宫的形状
maze = \
    '''
.........
.       .
.   x   .
.  xo   .
.........
'''
maze = maze.strip().split('\n')
maze = [[c for c in line] for line in maze]
print(maze)
#表示在上下方向的移动
up_down = [-1, 1, 0, 0]
#在左右方向的变化
left_right = [0, 0, -1, 1]

class maze_env(object):
    #初始化智能体A
    def __init__(self):
        self.maze_copy = copy.deepcopy(maze)
        self.x = 1
        self.y = 1
        self.step = 0
        self.total_reward = 0
        #跳出学习的条件，碰到'.'跳出学习
        self.is_end = False

    #action下获得的reward
    def interact(self, action):
        assert self.is_end is False
        new_x = self.x + up_down[action]
        new_y = self.y + left_right[action]
        new_pos_char = self.maze_copy[new_x][new_y]
        self.step += 1
        if new_pos_char == '.':
            reward = 0  #到达边界，奖励0，不更新位置，使用原来的位置
        elif new_pos_char == ' ':
            self.x = new_x
            self.y = new_y
            reward = 0
        elif new_pos_char == 'o':
            self.x = new_x
            self.y = new_y
            self.maze_copy[new_x][new_y] = ' '  #
            self.is_end = True  # 跳出学习
            reward = 100
        elif new_pos_char == 'x':
            self.x = new_x
            self.y = new_y
            self.maze_copy[new_x][new_y] = ' '  # update map
            self.is_end = True
            reward = -5
        self.total_reward += reward
        return reward

    #总的状态数，
    @property
    def state_num(self):
        rows = len(self.maze_copy)
        cols = len(self.maze_copy[0])
        return rows * cols

    #当前所在的状态，位置
    @property
    def present_state(self):
        cols = len(self.maze_copy[0])
        return self.x * cols + self.y




