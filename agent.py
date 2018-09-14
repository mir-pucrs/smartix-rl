import math
from environment import Environment


class Agent:

    def __init__(self):
        self.s = self.a = self.r = None
        self.q_table = dict()
        self.utility_table = {}
        self.env = Environment()
        ACTIONS = [self.env.add_index(), self.env.drop_index()]

    def reset(self):
        self.s = self.env.get_indexes()

    def train(self,env):
        executions = 0
        while executions < 100:
            # start with none
            self.prev_s, self.prev_a, self.prev_r = self.s, self.a, self.r

            # when the train over: reset the environment to get the initial state
            self.reset()