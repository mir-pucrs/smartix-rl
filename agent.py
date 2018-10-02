import math
import glob
import mysql.connector
import re
import numpy as np
from state import State
from environment import Environment
from q_value import Q_Value


class Agent:

    def __init__(self):
        self.s = self.a = self.r = self.prev_s = self.prev_a = self.prev_r = None
        self.q_table = dict()
        self.columns_of_table = list()
        self.columns_of_queries = list()
        self.utility_table = {}
        self.state = State()
        self.env = Environment()
        self.frequency = dict()

    def reset(self):
        self.s = self.state.get_indexes()
        return self.s


    def f(self, qv):
        if qv in self.frequency:
            return 0 #self.q_values[qv]
        else:
            return 9999999

    def argmax(self,state):
        v = -9999999
        pa = None
        for a in self.env.available_actions(state):
            qv = Q_Value(state, a)
            value = self.f(qv)
            if value > v:
                v = value
                pa = a
        return pa

    def get_action(self, state): 
        a = self.argmax(state)
        self.frequency[Q_Value(state, a)] = 1
        return a

    def train(self):
        executions = 0
        self.columns_of_table = self.state.get_columns_of_table()
        self.columns_of_queries = self.state.get_columns_of_queries()

        self.q_table = np.zeros((len(self.columns_of_table), len(self.columns_of_queries)))

        for e in range(executions):
            self.prev_s = self.reset()
            done = False
            while not done:
                if np.sum(self.q_table[self.prev_s, :]) == 0:
                    # if q_table empty then choose random action
                    self.a = np.random.randint(0, 2)
                    # self.s = random.choice(self.columns_of_queries)
                else:
                    # else select the action with highest cumulative reward
                    self.a = np.argmax(self.q_table[self.prev_s, :])
                self.s, self.r, done, _ = self.env.execute(self.a)
                self.q_table[self.prev_s, self.a] += self.r
                self.prev_s = self.s
                # self.env.add_index(self.s)
                # self.s = self.env.get_indexes()
        print('testing numpy zeros:', self.q_table)
        return self.q_table
