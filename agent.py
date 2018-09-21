import math
import glob
import mysql.connector
import re
import numpy as np
from state import State
from environment import Environment


class Agent:

    def __init__(self):
        self.s = self.a = self.r = self.prev_s = self.prev_a = self.prev_r = None
        self.q_table = dict()
        self.columns_of_table = list()
        self.columns_of_queries = list()
        self.map_indexes = list()
        self.utility_table = {}
        self.state = State()
        self.env = Environment()

    def reset(self):
        self.s = self.state.get_indexes()
        return self.s

    def get_columns_of_table(self):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute('SHOW COLUMNS FROM lineitem;')
        d = cursor.fetchall()
        aux = dict()
        for i in range(0, len(d)):
            aux[i] = d[i][0]
        self.columns_of_table = list(aux.values())
        return self.columns_of_table

    def get_columns_of_queries(self):
        filepaths = glob.glob('/home/priscillaneuhaus/SAP-Project/TPCH/2.17.3/dbgen/queries/*.sql')
        for file in filepaths:
            with open(file, 'r') as f:
                content = re.findall(r'l_[a-z]+', f.read())
                self.columns_of_queries += content
        return self.columns_of_queries

    def reset_map_indexes(self):
        size = self.get_columns_of_table()
        for i in range(len(size)):
            self.map_indexes.append(0)
        print(self.map_indexes)
        return self.map_indexes

    def train(self):
        executions = 0
        self.columns_of_table = self.get_columns_of_table()
        self.columns_of_queries = self.get_columns_of_queries()
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
