import math
import glob
import mysql.connector
import re
from environment import Environment


class Agent:

    def __init__(self):
        self.s = self.a = self.r = None
        self.q_table = dict()
        self.columns_of_table = list()
        self.columns_of_queries = list()
        self.utility_table = {}
        self.env = Environment()

    def reset(self):
        self.s = self.env.get_indexes()

    def get_columns_of_table(self):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute('SHOW COLUMNS FROM lineitem;')
        d = cursor.fetchall()
        aux = dict()
        for i in range(0, len(d)):
            aux[i] = d[i][0]
        self.columns_of_table = list(aux.values())
        print('Columns of table Lineitem: ', self.columns_of_table)
        return self.columns_of_table

    def get_columns_of_queries(self):
        filepaths = glob.glob('/home/priscillaneuhaus/SAP-Project/TPCH/2.17.3/dbgen/queries/*.sql')
        # columns = []
        for file in filepaths:
            with open(file, 'r') as f:
                content = re.findall(r'l_[a-z]+', f.read())
                self.columns_of_queries += content
        print('Columns of queries: ', self.columns_of_queries)
        return self.columns_of_queries

    def train(self,env):
        executions = 0
        while executions < 100:
            # start with none
            self.prev_s, self.prev_a, self.prev_r = self.s, self.a, self.r

            # when the training overs: reset the environment to get the initial state
            if executions == 100:
                self.reset()
                executions = 1
                self.prev_s = self.prev_a = self.prev_r = self.s = self.a = self.r = None