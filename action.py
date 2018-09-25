import mysql.connector
from agent import Agent
from state import State
import os


class Action:

    def __init__(self):
        self.current_state = dict()
        self.agent = Agent()
        self.state = State()
        self.map_indexes = self.state.reset_map_indexes()

    def drop_index(self, index):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        command = ('DROP INDEX %s ON lineitem;' % str(index))
        cursor.execute(command)

    def add_index(self, column):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        with open('../name_of_indexes', 'r') as fin:
            data = fin.readline().strip()
            i = data[5]
            index_numb = int(i)
            command = 'CREATE INDEX %s ON lineitem (%s);' % (data, column)
            cursor.execute(command)
            index_numb += 1
            name = 'index' + str(index_numb)
            print(name)
            with open('../name_of_indexes', 'w') as fout:
                fout.writelines(name)

    # def add_index_first(self, column):
    #     cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
    #     cursor = cnx.cursor(buffered=True)
    #     for line in open('../name_of_indexes'):
    #         command = ('CREATE INDEX %s ON lineitem (%s);\n' % (line.replace('\n', ''), column))
    #         cursor.execute(command)

