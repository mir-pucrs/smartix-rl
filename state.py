import glob
import mysql.connector
import re


class State:

    def __init__(self):
        self.columns_of_table = list()
        self.columns_of_queries = list()
        self.map_indexes = dict()
        self.indexes = list()

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
        # It is returning 138 columns - needs to verify duplicate columns
        filepaths = glob.glob('/home/priscillaneuhaus/SAP-Project/TPCH/2.17.3/dbgen/queries/*.sql')
        for file in filepaths:
            with open(file, 'r') as f:
                content = re.findall(r'l_[a-z]+', f.read())
                self.columns_of_queries += content
        return self.columns_of_queries

    def get_indexes(self):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute('show indexes from lineitem')
        d = cursor.fetchall()
        aux = dict()
        for i in range(0, len(d)):
            aux[i] = d[i][4]
        self.indexes = list(aux.values())
        return self.indexes

    def get_map_indexes(self):
        size = self.get_columns_of_table()
        index = self.get_indexes()
        for item in size:
            self.map_indexes[item] = 0
            for item2 in index:
                if item == item2:
                    self.map_indexes[item] = 1
        return self.map_indexes

    def reset_map_indexes(self):
        size = self.get_columns_of_table()
        for i in range(len(size)):
            self.map_indexes[size[i]] = 0
        return self.map_indexes
