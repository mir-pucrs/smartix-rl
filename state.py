import mysql.connector


class State:

    def __init__(self):
        self.current_state = list()
        self.map_indexes = list()
        self.indexes = list()

    def get_indexes(self):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute('show indexes from lineitem')
        d = cursor.fetchall()
        aux = dict()
        for i in range(0, len(d)):
            aux[i] = d[i][4]
        self.indexes = list(aux.values())
        print('indexes: ', self.indexes)
        return self.indexes

    def add_index(self, column):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        for line in open('../name_of_indexes'):
            command = ('CREATE INDEX %s ON lineitem (%s);\n' % (line.replace('\n', ''), column))
            cursor.execute(command)

    def drop_index(self, index):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        command = ('DROP INDEX %s ON lineitem;' % str(index))
        cursor.execute(command)
