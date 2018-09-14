import mysql.connector


class Environment:

    def __init__(self):
        self.current_state = list()
        self.reward = {}
        self.show_indexes = dict()
        self.indexes = dict()

    def get_indexes(self):
        cnx1 = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx1.cursor(buffered=True)
        cursor.execute('show indexes from lineitem')
        d = cursor.fetchall()
        for i in range(0, len(d)):
            self.show_indexes[i] = d[i][0], d[i][2], d[i][3], d[i][4], d[i][10]
            self.indexes[i] = d[i][4]
        print('show_indexes', self.show_indexes)
        print('indexes', self.indexes)
        return self.indexes

    def build_env(self):
        for item in self.indexes.items():
            self.current_state.append(item)
        print('list current_state', self.current_state)

    def add_index(self):
        """
        TODO: implement function to add index
        """

    def drop_index(self):
        """
        TODO: implement function to remove index
        """

    def run_benchmark(self):
        """
        TODO: implement function to run benchmark
        """

    def comp_rewards(self):
        """
        TODO: implement function to compute rewards
        """