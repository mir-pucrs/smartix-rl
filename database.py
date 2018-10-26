import pyodbc
import pprint

pyodbc.pooling = False

class Database:

    tables = {
        'customer': ['c_custkey', 'c_nationkey'],
        'lineitem': ['l_orderkey', 'l_linenumber', 'l_partkey', 'l_suppkey'],
        'nation': ['n_nationkey', 'n_regionkey'],
        'orders': ['o_orderkey', 'o_custkey'],
        'part': ['p_partkey'],
        'partsupp': ['ps_partkey', 'ps_suppkey'],
        'supplier': ['s_suppkey', 's_nationkey']
    }


    def __init__(self):
        self.connection_string = 'DRIVER={MySQL ODBC 8.0};SERVER=127.0.0.1;DATABASE=tpch1g;UID=dbuser;PWD=dbuser'
        # self.conn = pyodbc.connect(self.connection_string)
        # self.cur = self.conn.cursor()


    """
        Action-related methods
    """
    def drop_index(self, column, table):
        command = ("DROP INDEX idx_%s ON %s;" % (column, table))
        try:
            self.conn = pyodbc.connect(self.connection_string)
            self.cur = self.conn.cursor()
            self.cur.execute(command)
            self.conn.commit()
            self.cur.close()
            self.conn.close()
            print('Dropped index on (%s) %s' % (table, column))
        except pyodbc.Error as ex:
            print("Didn't drop index on %s, error %s" % (column, ex))


    def create_index(self, column, table):
        command = "CREATE INDEX idx_%s ON %s (%s);" % (column, table, column)
        try:
            self.conn = pyodbc.connect(self.connection_string)
            self.cur = self.conn.cursor()
            self.cur.execute(command)
            self.conn.commit()
            self.cur.close()
            self.conn.close()
            print('Created index on (%s) %s' % (table, column))
        except pyodbc.Error as ex:
            print("Didn't create index on %s, error %s" % (column, ex))


    """
        State-related methods
    """
    def get_table_columns(self, table):
        self.conn = pyodbc.connect(self.connection_string)
        self.cur = self.conn.cursor()
        self.cur.execute('SHOW COLUMNS FROM %s;' % table)
        table_columns = list()
        for row in self.cur.fetchall():
            if row[0] not in self.tables[table]:
                table_columns.append(row[0])
        self.conn.commit()
        self.cur.close()
        self.conn.close()
        return table_columns

    def get_table_indexed_columns(self, table):
        self.conn = pyodbc.connect(self.connection_string)
        self.cur = self.conn.cursor()
        self.cur.execute('SHOW INDEXES FROM %s;' % table)
        table_indexes = list()
        for index in self.cur.fetchall():
            if index[2] not in self.tables[table]:
                table_indexes.append(index[4])
        self.conn.commit()
        self.cur.close()
        self.conn.close()
        return table_indexes

    def get_indexes_map(self):
        indexes_map = dict()
        for table in self.tables.keys():
            indexes_map[table] = dict()
            indexed_columns = self.get_table_indexed_columns(table)
            table_columns = self.get_table_columns(table)
            for column in table_columns:
                indexes_map[table][column] = 0
                for index in indexed_columns:
                    if column == index:
                        indexes_map[table][column] = 1

        return indexes_map

    
    """
        Environment-related methods
    """
    def reset_indexes(self):
        # FETCH INDEX NAMES
        self.conn = pyodbc.connect(self.connection_string)
        self.cur = self.conn.cursor()

        for table in self.tables.keys():
            self.cur.execute('SHOW INDEXES FROM %s;' % table)
            index_names = list()

            for index in self.cur.fetchall():
                index_names.append(index[2])

            for index in index_names:
                if "idx_" in index:
                    self.cur.execute("DROP INDEX %s ON %s;" % (index, table))
        
        self.conn.commit()
        self.cur.close()
        self.conn.close()
        
        return True


if __name__ == "__main__":
    db = Database()

    # db.create_index('c_phone', 'customer')
    # db.create_index('l_commitdate', 'lineitem')
    # db.create_index('n_name', 'nation')
    # db.create_index('o_clerk', 'orders')
    # db.create_index('p_brand', 'part')
    # db.create_index('s_phone', 'supplier')
    # db.create_index('ps_availqty', 'partsupp')

    if (db.reset_indexes()):
        print("YEAHH!")

    indexes_map = db.get_indexes_map()

    pprint.pprint(indexes_map)