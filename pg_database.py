import psycopg2
import json


class PGDatabase():
    def __init__(self):
        self.connection = psycopg2.connect(user = "gabriel",
                                  password = "gabriel",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "tpch")

        self.dbname = 'tpch'

        # Get tables and indexes
        self.tables = self.get_tables()

    def get_query_cost(self, query):
        # # Analyze tables
        # for table in self.tables.keys():
        #     command = "ANALYZE {};".format(table)
        #     self.execute(command, verbose=False)
        # Get query cost
        command = "EXPLAIN (FORMAT JSON) {}".format(query)
        output = self.execute_fetchall(command, verbose=False)
        explain = output[0][0][0]
        cost = float(explain['Plan']['Total Cost'])
        return cost

    def get_tables(self):
        # Fetch constraints
        command = "SELECT kcu.column_name FROM information_schema.table_constraints tco JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tco.constraint_name AND kcu.constraint_schema = tco.constraint_schema AND kcu.constraint_name = tco.constraint_name WHERE tco.constraint_type = 'PRIMARY KEY' OR tco.constraint_type = 'FOREIGN KEY' ORDER BY kcu.table_name;"
        output = self.execute_fetchall(command, verbose=False)
        constraints = [row[0] for row in output]

        # Fetch all tables and columns
        command = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public';"
        output = self.execute_fetchall(command, verbose=False)
        tables = dict()
        for row in output:
            table, column = row
            if column not in constraints:
                if table not in tables.keys():
                    tables[table] = list()
                tables[table].append(column)
        
        # Return dict with valid columns for indexing
        return tables


    ######################### FIXXXXXX COMMAND BELOW
    def get_indexes(self):
        indexes = dict()
        for table in self.tables.keys():
            ######################### FIXXXXXX COMMAND BELOW
            command = 'SHOW INDEX FROM {}'.format(table)
            output = self.execute_fetchall(command, verbose=False)
            table_indexes = list(set([row[4] for row in output]))
            for column in self.tables[table]:
                if column in table_indexes:
                    indexes[column] = 1
                else:
                    indexes[column] = 0
        return indexes

    def drop_index(self, table, column):
        if 'smartix_' in column:
            command = ("DROP INDEX %s;" % (column))
        else:
            command = ("DROP INDEX smartix_%s;" % (column))
        self.execute(command)

    def create_index(self, table, column):
        command = "CREATE INDEX smartix_%s ON %s (%s);" % (column, table, column)
        self.execute(command)


    ######################### FIXXXXXX COMMAND BELOW
    def reset_indexes(self):
        for table in self.tables.keys():
            # Get indexes for table
            ######################### FIXXXXXX COMMAND BELOW
            command = "SHOW INDEXES FROM {};".format(table)
            output = self.execute_fetchall(command, verbose=False)

            index_names = list()
            for index in output:
                index_names.append(index[2])

            for index in index_names:
                if "smartix_" in index:
                    self.drop_index(table, index)
                    command = "DROP INDEX %s;" % (index)
        return True

    def close_connection(self):
        try:
            self.connection.close()
            return True
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))
            return False

    def execute(self, command, verbose=True):
        try:
            cur = self.connection.cursor()
            cur.execute(command)
            cur.close()
            if verbose: print('OK: {}'.format(command))
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))
    
    def execute_fetchall(self, command, verbose=True):
        try:
            cur = self.connection.cursor()
            cur.execute(command)
            output = cur.fetchall()
            # print(output)
            cur.close()
            if verbose: print('OK: {}'.format(command))
            return output
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))


if __name__ == "__main__":
    from pprint import pprint
    db = PGDatabase()

    # query = "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price, sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc, count(*) AS count_order FROM LINEITEM WHERE l_shipdate <= date '1994-7-17' - interval '108' day GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;"

    # cost = db.get_query_cost(query)
    pprint(db.tables)