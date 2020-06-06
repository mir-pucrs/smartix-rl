import pyodbc
import json


class MySQL_Database():
    def __init__(self):
        # Get database credentials
        with open('data/db_credentials.json', 'r') as f:
            credentials = f.read() 
        self.credentials = json.loads(credentials)

        # Database connection string
        self.connection_string = ("DRIVER={%s};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s" % (self.credentials['DRIVER'], self.credentials['SERVER'], self.credentials['DATABASE'], self.credentials['UID'], self.credentials['PWD']))

        # Get tables and indexes
        self.tables = self.get_tables()

    def get_query_cost(self, query):
        # Analyze tables
        for table in self.tables.keys():
            command = "ANALYZE TABLE {};".format(table)
            self.execute(command, verbose=False)
        # Get query cost
        command = "EXPLAIN FORMAT=JSON {}".format(query)
        output = self.execute_fetchall(command, verbose=False)
        explain = json.loads(output[0][0])
        cost = float(explain['query_block']['cost_info']['query_cost'])
        return cost

    def get_tables(self):
        # Fetch constraints
        command = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.key_column_usage WHERE TABLE_SCHEMA='{}';".format(self.credentials['DATABASE'])
        output = self.execute_fetchall(command, verbose=False)
        constraints = [row[0] for row in output]
        
        # Fetch all tables and columns
        command = "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.columns WHERE TABLE_SCHEMA='{}' ORDER BY TABLE_NAME, ORDINAL_POSITION;".format(self.credentials['DATABASE'])
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

    def get_indexes(self):
        indexes = dict()
        for table in self.tables.keys():
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
            command = ("DROP INDEX %s ON %s;" % (column, table))
        else:
            command = ("DROP INDEX smartix_%s ON %s;" % (column, table))
        self.execute(command)

    def create_index(self, table, column):
        command = "CREATE INDEX smartix_%s ON %s (%s);" % (column, table, column)
        self.execute(command)

    def reset_indexes(self):
        for table in self.tables.keys():
            # Get indexes for table
            command = "SHOW INDEXES FROM {};".format(table)
            output = self.execute_fetchall(command, verbose=False)

            index_names = list()
            for index in output:
                index_names.append(index[2])

            for index in index_names:
                if "smartix_" in index:
                    self.drop_index(table, index)
                    command = "DROP INDEX %s ON %s;" % (index, table)
        return True

    def close_connection(self):
        try:
            self.cur.close()
            self.conn.close()
            return True
        except pyodbc.Error as ex: 
            print('ERROR: {}'.format(ex))
            return False

    def execute(self, command, verbose=True):
        try:
            self.conn = pyodbc.connect(self.connection_string, autocommit=True)
            self.cur = self.conn.cursor()
            self.cur.execute(command)
            self.cur.close()
            self.conn.close()
            if verbose: print('OK: {}'.format(command))
        except pyodbc.Error as ex:
            print('ERROR: {}'.format(ex))
    
    def execute_fetchall(self, command, verbose=True):
        try:
            self.conn = pyodbc.connect(self.connection_string, autocommit=True)
            self.cur = self.conn.cursor()
            self.cur.execute(command)
            output = self.cur.fetchall()
            self.cur.close()
            self.conn.close()
            if verbose: print('OK: {}'.format(command))
            return output
        except pyodbc.Error as ex:
            print('ERROR: {}'.format(ex))


if __name__ == "__main__":
    from pprint import pprint
    db = Database()
    db.reset_indexes()

    pprint(db.tables)
    