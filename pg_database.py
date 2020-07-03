import psycopg2
import json


class PG_Database():
    def __init__(self, hypo=True, analyze=False):
        # Get credentials
        with open('data/db_credentials_pg.json', 'r') as f:
            self.credentials = json.load(f)

        # Connect to database
        try:
            self.conn = psycopg2.connect(user = self.credentials['user'],
                                         password = self.credentials['password'],
                                         host = self.credentials['host'],
                                         port = self.credentials['port'],
                                         database = self.credentials['database'])
            # Set to autocommit transactions
            self.conn.autocommit = True
        except psycopg2.Error as err: 
            print("ERROR: {}".format(err))

        # Create hypothetical indexes
        self.hypo = hypo
        # Analyze tables before getting cost
        self.analyze = analyze
        # Get tables and indexes
        self.tables = self.get_tables()

    def get_query_cost(self, query):
        # Get cost
        command = "EXPLAIN (FORMAT JSON) {}".format(query)
        output = self.execute_fetchall(command)
        explain = output[0][0][0]
        cost = float(explain['Plan']['Total Cost'])
        return cost
    
    def get_query_use(self, query, column):
        # Get explain plan
        command = "EXPLAIN {}".format(query)
        output = self.execute_fetchall(command)
        # Verify
        for row in output:
            if 'pkey' in row[0]: continue
            if 'fkey' in row[0]: continue
            if 'Index Scan on' in row[0] and column in row[0]: return 1
            elif 'Index Scan using' in row[0] and column in row[0]: return 1
        return 0

    def get_tables(self):
        # Fetch constraints
        command = "SELECT kcu.column_name FROM information_schema.table_constraints tco JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tco.constraint_name AND kcu.constraint_schema = tco.constraint_schema AND kcu.constraint_name = tco.constraint_name WHERE tco.constraint_type = 'PRIMARY KEY' OR tco.constraint_type = 'FOREIGN KEY' ORDER BY kcu.table_name;"
        output = self.execute_fetchall(command)
        constraints = [row[0] for row in output]

        # Fetch all tables and columns
        command = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public' AND is_updatable='YES';"
        output = self.execute_fetchall(command)
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
        if self.hypo:
            command = "SELECT * FROM hypopg_list_indexes();"
            output = self.execute_fetchall(command)
            index_names = list(set([row[1] for row in output]))
            indexes = dict()
            for table in self.tables.keys():
                for column in self.tables[table]:
                    if column in str(index_names):
                        indexes[column] = 1
                    else:
                        indexes[column] = 0
            return indexes
        else:
            command = "SELECT t.relname AS table_name, i.relname AS index_name, a.attname AS column_name FROM pg_class t, pg_class i, pg_index ix, pg_indexes ixs, pg_attribute a WHERE t.oid = ix.indrelid AND i.oid = ix.indexrelid AND a.attrelid = t.oid AND a.attnum = ANY(ix.indkey) AND ixs.schemaname = 'public' AND i.relname = ixs.indexname ORDER BY t.relname, i.relname;"
            output = self.execute_fetchall(command)
            index_names = list(set([row[1] for row in output]))
            indexes = dict()
            for table in self.tables.keys():
                for column in self.tables[table]:
                    if column in str(index_names):
                        indexes[column] = 1
                    else:
                        indexes[column] = 0
            return indexes

    def drop_index(self, table, column, verbose=False):
        if self.hypo:
            # Get all indexes
            command = "SELECT * FROM hypopg_list_indexes();"
            indexes = self.execute_fetchall(command)
            # Iterate indexes and check column match
            for index in indexes:
                if table == index[3] and column in index[1]:
                    command = "SELECT * FROM hypopg_drop_index(%s);" % (index[0])
                    self.execute(command, verbose)
        else:
            if 'smartix_' in column or '_idx' in column:
                command = ("DROP INDEX %s;" % (column))
            else:
                command = ("DROP INDEX smartix_%s;" % (column))
            self.execute(command, verbose)

    def create_index(self, table, column, verbose=False):
        if self.hypo:
            command = "SELECT * FROM hypopg_create_index('CREATE INDEX smartix_%s ON %s (%s)');" % (column, table, column)
            self.execute(command, verbose)
        else:
            command = "CREATE INDEX smartix_%s ON %s (%s);" % (column, table, column)
            self.execute(command, verbose)
        if self.analyze:
            for table in self.tables.keys():
                command = "ANALYZE {};".format(table)
                self.execute(command)

    def reset_indexes(self):
        if self.hypo:
            command = "SELECT * FROM hypopg_reset();"
            self.execute(command)
        else:
            command = "SELECT t.relname AS table_name, i.relname AS index_name, a.attname AS column_name FROM pg_class t, pg_class i, pg_index ix, pg_indexes ixs, pg_attribute a WHERE t.oid = ix.indrelid AND i.oid = ix.indexrelid AND a.attrelid = t.oid AND a.attnum = ANY(ix.indkey) AND ixs.schemaname = 'public' AND i.relname = ixs.indexname ORDER BY t.relname, i.relname;"
            output = self.execute_fetchall(command)
            print(output)
            for index in output:
                index_name = index[1]
                if "smartix_" in index_name or "_idx" in index_name:
                    print("Drop", index_name)
                    self.drop_index(None, index_name)

    def close_connection(self):
        try:
            self.conn.close()
            return True
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))
            return False

    def execute(self, command, verbose=False):
        try:
            cur = self.conn.cursor()
            cur.execute(command)
            cur.close()
            if verbose: print('OK: {}'.format(command))
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))
    
    def execute_fetchall(self, command, verbose=False):
        try:
            cur = self.conn.cursor()
            cur.execute(command)
            output = cur.fetchall()
            cur.close()
            if verbose: print('OK: {}'.format(command))
            return output
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint


    #################################################


    db = PG_Database(hypo=True)

    # Get workload
    with open('data/workload/tpch_shift.sql', 'r') as f:
        data = f.read()
    workload = data.split('\n')

    db.create_index('lineitem', 'l_shipdate')
    db.create_index('part', 'p_size')
    db.create_index('part', 'p_container')
    db.create_index('part', 'p_brand')
    db.create_index('orders', 'o_orderdate')
    db.create_index('customer', 'c_acctbal')

    # Count uses
    for col in ['l_shipdate', 'p_size', 'p_container', 'p_brand', 'o_orderdate', 'c_acctbal']:
        total_count = 0
        for i, q in enumerate(workload):
            count = db.get_query_use(q, col)
            print(col, i, count)
            total_count += count
        print("Total count:", total_count, col)

    db.reset_indexes()

    db.close_connection()
