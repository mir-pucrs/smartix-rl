import os
import mysql.connector
import subprocess


class Throughput:

    def __init__(self):
        self.res_time_execution = {}

# def generate_dbgen_file():
#     os.chdir('TPCH/dbgen')
#     subprocess.call('./generatedbgen.sh')


# def drop_caches():
#     os.chdir('path-to-file')
#     os.popen('sudo -S ./drop_caches.sh', 'w').write('your-password-admin')

    def execute_queries(self):
        cnx = mysql.connector.connect(host='127.0.0.1',
                                           user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute("SET profiling=1")
        cursor.execute("SELECT l_returnflag, l_linestatus, "
                            "sum(l_quantity) as sum_qty, "
                            "sum(l_extendedprice) as sum_base_price, "
                            "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
                            "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
                            "avg(l_quantity) as avg_qty, "
                            "avg(l_extendedprice) as avg_price, "
                            "avg(l_discount) as avg_disc, "
                            "count(*) as count_order "
                            "from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day "
                            "group by l_returnflag, l_linestatus "
                            "order by l_returnflag, l_linestatus;")
        cursor.execute("select sum(l_extendedprice * l_discount) as revenue "
                            "from lineitem where l_shipdate >= date '1994-01-01' "
                            "and l_shipdate < date '1994-01-01' + interval '1' year "
                            "and l_discount between 0.06 - 0.01 and 0.06 + 0.01 "
                            "and l_quantity < 24;")
        cursor.execute("SHOW PROFILES")
        result_time = cursor.fetchall()
        cnx.close()
        return result_time

    def comp_time_execution(self):
        global res_time_execution
        res_time_execution = self.execute_queries()
        d = dict()
        for row in res_time_execution:
            d[row[0]] = row[1]
        res_time_execution = d
        return res_time_execution

    def run_functions(self):
        self.comp_time_execution()

# TO DO:
# 1.IMPLEMENT FUNCTION THAT FULL BUFFER
# 2. ADD QUERIES FROM POWER SIZE
# 3. RL ALGORITHM - AGENT



