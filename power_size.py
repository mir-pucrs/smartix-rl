import os
import subprocess
import mysql.connector


class PowerSize:

    def __init__(self):
        self.res_time_execution = {}

    def execute_queries(self):
        cnx = mysql.connector.connect(host='127.0.0.1',
                                           user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        cursor.execute("SET profiling=1")
        cursor.execute(self.refresh_function())
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
        cursor.execute("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/orders.tbl.u1' into table orders fields terminated by '|' lines terminated by '\n';")
        cursor.execute("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/lineitem.tbl.u1' into table lineitem fields terminated by '|' lines terminated by '\n';")
        cnx.commit()
        cursor.execute("create temporary table t_lineItem (t_orderkey bigint(20) NOT NULL);")
        cursor.execute("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/delete.2' into table t_lineItem fields terminated by '|' lines terminated by '\n';")
        cursor.execute("delete from lineitem where l_orderkey in (select t_orderkey from t_lineItem);")
        cursor.execute("delete from orders where o_orderkey in (select t_orderkey from t_lineItem);")
        cursor.execute("drop table t_lineItem;")
        cnx.commit()
        cursor.execute("SHOW PROFILES")
        result_time = cursor.fetchall()
        cnx.close()
        return result_time

    def generate_dbgen_file(self):
        os.chdir('..')
        p = subprocess.call('ls')
        print(p)
        # os.chdir('TPCH/dbgen')
        # subprocess.call('./generatedbgen.sh')

    def refresh_function(self):
        self.generate_dbgen_file()

    def comp_time_power_size(self):
        global time_power_size
        time_power_size = self.execute_queries()
        d = dict()
        for row in time_power_size:
            d[row[0]] = row[1]
        del d[5], d[6], d[9], d[10], d[11]
        prod = (d[1] * d[2]) + (d[3] + d[4]) * (d[7] + d[8])
        d['count'] = 6
        d['total_time'] = prod
        return d

    def run_functions(self):
        self.comp_time_power_size()

