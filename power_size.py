import os
import subprocess
import mysql.connector


class Power:

    def __init__(self):
        self.time_power_size = {}


cnx1 = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')

cursor = cnx1.cursor(buffered=True)
cursor1 = cnx1.cursor(buffered=True)
cursor2 = cnx1.cursor(buffered=True)
cursor3 = cnx1.cursor(buffered=True)
cursor4 = cnx1.cursor(buffered=True)
cursor5 = cnx1.cursor(buffered=True)
cursor6 = cnx1.cursor(buffered=True)
cursor7 = cnx1.cursor(buffered=True)
cursor8 = cnx1.cursor(buffered=True)
cursor9 = cnx1.cursor(buffered=True)
cursor10 = cnx1.cursor(buffered=True)
cursor11 = cnx1.cursor(buffered=True)


def generate_dbgen_file():
    os.chdir('TPCH/dbgen')
    subprocess.call('./generatedbgen.sh')


set_profiling = "SET profiling=1"

query1 = ("SELECT l_returnflag, l_linestatus, "
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


query2 = ("select sum(l_extendedprice * l_discount) as revenue "
			"from lineitem where l_shipdate >= date '1994-01-01' "
			"and l_shipdate < date '1994-01-01' + interval '1' year "
			"and l_discount between 0.06 - 0.01 and 0.06 + 0.01 "
			"and l_quantity < 24;" )


# QUERIES FOR INSERTIONS
query3 = ("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/orders.tbl.u1' into table orders fields terminated by '|' lines terminated by '\n';")

query4 = ("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/lineitem.tbl.u1' into table lineitem fields terminated by '|' lines terminated by '\n';")

# QUERIES FOR DELETIONS
query5 = ("create temporary table t_lineItem (t_orderkey bigint(20) NOT NULL);")

query6 = ("load data local infile '/home/priscillaneuhaus/PycharmProjects/sap_project/TPCH/dbgen/delete.2' into table t_lineItem fields terminated by '|' lines terminated by '\n';")

query7 = ("delete from lineitem where l_orderkey in (select t_orderkey from t_lineItem);")

query8 = ("delete from orders where o_orderkey in (select t_orderkey from t_lineItem);")

# Delete temporary table
query9 = ("drop table t_lineItem;")

# New indexing options
#  query10 = ("create index t_lineitem on lineitem (l_shipdate,l_receiptdate);")

# Get the execution time
show_profiles = "SHOW PROFILES"


def refresh_function():
    generate_dbgen_file()


def comp_time_power_size():
    total_time = 1.0
    count = 0
    global time_power_size
    time_power_size = cursor11.fetchall()
    d = dict()
    for row in time_power_size:
        d[row[0]] = row[1]
    del d[5], d[6], d[9], d[10], d[11]
    prod = (d[1] * d[2]) + (d[3] + d[4]) * (d[7] + d[8])
    d['count'] = 6
    d['total_time'] = prod
    return d


def run_functions():
    cursor.execute(set_profiling)
    cursor1.execute(refresh_function())
    # drop_caches()
    cursor2.execute(query1)
    cursor3.execute(query2)

    # Execute queries for insertions
    cursor4.execute(query3)
    cursor5.execute(query4)
    cnx1.commit()

    # Execute queries for deletions
    cursor6.execute(query5)
    cursor7.execute(query6)
    cursor8.execute(query7)
    cursor9.execute(query8)
    cnx1.commit()

    cursor10.execute(query9)
    cursor11.execute(show_profiles)
    cnx1.close()

