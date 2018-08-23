import os
import mysql.connector
import subprocess
import threading


class Query:

    def __init__(self):
        self.res_time_execution = {}


cnx = mysql.connector.connect(host='sap-server.local',
                              user='dbuser', passwd='dbuser', db='tpch')


cursor = cnx.cursor(buffered=True)
cursor1 = cnx.cursor(buffered=True)
cursor2 = cnx.cursor(buffered=True)
cursor3 = cnx.cursor(buffered=True)
cursor4 = cnx.cursor(buffered=True)
cursor5 = cnx.cursor(buffered=True)


def generate_dbgen_file():
    os.chdir('TPCH/dbgen')
    subprocess.call('./generatedbgen.sh')


def drop_caches():
    os.chdir('path-to-file')
    os.popen('sudo -S ./drop_caches.sh', 'w').write('your-password-admin')


set_profiling = "SET profiling=1"

query = ("SELECT l_returnflag, l_linestatus, "
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

# query2 = ("SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment "
#              "FROM part, supplier, partsupp, nation, region "
#              "WHERE p_partkey = ps_partkey "
#              "and s_suppkey = ps_suppkey "
#             "and p_size = 15 "
#             "and p_type like '%BRASS' "
#             "and s_nationkey = n_nationkey "
#             "and n_regionkey = r_regionkey "
#             "and r_name = 'EUROPE' "
#             "and ps_supplycost = (SELECT MIN(ps_supplycost) "
#             "from partsupp, supplier, nation, region "
#             "where p_partkey = ps_partkey and s_suppkey = ps_suppkey "
#             "and s_nationkey = n_nationkey "
#             "and n_regionkey = r_regionkey "
#             "and r_name = 'EUROPE') "
#             "order by s_acctbal desc, n_name, s_name, p_partkey" )

# query3 = ("SELECT l_orderkey, "
# 	        "sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority "
#             "FROM customer, orders, lineitem "
#             "WHERE c_mktsegment = 'BUILDING' "
# 	        "and c_custkey = o_custkey "
#             "and l_orderkey = o_orderkey "
#             "and o_orderdate < date '1995-03-15' "
#             "and l_shipdate > date '1995-03-15' "
# "group by l_orderkey, o_orderdate, o_shippriority "
# "order by revenue desc, o_orderdate " )


show_profiles = "SHOW PROFILES"


def refresh_function():
    #implementar depois que gerar dados com o dbgen
    generate_dbgen_file()
    load_data = "LOAD DATA INFILE 'orders_update.txt' INTO TABLE orders;"


def comp_time_execution():
    global res_time_execution
    res_time_execution = cursor5.fetchall()
    d = dict()
    for row in res_time_execution:
        d[row[0]] = row[1]
    res_time_execution = d
    return res_time_execution


def run_functions():
    cursor.execute(set_profiling)
    cursor1.execute(refresh_function())
    drop_caches()
    t1 = threading.Thread(target=cursor2.execute(query))
    t2 = threading.Thread(target=cursor3.execute(query2))
    #t3 = threading.Thread(target=cursor4.execute(query3))
    t1.start()
    t2.start()
    #t3.start()
    # cursor2.execute(query)
    # cursor3.execute(query2)
    # cursor4.execute(query3)
    cursor5.execute(show_profiles)
    cnx.close()


# O QUE FALTA:
# 1.IMPLEMENTAR FUNCTION QUE ENCHE BUFFER
# 2. INCLUIR QUERIES DO POWER SIZE
# 3. RL ALGORITMO - AGENTE



