import os
import subprocess
import mysql.connector
import math
import time

from multiprocessing import Process, Queue # FOR PARALLEL EXECUTION

from scipy import stats # FOR GEOMETRIC MEAN



refresh_stream_number = 250 # FOR KEEPING TRACK OF REFRESH FILES TO BE USED

def load_refresh_stream_data():
    global refresh_stream_number
    print("!!! Current refresh stream number: ", refresh_stream_number)
    conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='tpch')
    cursor = conn.cursor()
    delete = "load data local infile '/home/gabriel/sap/tpch-qsrf/RefreshFiles/delete.{}' into table rfdelete fields terminated by '|' lines terminated by '\n';".format(refresh_stream_number)
    orders = "load data local infile '/home/gabriel/sap/tpch-qsrf/RefreshFiles/orders.tbl.u{}' into table orders_temp fields terminated by '|' lines terminated by '\n';".format(refresh_stream_number)
    lineitem = "load data local infile '/home/gabriel/sap/tpch-qsrf/RefreshFiles/lineitem.tbl.u{}' into table lineitem_temp fields terminated by '|' lines terminated by '\n';".format(refresh_stream_number)
    cursor.execute(delete)
    cursor.execute(orders)
    cursor.execute(lineitem)
    conn.close()
    refresh_stream_number += 1

def insert_refresh_function():
    conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='tpch')
    cursor = conn.cursor()
    cursor.callproc("INSERT_REFRESH_FUNCTION")
    conn.close()

def delete_refresh_function():
    conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='tpch')
    cursor = conn.cursor()
    cursor.callproc("DELETE_REFRESH_FUNCTION")
    conn.close()

def run_refresh_stream():
    print("\n!!! Loading refresh stream data... (RS1)")
    load_refresh_stream_data()
    print("!!! Start insert refresh function (RS1)")
    insert_refresh_function()
    print("!!! Start delete refresh function (RS1)")
    delete_refresh_function()

    print("\n!!! Loading refresh stream data... (RS2)")
    load_refresh_stream_data()
    print("!!! Start insert refresh function (RS2)")
    insert_refresh_function()
    print("!!! Start delete refresh function (RS2)")
    delete_refresh_function()

    print("!!! Refresh streams finished")



def run_query_stream(results_queue):
    # START DB CONNECTION
    conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='tpch')
    cursor = conn.cursor()

    # SET PROFILING
    cursor.execute("SET PROFILING_HISTORY_SIZE = 22")
    cursor.execute("SET PROFILING = 1")

    # QUERY STREAM
    cursor.callproc("QUERY_STREAM")

    # SHOW PROFILES
    cursor.execute("SHOW PROFILES")

    # CLOSE DB CONNECTION
    conn.close()

    # TRANSFORM PROFILES INTO DICT OF (QUERY NUM: DURATION)
    profiles = dict()
    for row in cursor.fetchall():
        profiles[row[0]] = row[1]

    # RETURN PROFILES RESULT
    results_queue.put(profiles)



def run_power_test():

    # LOAD REFRESH STREAM DATA
    print("\n!!! Loading refresh stream data... (RS0)")
    load_refresh_stream_data()

    # INSERT REFRESH FUNCTION
    print("\n!!! Start insert refresh function (RS0)")
    start_time_insert_rf = time.time()
    insert_refresh_function()
    elapsed_time_insert_rf = time.time() - start_time_insert_rf
    print("\n!!! Insert RF elapsed time: ", elapsed_time_insert_rf)

    # QUEUE TO GET RESULTS FROM QUERY STREAMS
    profiles_queue = Queue()

    # DECLARE PROCESS WHICH WILL RUN QUERY STREAM
    print("\n!!! Declare QS")
    qs = Process(target=run_query_stream, args=(profiles_queue, ))

    # START QUERY STREAM PROCESS
    print("\n!!! Start QS")
    qs.start()

    # GET RESULTS FROM QUEUE
    print("\n!!! Get queue QS")
    power_test_profiles = profiles_queue.get()

    # JOIN QUERY STREAM PROCESS
    print("\n!!! Join QS")
    qs.join()
    
    # DELETE REFRESH FUNCTION
    print("\n!!! Start delete refresh function (RS0)")
    start_time_delete_rf = time.time()
    delete_refresh_function()
    elapsed_time_delete_rf = time.time() - start_time_delete_rf
    print("\n!!! Delete RF elapsed time: ", elapsed_time_delete_rf)

    print("\n!!! QS elapsed time: ", sum(power_test_profiles.values()))

    execution_times = list(power_test_profiles.values())
    execution_times.append(elapsed_time_insert_rf)
    execution_times.append(elapsed_time_delete_rf)
    print("\nexecution_times: ", execution_times)

    geo_mean = stats.gmean(execution_times)
    print("\nGeo. mean: ", geo_mean)

    return (3600/geo_mean) * 1



def run_throughput_test():

    # DECLARE PROCESSES
    print("!!! Declare QS1")
    qs1 = Process(target=run_query_stream, args=(Queue(), ))
    print("!!! Declare QS2")
    qs2 = Process(target=run_query_stream, args=(Queue(), ))
    print("!!! Declare RS")
    rs = Process(target=run_refresh_stream)

    # START TIMING EXECUTION
    start_time = time.time()

    # START PROCESSES
    print("!!! Start QS1")
    qs1.start()
    print("!!! Start QS2")
    qs2.start()
    print("!!! Start RS")
    rs.start()

    # JOIN PROCESSES
    print("!!! Join QS1")
    qs1.join()
    print("!!! Join QS2")
    qs2.join()
    print("!!! Join RS")
    rs.join()

    # FINISH TIMING EXECUTION
    elapsed_time = time.time() - start_time

    print("\n!!! Throughput test elapsed time: ", elapsed_time)

    return ((2 * 22) / elapsed_time) * 3600 * 1



def main():
    # RUN POWER@SIZE
    print("\n!!! START POWER TEST !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!\n")
    power_size = run_power_test()

    # RUN THROUGHPUT
    print("\n!!! START THROUGHPUT !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!\n")
    throughput = run_throughput_test()

    # CALCULATE QphH (COmposite Query-Per-Hour)
    QphH = math.sqrt(power_size * throughput)

    print("\n!!! RESULTED METRICS !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!\n")
    print("Power@Size = ", power_size)
    print("Throughput@Size = ", throughput)
    print("QphH@Size = ", QphH)

    global refresh_stream_number
    print("\nNext refresh stream number: ", refresh_stream_number)

if __name__ == '__main__':
    main()