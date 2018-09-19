import os
import subprocess
import mysql.connector
import math
import time

from multiprocessing import Process, Queue # FOR PARALLEL EXECUTION

from scipy import stats # FOR GEOMETRIC MEAN



'''
    1. Colocar refresh funcs como stored procedures
        Fica + fácil de medir o tempo
        Mas e a questão de ?coerencia?
    2. Clear Caches?!?!?!?!
    3. Loop de performance tests sugerindo diferentes índices
        Armazenar métricas resultantes e comparar
'''



def insert_refresh_function():
    os.chdir('/home/gabriel/sap/tpch-qsrf')
    subprocess.call('./03_RF_INSERT.sh')



def delete_refresh_function():
    os.chdir('/home/gabriel/sap/tpch-qsrf')
    subprocess.call('./04_RF_DELETE.sh')



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

    # INSERT REFRESH FUNCTION
    # print("!!!---!!! START INSERT REFRESH FUNCTION")
    # insert_refresh_function()
    # print("!!!---!!! FINISH INSERT REFRESH FUNCTION")

    # QUEUE TO GET RESULTS FROM QUERY STREAMS
    profiles_queue = Queue()

    # DECLARE PROCESS WHICH WILL RUN QUERY STREAM
    qs = Process(target=run_query_stream, args=(profiles_queue, ))
    
    # START QUERY STREAM PROCESS
    qs.start()

    # GET RESULTS FROM QUEUE
    power_test_profiles = profiles_queue.get()

    # JOIN QUERY STREAM PROCESS
    qs.join()
    
    # DELETE REFRESH FUNCTION
    # print("!!!---!!! START DELETE REFRESH FUNCTION")
    # delete_refresh_function()
    # print("!!!---!!! FINISH DELETE REFRESH FUNCTION")

    print("\npower_test_profiles:\n\n", power_test_profiles, "\n")

    geo_mean = stats.gmean(list(power_test_profiles.values()))
    power_size = (3600/geo_mean) * 1

    print("Total: ", sum(power_test_profiles.values()))
    print("Geo. mean: ", geo_mean)
    print("Size: ", len(power_test_profiles))

    return power_size



def run_throughput_test():

    profiles_queue = Queue()
    streams_profiles = []
    query_streams = []
    
    # START TIMING EXECUTION
    start_time = time.time()

    for i in range(0, 2):
        print("\n!!! Start QS", i)
        # rf1 = Process(target=insert_refresh_function, args=(, ))
        qs = Process(target=run_query_stream, args=(profiles_queue, ))
        # rf2 = Process(target=delete_refresh_function, args=(, ))
        query_streams.append(qs)
        qs.start()

    for qs in query_streams:
        print("\n!!! Get from", qs)
        streams_profiles.append(profiles_queue.get())

    for qs in query_streams:
        print("\n!!! Join", qs)
        qs.join()

    # FINISH TIMING EXECUTION
    elapsed_time = time.time() - start_time

    print("\n!!! Print Results\n")
    print(elapsed_time)
    # print(streams_profiles)

    # execution_time = 0
    # for row in streams_profiles:
    #     execution_time += sum(row.values())

    # print("Ts: ", execution_time)

    return ((2 * 22) / elapsed_time) * 3600 * 1



def main():
    # RUN POWER@SIZE
    # power_size = run_power_test()
    # print("Power@Size = ", power_size)

    # RUN THROUGHPUT
    throughput = run_throughput_test()
    print("Throughput@Size = ", throughput)

    # CALCULATE QphH (Composite Query-Per-Hour)
    # QphH = math.sqrt(power_size * throughput)
    # print("QphH@Size = ", QphH)



if __name__ == '__main__':
    main()