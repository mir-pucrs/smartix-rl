import mysql.connector
import math
import time

from multiprocessing import Process, Queue # FOR PARALLEL EXECUTION
from scipy import stats # FOR GEOMETRIC MEAN



'''
    Global configuration
'''
DB_CONFIG = {'user': 'root', 'password': 'root', 'host': 'localhost', 'database': 'tpch'}
REFRESH_FILES_PATH = '/home/gabriel/sap/tpch-qsrf/RefreshFiles'



'''
    SCALE_FACTOR    1   10  30  100
    NUM_STREAMS     2   3   4   5
'''
SCALE_FACTOR = 1
NUM_STREAMS = 2



'''
    Keeps track of refresh files order
    by storing sequence number in a .txt file
'''
def get_refresh_stream_number():
    global refresh_stream_number
    rsn = open("refresh_stream_number.txt", "r")
    refresh_stream_number = int(rsn.read())
    rsn.close()
def set_refresh_stream_number():
    global refresh_stream_number
    rsn = open("refresh_stream_number.txt", "w")
    rsn.write("%d" % refresh_stream_number)
    rsn.close()



'''
    Loads data from refresh files to temporary tables in the database
'''
def load_refresh_stream_data():
    global refresh_stream_number
    print("*** Load refresh stream number:", refresh_stream_number)

    # STRINGS TO BE EXECUTED BY CURSOR
    delete = "load data local infile '{}/delete.{}' into table rfdelete fields terminated by '|' lines terminated by '\n';".format(REFRESH_FILES_PATH, refresh_stream_number)
    orders = "load data local infile '{}/orders.tbl.u{}' into table orders_temp fields terminated by '|' lines terminated by '\n';".format(REFRESH_FILES_PATH, refresh_stream_number)
    lineitem = "load data local infile '{}/lineitem.tbl.u{}' into table lineitem_temp fields terminated by '|' lines terminated by '\n';".format(REFRESH_FILES_PATH, refresh_stream_number)

    # OPEN DB CONNECTION, EXECUTE LOADS AND CLOSE CONNECTION
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(delete)
    cursor.execute(orders)
    cursor.execute(lineitem)
    conn.close()
    
    # INCREMENT REFRESH STREAM NUMBER FOR NEXT STREAM
    refresh_stream_number += 1



'''
    Refresh functions calling respective procedures in the DB
    Each function returns its duration
'''
def insert_refresh_function():
    # OPENS CONNECTION, EXECUTES PROCEDURE AND CLOSES CONNECTION
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SET PROFILING = 1")
    cursor.callproc("INSERT_REFRESH_FUNCTION")
    cursor.execute("SHOW PROFILES")
    conn.close()
    return cursor.fetchall()

def delete_refresh_function():
    # OPENS CONNECTION, EXECUTES PROCEDURE AND CLOSES CONNECTION
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SET PROFILING = 1")
    cursor.callproc("DELETE_REFRESH_FUNCTION")
    cursor.execute("SHOW PROFILES")
    conn.close()
    return cursor.fetchall()



'''
    Refresh streams executed in parallel by process in throughput test
'''
def run_refresh_streams():
    # DICTIONARY FOR STORING REFRESH FUNCTIONS DURATION ([STREAM_NUM][INSERT/DELETE]: DURATION)
    refresh_streams_duration = dict()

    # RUNS A NUMBER OF REFRESH STREAMS ACCORDING TO SCALE FACTOR
    for i in range(NUM_STREAMS):
        current_refresh_stream = dict()
        print("*** Loading refresh stream data... (RS%d)" % i)
        load_refresh_stream_data()
        print("*** Start insert refresh function (RS%d)" % i)
        current_refresh_stream['INSERT'] = insert_refresh_function()
        print("*** Start delete refresh function (RS%d)" % i)
        current_refresh_stream['DELETE'] = delete_refresh_function()
        refresh_streams_duration[i] = current_refresh_stream

    print("*** Refresh streams finished")
    print("*** Refresh streams duration:", refresh_streams_duration)



'''
    Query stream called from power test and throughput test (NUM_STREAMS in parallel)
'''
def run_query_stream(results_queue):

    # START DB CONNECTION
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # SET PROFILING
    cursor.execute("SET PROFILING_HISTORY_SIZE = 22")
    cursor.execute("SET PROFILING = 1")

    # CALL QUERY STREAM PROCEDURE
    cursor.callproc("QUERY_STREAM")

    # SHOW PROFILES
    cursor.execute("SHOW PROFILES")

    # CLOSE DB CONNECTION
    conn.close()

    # TRANSFORM FETCHED PROFILES INTO DICT OF (QUERY NUM: DURATION)
    profiles = dict()
    for row in cursor.fetchall():
        profiles[row[0]] = row[1]

    # RETURN PROFILES RESULT
    results_queue.put(profiles)



'''
    Runs the whole power test, composed of: (1) Insert RF; (2) Query stream; (3) Delete RF
'''
def run_power_test():

    # LOAD REFRESH STREAM DATA
    print("*** Loading refresh stream data...")
    load_refresh_stream_data()

    # INSERT REFRESH FUNCTION
    print("*** Start insert refresh function")
    start_time_insert_rf = time.time()
    insert_refresh_profiles = insert_refresh_function()
    elapsed_time_insert_rf = time.time() - start_time_insert_rf
    print("*** Insert RF elapsed time:", elapsed_time_insert_rf)
    print("*** Insert RF profiles from DB:", insert_refresh_profiles)

    # QUEUE TO GET RESULTS FROM QUERY STREAMS
    profiles_queue = Queue()
    # DECLARE PROCESS WHICH WILL RUN QUERY STREAM
    print("*** Declare QS")
    qs = Process(target=run_query_stream, args=(profiles_queue, ))
    # START QUERY STREAM PROCESS
    print("*** Start QS")
    qs.start()
    # GET RESULTS FROM QUEUE
    print("*** Get queue QS")
    power_test_profiles = profiles_queue.get()
    # JOIN QUERY STREAM PROCESS
    print("*** Join QS")
    qs.join()
    
    # DELETE REFRESH FUNCTION
    print("*** Start delete refresh function")
    start_time_delete_rf = time.time()
    delete_refresh_profiles = delete_refresh_function()
    elapsed_time_delete_rf = time.time() - start_time_delete_rf
    print("*** Delete RF elapsed time:", elapsed_time_delete_rf)
    print("*** Insert RF profiles from DB:", delete_refresh_profiles)

    # CREATES LIST OF DURATIONS FROM THE 22 QUERIES
    execution_times = list(power_test_profiles.values())
    # APPEND REFRESH FUNCTIONS DURATIONS
    execution_times.append(elapsed_time_insert_rf)
    execution_times.append(elapsed_time_delete_rf)

    print("*** QS elapsed time:", sum(power_test_profiles.values()))
    print("*** Power test execution times:", execution_times)

    geo_mean = stats.gmean(execution_times)
    print("*** Geometric mean:", geo_mean)

    return (3600 / geo_mean) * SCALE_FACTOR



'''
    Runs the whole throughput test, composed of # processes for # query streams and one process for # refresh streams
'''
def run_throughput_test():

    # DECLARING PROCESSES
    print("*** Declaring processes...")
    streams = []
    for _ in range(NUM_STREAMS):
        streams.append(Process(target=run_query_stream, args=(Queue(), )))
    streams.append(Process(target=run_refresh_streams))

    # START TIMING EXECUTION
    start_time = time.time()

    # START PROCESSES
    print("*** Starting processes...")
    for p in streams:
        p.start()

    # JOIN PROCESSES
    print("*** Joining processes...")
    for p in streams:
        p.join()

    # FINISH TIMING EXECUTION
    elapsed_time = time.time() - start_time
    print("*** Throughput test elapsed time:", elapsed_time)

    return ((2 * 22) / elapsed_time) * 3600 * SCALE_FACTOR



'''
    Main functin:
        - First, it has to read the initial refresh stream number
        - Then it can start running the tests
        - Lastly, it has to write the refresh stream number to be used in the next run
'''
if __name__ == '__main__':
    # READ LAST REFRESH STREAM NUMBER
    get_refresh_stream_number()

    # RUN POWER@SIZE
    print("\n\n\n*** STARTING POWER TEST...\n")
    power_size = run_power_test()

    # RUN THROUGHPUT
    print("\n\n\n*** STARTING THROUGHPUT TEST...\n")
    throughput = run_throughput_test()

    # SHOW RESULTING METRICS
    print("\n\n\n*** RESULTING METRICS:\n")
    print("Power@Size =", power_size)
    print("Throughput@Size =", throughput)
    print("QphH@Size =", math.sqrt(power_size * throughput))

    # FIXES BAD PROGRAMMING TECHNIQUE
    global refresh_stream_number
    refresh_stream_number += NUM_STREAMS

    # WRITE LAST REFRESH STREAM NUMBER
    set_refresh_stream_number()
