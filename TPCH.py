import mysql.connector
import math
import time
import os

from multiprocessing import Process, Queue # FOR PARALLEL EXECUTION
from scipy import stats # FOR GEOMETRIC MEAN


class TPCH:


    '''
        SCALE_FACTOR    1   10  30  100
        NUM_STREAMS     2   3   4   5
    '''
    SCALE_FACTOR = 1
    NUM_STREAMS = 2


    '''
        Global configuration
    '''
    # SERVER
    DB_CONFIG = {'user': 'dbuser', 'password': 'dbuser', 'host': '127.0.0.1', 'database': 'tpch'}
    REFRESH_FILES_PATH = '/home/sap-server/smartix/tpch-tools/dbgen/%d' % SCALE_FACTOR
    # LOCAL
    # DB_CONFIG = {'user': 'root', 'password': 'root', 'host': '127.0.0.1', 'database': 'tpch'}
    # REFRESH_FILES_PATH = '/home/gabriel/sap/tpch-tools/dbgen/%d' % SCALE_FACTOR


    '''
        Refresh stream sequence number (leave it at 1)
    '''
    refresh_stream_number = 1


    '''
        Keeps track of refresh files order by storing sequence number in a .txt file
        If the file does not exist, it will create and start from number 1
    '''
    def __get_refresh_stream_number(self):
        if os.path.exists("%s/refresh_stream_number.txt" % self.REFRESH_FILES_PATH):
            with open("%s/refresh_stream_number.txt" % self.REFRESH_FILES_PATH, "r+") as f:
                self.refresh_stream_number = int(f.read())
        else:
            with open("%s/refresh_stream_number.txt" % self.REFRESH_FILES_PATH, "w+") as f:
                f.write("%d" % self.refresh_stream_number)

    def __set_refresh_stream_number(self):
        with open("%s/refresh_stream_number.txt" % self.REFRESH_FILES_PATH, "w+") as f:
                f.write("%d" % self.refresh_stream_number)


    '''
        Loads data from refresh files to temporary tables in the database
    '''
    def __load_refresh_stream_data(self):
        # print("*** Load refresh stream number:", self.refresh_stream_number)

        # STRINGS TO BE EXECUTED BY CURSOR
        delete = "load data local infile '{}/delete.{}' into table rfdelete fields terminated by '|' lines terminated by '\n';".format(self.REFRESH_FILES_PATH, self.refresh_stream_number)
        orders = "load data local infile '{}/orders.tbl.u{}' into table orders_temp fields terminated by '|' lines terminated by '\n';".format(self.REFRESH_FILES_PATH, self.refresh_stream_number)
        lineitem = "load data local infile '{}/lineitem.tbl.u{}' into table lineitem_temp fields terminated by '|' lines terminated by '\n';".format(self.REFRESH_FILES_PATH, self.refresh_stream_number)

        # OPEN DB CONNECTION, EXECUTE LOADS AND CLOSE CONNECTION
        conn = mysql.connector.connect(**self.DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(delete)
        cursor.execute(orders)
        cursor.execute(lineitem)
        conn.close()
        
        # INCREMENT REFRESH STREAM NUMBER FOR NEXT STREAM
        self.refresh_stream_number += 1


    '''
        Refresh functions calling respective procedures in the DB
        Each function returns its duration
    '''
    def __insert_refresh_function(self):
        # OPENS CONNECTION, EXECUTES PROCEDURE AND CLOSES CONNECTION
        conn = mysql.connector.connect(**self.DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SET PROFILING = 1")
        cursor.callproc("INSERT_REFRESH_FUNCTION")
        cursor.execute("SHOW PROFILES")
        results = cursor.fetchall()
        conn.close()

        # SUM AND RETURN TOTAL EXECUTION TIME FROM FETCHED RESULTS
        duration = 0
        for row in results: 
            duration += row[1]
        return duration

    def __delete_refresh_function(self):
        # OPENS CONNECTION, EXECUTES PROCEDURE AND CLOSES CONNECTION
        conn = mysql.connector.connect(**self.DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SET PROFILING = 1")
        cursor.callproc("DELETE_REFRESH_FUNCTION")
        cursor.execute("SHOW PROFILES")
        results = cursor.fetchall()
        conn.close()
        
        # SUM AND RETURN TOTAL EXECUTION TIME FROM FETCHED RESULTS
        duration = 0
        for row in results: 
            duration += row[1]
        return duration


    '''
        Refresh streams executed in parallel by process in throughput test
    '''
    def __run_refresh_streams(self, results_queue):
        # LIST FOR STORING REFRESH FUNCTIONS DURATION
        refresh_streams_duration = []

        # RUNS A NUMBER OF REFRESH STREAMS ACCORDING TO SCALE FACTOR
        for _ in range(self.NUM_STREAMS):
            self.__load_refresh_stream_data()
            refresh_streams_duration.append(self.__insert_refresh_function())
            refresh_streams_duration.append(self.__delete_refresh_function())

        # print("\n*** Refresh streams finished")
        # print("*** Refresh streams duration:", refresh_streams_duration)
        
        # RETURNS TOTAL REFRESH STREAMS EXECUTION TIME
        results_queue.put(refresh_streams_duration)



    '''
        Query stream called from power test and throughput test (NUM_STREAMS in parallel)
    '''
    def __run_query_stream(self, results_queue):
        # START DB CONNECTION
        conn = mysql.connector.connect(**self.DB_CONFIG)
        cursor = conn.cursor()

        # SET PROFILING
        cursor.execute("SET PROFILING_HISTORY_SIZE = 22")
        cursor.execute("SET PROFILING = 1")

        # CALL QUERY STREAM PROCEDURE
        cursor.callproc("QUERY_STREAM")

        # SHOW PROFILES
        cursor.execute("SHOW PROFILES")

        results = cursor.fetchall()

        # CLOSE DB CONNECTION
        conn.close()

        # TRANSFORM FETCHED PROFILES INTO DICT OF (QUERY NUM: DURATION)
        profiles = dict()
        for row in results:
            profiles[row[0]] = row[1]

        # RETURN PROFILES RESULT
        results_queue.put(profiles) # IF RUNNING IN A PROCESS
        return profiles



    '''
        Runs the whole power test, composed of: (1) Insert RF; (2) Query stream; (3) Delete RF
    '''
    def __run_power_test(self):
        # LOAD REFRESH STREAM DATA
        # print("\n*** Loading refresh stream data...")
        self.__load_refresh_stream_data()

        # INSERT REFRESH FUNCTION
        # print("\n*** Start insert refresh function")
        insert_refresh_profile = self.__insert_refresh_function()
        # print("*** Insert RF duration:", insert_refresh_profile)

        # RUN QUERY STREAM
        # print("\n*** Start query stream")
        query_stream_profiles = self.__run_query_stream(Queue())
        # print("*** Query stream duration:", sum(query_stream_profiles.values()))
        
        # DELETE REFRESH FUNCTION
        # print("\n*** Start delete refresh function")
        delete_refresh_profile = self.__delete_refresh_function()
        # print("*** Delete RF duration:", delete_refresh_profile)

        # CREATES LIST OF DURATIONS OF THE 22 QUERIES AND REFRESH FUNCTIONS
        power_test_profiles = list(query_stream_profiles.values())
        power_test_profiles.append(insert_refresh_profile)
        power_test_profiles.append(delete_refresh_profile)

        # CALCULATES GEOMETRIC MEAN
        geo_mean = stats.gmean(power_test_profiles)
        power = (3600 / geo_mean) * self.SCALE_FACTOR

        # print("\n*** Power test execution profiles:", power_test_profiles)
        # print("*** Geometric mean:", geo_mean)

        # RETURN POWER@SIZE METRIC
        return power


    '''
        Runs the whole throughput test, composed of # processes for # query streams and one process for # refresh streams
    '''
    def __run_throughput_test(self):
        # DECLARING PROCESSES
        # print("\n*** Declaring processes...")
        results_queue = Queue()
        throughput_test_profiles = []
        streams = []
        for _ in range(self.NUM_STREAMS):
            streams.append(Process(target=self.__run_query_stream, args=(results_queue, )))
        streams.append(Process(target=self.__run_refresh_streams, args=(results_queue, )))

        # START TIMING EXECUTION
        start_time = time.time()

        # START PROCESSES
        # print("*** Starting processes...")
        for p in streams:
            p.start()

        # print("*** Getting results...")
        for p in streams:
            throughput_test_profiles.append(results_queue.get())

        # JOIN PROCESSES
        # print("*** Joining processes...")
        for p in streams:
            p.join()

        # print("\n*** Throughput test execution profiles:", throughput_test_profiles)

        # FINISH TIMING EXECUTION
        elapsed_time = time.time() - start_time
        # print("\n*** Throughput test elapsed time:", elapsed_time)

        return ((self.NUM_STREAMS * 22) / elapsed_time) * 3600 * self.SCALE_FACTOR


    def run(self):

        print("\nStarting benchmark...")

        # READ LAST REFRESH STREAM NUMBER
        self.__get_refresh_stream_number()

        # RUN POWER@SIZE
        # print("\n*** STARTING POWER TEST...")
        power = self.__run_power_test()

        # RUN THROUGHPUT
        # print("\n*** STARTING THROUGHPUT TEST...")
        throughput = self.__run_throughput_test()

        # RUN THROUGHPUT
        # print("\n*** CALCULATING QPHH...")
        qphh = math.sqrt(power * throughput)

        # SHOW RESULTING METRICS
        # print("\n*** RESULTING METRICS:\n")
        print("Power@Size =", power)
        print("Throughput@Size =", throughput)
        print("QphH@Size =", qphh, end="\n")

        # FIXES BAD PROGRAMMING TECHNIQUE
        self.refresh_stream_number += self.NUM_STREAMS

        # WRITE LAST REFRESH STREAM NUMBER
        self.__set_refresh_stream_number()

        # PUT METRICS INTO AN ARRAY AND RETURN
        # results = []
        # results.append(power)
        # results.append(throughput)
        # results.append(qphh)

        # return results

        return qphh
