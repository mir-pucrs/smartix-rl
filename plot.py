import PyGnuplot as gp

with open("plots/averages_rewards_history.gnu") as f: 
    gp.c(f.read())
    gp.pdf('rewards_history_plot.pdf')


'''
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/mysql/connector/connection_cext.py", line 273, in get_rows
    row = self._cmysql.fetch_row()
_mysql_connector.MySQLInterfaceError: Server shutdown in progress

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "environment.py", line 185, in <module>
    agent.train(env)
  File "/home/sap-server/smartix/smartix-rl/agent.py", line 177, in train
    self.next_state, self.reward = self.env.step(self.action)
  File "environment.py", line 37, in step
    reward = self.get_reward(state)
  File "environment.py", line 60, in get_reward
    self.rewards[state] = self.benchmark.run()
  File "/home/sap-server/smartix/smartix-rl/benchmark.py", line 12, in run
    results = TPCH().run()
  File "/home/sap-server/smartix/smartix-rl/TPCH.py", line 256, in run
    power = self.__run_power_test()
  File "/home/sap-server/smartix/smartix-rl/TPCH.py", line 184, in __run_power_test
    query_stream_profiles = self.__run_query_stream(Queue())
  File "/home/sap-server/smartix/smartix-rl/TPCH.py", line 148, in __run_query_stream
    cursor.callproc("QUERY_STREAM")
  File "/usr/local/lib/python3.6/dist-packages/mysql/connector/cursor_cext.py", line 447, in callproc
    cur._handle_result(result)
  File "/usr/local/lib/python3.6/dist-packages/mysql/connector/cursor_cext.py", line 163, in _handle_result
    self._handle_resultset()
  File "/usr/local/lib/python3.6/dist-packages/mysql/connector/cursor_cext.py", line 651, in _handle_resultset
    self._rows = self._cnx.get_rows()[0]
  File "/usr/local/lib/python3.6/dist-packages/mysql/connector/connection_cext.py", line 295, in get_rows
    sqlstate=exc.sqlstate)
mysql.connector.errors.OperationalError: 1053 (08S01): Server shutdown in progress
'''