from pg_database import PGDatabase
import numpy as np
import time
import random

class Environment():
    def __init__(self, workload_path='workload/tpch.sql', allow_columns=True, window_size = 20):
        # Database instance
        self.db = PGDatabase()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0
        self.column_order = list(self.db.get_indexes().keys())
        

        # Window-related
        self.window_size = window_size
        workload_buffer, column_count, cost_history = self.initialize_window(self.window_size)
        self.workload_buffer = workload_buffer
        self.column_count = column_count
        self.cost_history = cost_history

        # Environment
        self.allow_columns = allow_columns
        if self.allow_columns:
            self.n_features = len(self.column_order) * 2
            self.n_actions = len(self.column_order)
        else:
            self.n_features = len(self.column_order)
            self.n_actions = len(self.column_order)

        # Stats
        self.trues = 0
        self.falses = 0
        self.acts = []

    """
        Acessed from outside
        - step(action)
        - reset()
        - close()
    """

    def step(self, action):
        # Apply action
        s_prime, reward = self.apply_transition(action)

        # Stats
        self.acts.append(action)
        acts = 999
        falses = self.falses
        trues = self.trues
        if (self.trues + self.falses) == 256:
            self.trues = 0
            self.falses = 0
            acts = len(list(set(self.acts)))
            self.acts = []
            
        return s_prime, reward, False, dict(), trues, falses, acts
        # return s_prime, reward, False, dict()
    
    def reset(self):
        # Workload and indexes
        self.db.reset_indexes()
        self.workload_iterator = 0
        # Window-related
        workload_buffer, column_count, cost_history = self.initialize_window(self.window_size)
        self.workload_buffer = workload_buffer
        self.column_count = column_count
        self.cost_history = cost_history
        # Stats
        self.trues = 0
        self.falses = 0
        self.acts = []

        return self.get_state()
    
    def close(self):
        self.db.reset_indexes()
        close = self.db.close_connection()
        return close

    """
        Reward functions
        - compute_reward_index_scan(query)
        - compute_reward_sum_cost_all()
        - compute_reward_avg_cost_window(query)
    """

    def compute_reward_index_scan(self, drop, table, column):
        # Check whether the index could be used in the last len(window) queries
        total_count = 0
        if drop:
            print("Foi drop, cria ficticio")
            self.db.create_index(table, column, verbose=False)
        for q in self.workload_buffer:
            count = self.db.get_query_use(q, column)
            total_count += count
        if drop:
            print("E dropa ficticio")
            self.db.drop_index(table, column, verbose=False)

        # window_count = self.get_column_count_window()
        # Multiply it by its count in the state

        reward = total_count * 1000
        if drop and total_count > 0:
            reward = reward * -1
            print("DROP", column, total_count, reward)
        elif drop:
            reward = 1
            print("DROP", column, total_count, reward)
        elif not drop and total_count == 0:
            reward = -1000
            print("CREATE", column, total_count, reward)
        else:
            print("CREATE", column, total_count, reward)

        print("")
        return reward

    def compute_reward_sum_cost_all(self):
        costs = [self.db.get_query_cost(q) for q in self.workload]
        reward = (1/sum(costs)) * 10000000000
        return reward

    def compute_reward_avg_cost_window(self):
        # Get avg of last 10 costs
        if len(self.cost_history) >= self.window_size:
            cost = sum(self.cost_history[-self.window_size:])
            cost_avg = cost/self.window_size
        else:
            cost = sum(self.cost_history)
            cost_avg = cost/len(self.cost_history)
        reward = (1/cost_avg) * 100000000
        return reward

    """
        Transition methods
        - apply_transition(action)
        - apply_index_change(action)
        - step_workload()
        - random_step_workload()
    """

    def apply_transition(self, action):
        # Apply index change
        changed, drop, table, column = self.apply_index_change(action)

        # Execute next_query
        next_query, elapsed_time = self.step_workload()
        # next_query, elapsed_time = self.random_step_workload()

        # Compute reward
        if changed:
            # reward = self.compute_reward_sum_cost_all()
            # reward = self.compute_reward_avg_cost_window(next_query)
            reward = self.compute_reward_index_scan(drop, table, column)
        else:
            reward = -1000
            print("NONE!", column, reward)
            print("")

        # Get next state
        s_prime = self.get_state()

        return s_prime, reward
    
    def apply_index_change(self, action):
        indexes = self.db.get_indexes()
        drop = False
        if action >= self.n_actions/2: 
            drop = True
            action = int(action - self.n_actions/2)
        for idx, column in enumerate(indexes):
            if idx == action:
                # Get the table
                for table in self.db.tables.keys():
                    if column in self.db.tables[table]:
                        if drop:
                            if indexes[column] == 0: 
                                # print("FALSE - DRP", column)
                                self.falses += 1
                                return False, drop, table, column
                            else:
                                # print("TRUE  - DRP", column)
                                self.db.drop_index(table, column, verbose=False)
                                self.trues += 1
                                return True, drop, table, column
                        else:
                            if indexes[column] == 1:
                                # print("FALSE - CRT", column)
                                self.falses += 1
                                return False, drop, table, column
                            else: 
                                # print("TRUE  - CRT", column)
                                self.db.create_index(table, column, verbose=False)
                                self.trues += 1
                                return True, drop, table, column

    def step_workload(self):
        query = self.workload[self.workload_iterator]

        # Execute query
        start = time.time()
        # self.db.execute(query, verbose=False)
        end = time.time()
        elapsed_time = end - start
        
        # Update window
        self.update_column_count(query)
        self.cost_history.append(self.db.get_query_cost(query))
        self.workload_buffer.append(query)
        self.workload_buffer.pop(0)

        # Manage iterator
        if self.workload_iterator+1 == len(self.workload):
            self.workload_iterator = 0
        else:
            self.workload_iterator += 1
        
        return query, elapsed_time
    
    def random_step_workload(self):
        query = random.choice(self.workload)

        # Execute query
        start = time.time()
        # self.db.execute(query, verbose=False)
        end = time.time()
        elapsed_time = end - start
        
        # Update window
        self.update_column_count(query)
        self.cost_history.append(self.db.get_query_cost(query))
        self.workload_buffer.append(query)
        self.workload_buffer.pop(0)

        return query, elapsed_time

    """
        Count updates
        - update_column_count(query)
        - get_column_count_window()
    """

    def update_column_count(self, query):
        column_count = [0] * len(self.column_order)
        select_split = query.split("SELECT")
        for select in select_split:
            if select != '':
                if "WHERE" in str(select).upper():
                    where = select.split("WHERE")[1]
                    avoid = ['GROUP BY', 'ORDER BY', 'LIMIT']
                    for item in avoid:
                        if item in where:
                            where = where.split(item)[0]
                            break
                    for idx, column in enumerate(self.column_order):
                        if str(column).upper() in str(where).upper():
                            column_count[idx] += 1
        self.column_count.append(column_count)
    
    def get_column_count_window(self):
        # Get last 10 items count
        if len(self.column_count) >= self.window_size:
            total_count = [0] * len(self.column_order)
            for count in self.column_count[-self.window_size:]:
                total_count = [sum(col) for col in zip(total_count, count)]
        else:
            # Get the count of what we have
            total_count = [0] * len(self.column_order)
            for count in self.column_count:
                total_count = [sum(col) for col in zip(total_count, count)]
        return total_count
    
    """
        Initialization methods
    """

    def initialize_window(self, window_size):
        workload_buffer = list()
        self.column_count = list()
        cost_history = list()

        sample_queries = random.sample(self.workload, window_size)
        for query in sample_queries:
            workload_buffer.append(query)
            cost_history.append(self.db.get_query_cost(query))
            self.update_column_count(query)

        return workload_buffer, self.column_count, cost_history
    
    def get_state(self):
        if self.allow_columns:
            indexes = np.array(list(self.db.get_indexes().values()))
            current_count = np.array(self.get_column_count_window())
            state = np.concatenate((indexes, current_count))
        else:
            indexes = np.array(list(self.db.get_indexes().values()))
            state = indexes
        return state
    
    def load_workload(self, path):
        with open(path, 'r') as f:
            data = f.read()
        workload = data.split('\n')
        return workload


if __name__ == "__main__":
    from pprint import pprint
    env = Environment()

    pprint(env.column_order)