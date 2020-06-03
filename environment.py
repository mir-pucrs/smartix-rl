from pg_database import PGDatabase
import numpy as np
import time
import random

class Environment():
    def __init__(self, workload_path='workload/tpch.sql', allow_columns=False):
        # Database instance
        self.db = PGDatabase()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0

        # State and reward
        self.column_order = list(self.db.get_indexes().keys())
        self.column_count = list()
        self.window_size = 10
        self.cost_history = self.initialize_cost_history(self.window_size)
        # self.time_history = self.initialize_time_history(self.window_size)

        ######################
        self.allow_columns = allow_columns
        ######################

        # Environment
        if self.allow_columns:
            self.n_features = len(self.column_order) * 2
            self.n_actions = self.n_features
        else:
            self.n_features = len(self.column_order)
            self.n_actions = self.n_features

    def step(self, action):
        # Apply action
        s_prime, reward = self.apply_transition(action)
        return s_prime, reward, False, dict()

    def apply_transition(self, action):
        # Get next state
        # s = self.get_state()

        # Apply index change
        # print("Applying index...")
        self.apply_index_change(action)

        # Execute next_query
        # print("Executing query...")
        next_query, elapsed_time = self.step_workload()
        # next_query, elapsed_time = self.random_step_workload()

        # Compute reward
        reward = self.compute_reward_cost_all()
        # reward = self.compute_reward_cost(next_query)
        # reward = self.compute_reward_columns(s, action)
        # reward = self.compute_reward_time(elapsed_time)

        # Get next state
        s_prime = self.get_state()

        return s_prime, reward

    def compute_reward_cost_all(self):
        costs = [self.db.get_query_cost(q) for q in self.workload]
        reward = (1/sum(costs)) * 10000000000
        return reward

    def compute_reward_cost(self, query):
        # cost = self.db.get_query_cost(query)
        # return (1/cost) * 100000000

        self.cost_history.append(self.db.get_query_cost(query))
        # Get avg of last 10 costs
        if len(self.cost_history) >= self.window_size:
            cost = sum(self.cost_history[-self.window_size:])
            cost_avg = cost/self.window_size
        else:
            cost = sum(self.cost_history)
            cost_avg = cost/len(self.cost_history)
        reward = (1/cost_avg) * 100000000
        return reward

    def compute_reward_columns(self, state, action):
        indexes = state.tolist()[:int(-self.n_features/2)]
        current_count = self.get_column_count()
        if action >= self.n_actions/2:  # DROP
            index = int(action - self.n_actions/2)
            print('DROP', index)
            if indexes[index] == 0:
                reward = -100
                print("- REPEATED!!!")
            else:
                reward = 100 * (current_count[index]+1)
        else:  # CREATE
            index = int(action)
            print('CREATE', index)
            if indexes[index] == 1:
                reward = -100
                print("- REPEATED!!!")
            else:
                reward = 100 * (current_count[index]+1)
        return reward

    def compute_reward_time(self, elapsed_time):
        self.time_history.append(elapsed_time)
        if len(self.time_history) >= self.window_size:
            time = sum(self.time_history[-self.window_size:])
            time_avg = time/self.window_size
        else:
            time = sum(self.time_history)
            time_avg = time/len(self.time_history)
        reward = (1/time_avg) * 100
        return reward

    def get_state(self):
        if self.allow_columns:
            indexes = np.array(list(self.db.get_indexes().values()))
            current_count = np.array(self.get_column_count())
            state = np.concatenate((indexes, current_count))
        else:
            indexes = np.array(list(self.db.get_indexes().values()))
            state = indexes
        return state

    def get_column_count(self):
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
                            self.db.drop_index(table, column, verbose=False)
                        else:
                            self.db.create_index(table, column, verbose=False)
                        break
                break

    def step_workload(self):
        query = self.workload[self.workload_iterator]

        start = time.time()
        # self.db.execute(query, verbose=False)
        end = time.time()
        elapsed_time = end - start
        
        self.update_column_count(query)
        if self.workload_iterator+1 == len(self.workload):
            self.workload_iterator = 0
        else:
            self.workload_iterator += 1
        return query, elapsed_time
    
    def random_step_workload(self):
        query = random.choice(self.workload)

        start = time.time()
        # self.db.execute(query, verbose=False)
        end = time.time()
        elapsed_time = end - start
        
        self.update_column_count(query)

        return query, elapsed_time
    
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
    
    def initialize_cost_history(self, window_size):
        cost_history = list()
        sample_queries = random.sample(self.workload, window_size)
        for query in sample_queries:
            cost_history.append(self.db.get_query_cost(query))
            self.update_column_count(query)
        return cost_history
    
    def initialize_time_history(self, window_size):
        import random
        time_history = list()
        sample_queries = random.sample(self.workload, window_size)
        for query in sample_queries:
            start = time.time()
            self.db.execute(query, verbose=False)
            end = time.time()
            elapsed_time = end - start
            time_history.append(elapsed_time)
            self.update_column_count(query)
        return time_history

    def load_workload(self, path):
        with open(path, 'r') as f:
            data = f.read()
        workload = data.split('\n')
        return workload

    def reset(self):
        self.workload_iterator = 0
        self.column_count = list()
        self.cost_history = self.initialize_cost_history(self.window_size)
        # self.time_history = self.initialize_time_history(self.window_size)
        self.db.reset_indexes()
        return self.get_state()
    
    def close(self):
        self.db.reset_indexes()
        return self.db.close_connection()


if __name__ == "__main__":
    from pprint import pprint
    env = Environment()

    print(env.column_order)