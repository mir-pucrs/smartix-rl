from pg_database import PG_Database
import numpy as np
import time
import random

class Environment():
    def __init__(self, workload_path='data/workload/tpch.sql', hypo=True, allow_columns=False, flip=False, window_size = 40):
        # Database instance
        self.db = PG_Database(hypo=hypo)

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
        self.flip = flip
        if self.allow_columns:
            self.n_features = len(self.column_order) * 2
            self.n_actions = len(self.column_order) * 2
        else:
            self.n_features = len(self.column_order)
            self.n_actions = len(self.column_order) * 2
        if self.flip:
            self.n_actions = len(self.column_order) ################# + 1

        
        # DEBUG
        self.trues = 0
        self.falses = 0
        self.counter_create_true = 0
        self.counter_create_false = 0
        self.counter_drop_true = 0
        self.counter_drop_false = 0
        self.cnt_drop_bom = 0
        self.cnt_drop_ruim = 0
        self.cnt_create_bom = 0
        self.cnt_create_ruim = 0
        self.cnt_no_op = 0

    """
        Acessed from outside
        - step(action)
        - reset()
        - close()
    """
    def step(self, action):
        # NO OP
        if self.flip and action == 8: ##################### action == len(self.column_order) or 
            self.cnt_no_op += 1
            return self.get_state(), -5, False, dict()
        
        # Apply action
        s_prime, reward = self.apply_transition(action)

        return s_prime, reward, False, dict()
    
    def reset(self):
        # Workload and indexes
        self.db.reset_indexes()
        self.workload_iterator = 0
        # Window-related
        workload_buffer, column_count, cost_history = self.initialize_window(self.window_size)
        self.workload_buffer = workload_buffer
        self.column_count = column_count
        self.cost_history = cost_history

        return self.get_state()
    
    def close(self):
        self.db.reset_indexes()
        close = self.db.close_connection()
        return close

    def debug(self):
        indexes_dict = self.db.get_indexes()
        if indexes_dict['c_acctbal'] == 1: print('c_acctbal')
        if indexes_dict['p_brand'] == 1: print('p_brand')
        if indexes_dict['p_container'] == 1: print('p_container')
        if indexes_dict['p_size'] == 1: print('p_size')
        if indexes_dict['l_shipdate'] == 1: print('l_shipdate')
        print(sum(indexes_dict.values()))
        
        if self.flip:
            print('CREATE BOM  ', self.cnt_create_bom)
            print('CREATE RUIM ', self.cnt_create_ruim)
            print('DROP   BOM  ', self.cnt_drop_bom)
            print('DROP   RUIM ', self.cnt_drop_ruim)
            print('NO OP', self.cnt_no_op)
        else:
            print('Trues  ', self.trues)
            print('Falses ', self.falses)
            print('CRT T', self.counter_create_true)
            print('CRT F', self.counter_create_false)
            print('DRP T', self.counter_drop_true)
            print('DRP F', self.counter_drop_false)
        
        self.counter_create_true = 0
        self.counter_create_false = 0
        self.counter_drop_true = 0
        self.counter_drop_false = 0
        self.cnt_drop_bom = 0
        self.cnt_drop_ruim = 0
        self.cnt_create_bom = 0
        self.cnt_create_ruim = 0
        self.cnt_no_op = 0
        self.trues = 0
        self.falses = 0

    """
        Reward functions
        - compute_reward_index_scan(query)
        - compute_reward_sum_cost_all()
        - compute_reward_avg_cost_window(query)
    """

    def compute_reward_index_scan(self, changed, drop, table, column):
        # If action was a drop, create hypothetical
        if drop and changed: self.db.create_index(table, column, verbose=False)

        # Check amount of queries the index could be used in the last len(window)
        total_count = 0
        for q in self.workload_buffer:
            count = self.db.get_query_use(q, column)
            total_count += count

        # Drop hypothetical in case it was a drop
        if drop and changed: self.db.drop_index(table, column, verbose=False)

        if changed and drop:
            if total_count == 0: # DROPA == 0 - BOM
                self.cnt_drop_bom += 1
                reward = 5
            else: # DROPA > 0 - RUIM
                self.cnt_drop_ruim += 1
                reward = -20
        if changed and not drop:
            if total_count == 0: # CRIA == 0 - RUIM
                self.cnt_create_ruim += 1
                reward = -15
            else: # CRIA > 0 - BOM
                self.cnt_create_bom += 1
                reward = 10

        # if not changed and drop: # DROPA JA DROPADO
        #     print('AQUI NAO')
        #     if total_count == 0:
        #         reward = -1
        #     else:
        #         reward = -10
        # if not changed and not drop: # CRIA JA CRIADO
        #     print('AQUI NAO')
        #     if total_count == 0:
        #         reward = -10
        #     else:
        #         reward = -1
        
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
        if self.flip:
            changed, drop, table, column = self.apply_index_change_flip(action)
        else:
            changed, drop, table, column = self.apply_index_change(action)

        # Execute next_query
        next_query, elapsed_time = self.step_workload()
        # next_query, elapsed_time = self.random_step_workload()

        # Compute reward
        reward = self.compute_reward_index_scan(changed, drop, table, column)

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
                        counter = False
                        if column == 'c_acctbal': counter = True
                        if column == 'p_brand': counter = True
                        if column == 'p_container': counter = True
                        if column == 'p_size': counter = True
                        if column == 'l_shipdate': counter = True
                        if drop:
                            if indexes[column] == 0: 
                                # print("FALSE - DRP", column)
                                self.falses += 1
                                if counter: self.counter_drop_false += 1
                                return False, drop, table, column
                            else:
                                # print("TRUE  - DRP", column)
                                self.trues += 1
                                if counter: self.counter_drop_true += 1
                                self.db.drop_index(table, column, verbose=False)
                                return True, drop, table, column
                        else:
                            if indexes[column] == 1:
                                # print("FALSE - CRT", column)
                                self.falses += 1
                                if counter: self.counter_create_false += 1
                                return False, drop, table, column
                            else: 
                                # print("TRUE  - CRT", column)
                                self.trues += 1
                                if counter: self.counter_create_true += 1
                                self.db.create_index(table, column, verbose=False)
                                return True, drop, table, column

    def apply_index_change_flip(self, action):
        indexes = self.db.get_indexes()
        for idx, column in enumerate(indexes):
            if idx == action:
                # Get the table
                for table in self.db.tables.keys():
                    if column in self.db.tables[table]:
                        if indexes[column] == 1:
                            drop = True
                            cnt = 0
                            while self.db.get_indexes()[column] != 0:
                                self.db.drop_index(table, column, verbose=False)
                                if cnt != 0: print('WHILE', cnt, idx, action, table, column)
                                cnt += 1
                            if cnt != 1: print('SAIU WHILE')
                            return True, drop, table, column
                        else:
                            drop = False
                            cnt = 0
                            while self.db.get_indexes()[column] != 1:
                                self.db.create_index(table, column, verbose=False)
                                if cnt != 0: print('WHILE', cnt, idx, action, table, column)
                                cnt += 1
                            if cnt != 1: print('SAIU WHILE')
                            return True, drop, table, column

    def step_workload(self):
        query = self.workload[self.workload_iterator]

        # Execute query
        start = time.time()
        # print(self.workload_iterator)
        # self.db.execute(query, verbose=True)
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

        sample_queries = list()
        while len(sample_queries) < window_size:
            sample_queries.append(random.sample(self.workload, 1)[0])
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
    # env = Environment(hypo=False)
    # env.reset()
    # env.close()
    env = Environment(hypo=True)
    env.reset()
    while True:
        env.step(7)