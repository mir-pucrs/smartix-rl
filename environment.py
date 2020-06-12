from pg_database import PG_Database
import numpy as np
import time
import random

class Environment():
    def __init__(self, workload_path='data/workload/tpch.sql', hypo=True, allow_columns=False, flip=False, no_op=False, window_size = 50):
        # DBMS
        self.db = PG_Database(hypo=hypo)

        # Database
        self.table_columns = self.db.tables
        self.tables = list(self.table_columns.keys())
        self.columns = np.concatenate(list(self.table_columns.values())).tolist()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0
        self.window_size = window_size

        # Environment
        self.allow_columns = allow_columns
        self.flip = flip
        self.no_op = no_op
        if self.allow_columns:
            self.n_features = len(self.columns) * 2
            self.n_actions = len(self.columns) * 2
        else:
            self.n_features = len(self.columns)
            self.n_actions = len(self.columns) * 2
        if self.flip:
            self.n_actions = len(self.columns)
        if self.no_op:
            self.n_actions += 1

    """
        Acessed from outside
        - step(action)
        - reset()
        - close()
    """
    def step(self, action):
        if self.flip and action == len(self.columns):
            return self.get_state(), -1, False, dict()
        
        # Apply action
        s_prime, reward = self.apply_transition(action)

        return s_prime, reward, False, dict()
    
    def reset(self):
        # Workload and indexes
        self.db.reset_indexes()
        self.workload_iterator = 0

        # Window-related
        self.workload_buffer, self.column_count, self.index_count, self.cost_history = self.initialize_window(self.window_size)

        return self.get_state()
    
    def close(self):
        self.db.reset_indexes()
        return self.db.close_connection()

    def debug(self):
        indexes_dict = self.db.get_indexes()
        if indexes_dict['c_acctbal'] == 1: print('c_acctbal')
        if indexes_dict['p_brand'] == 1: print('p_brand')
        if indexes_dict['p_container'] == 1: print('p_container')
        if indexes_dict['p_size'] == 1: print('p_size')
        if indexes_dict['l_shipdate'] == 1: print('l_shipdate')
        if indexes_dict['o_orderdate'] == 1: print('o_orderdate')
        print(sum(indexes_dict.values()))

    """
        Reward functions
        - compute_reward_index_scan(query)
        - compute_reward_sum_cost_all()
        - compute_reward_avg_cost_window(query)
    """

    def compute_reward_weight_columns(self, changed, drop, table, column):
        if not changed: 
            print('NOT CHANGED')
            return -1

        column_count = self.get_column_count_window()
        count = column_count[self.columns.index(column)]
        
        threshold = 1
        reward = 0
        if drop and count > threshold:
            reward = -count * 2
        elif drop and count <= threshold:
            reward = count
        elif not drop and count <= threshold:
            reward = -count * 2
        elif not drop and count > threshold:
            reward = count

        return reward
    
    def compute_reward_weight_indexes(self, changed, drop, table, column):
        if not changed: 
            print('NOT CHANGED')
            return -1

        index_count = self.get_index_count_window()
        count = index_count[self.columns.index(column)]

        if count > 0: print(column, count)

        reward = 10
        if drop and count > 0:
            reward = -(reward*count) * 4
        if not drop and count > 0:
            reward = (reward*count) * 2
        if not drop and count == 0:
            reward = -(reward*count) * 4

        return reward


    def compute_reward_index_scan(self, changed, drop, table, column):
        # If action was a drop, create hypothetical
        if drop and changed: self.db.create_index(table, column)

        # Check amount of queries the index could be used in the last len(window)
        total_count = 0
        for q in self.workload_buffer:
            if self.db.get_query_use(q, column) > 0:
                total_count = 1
                break

        # Drop hypothetical in case it was a drop
        if drop and changed: self.db.drop_index(table, column)

        reward = 0
        if changed and drop:
            if total_count == 0: # DROPA == 0 - BOM
                reward = 5
            else: # DROPA > 0 - RUIM
                reward = -20
        if changed and not drop:
            if total_count == 0: # CRIA == 0 - RUIM
                reward = -15
            else: # CRIA > 0 - BOM
                reward = 10
        
        return reward

    def compute_reward_avg_cost_window(self):
        cost = sum(self.cost_history[-self.window_size:])
        cost_avg = cost/self.window_size
        reward = (1/cost_avg) * 100000
        return reward
    
    def compute_reward_cost(self, query):
        cost = self.db.get_query_cost(query)
        reward = (1/cost) * 100000
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
        query, elapsed_time = self.step_workload()
        # query, elapsed_time = self.random_step_workload()

        # Compute reward
        # reward = self.compute_reward_index_scan(changed, drop, table, column)
        # reward = self.compute_reward_avg_cost_window()
        # reward = self.compute_reward_cost(query)
        # reward = self.compute_reward_weight_columns(changed, drop, table, column)
        reward = self.compute_reward_weight_indexes(changed, drop, table, column)

        # Get next state
        s_prime = self.get_state()

        return s_prime, reward
    
    def apply_index_change(self, action):
        indexes = self.db.get_indexes()

        # Check if drop
        drop = False
        if action >= self.n_actions/2: 
            drop = True
            action = int(action - self.n_actions/2)

        # Get table and column
        column = self.columns[action]
        for table_name in self.tables:
            if column in self.table_columns[table_name]:
                table = table_name

        # Apply change
        if drop:
            if indexes[column] == 0: 
                changed = False
            else:
                self.db.drop_index(table, column)
                changed = True
        else:
            if indexes[column] == 1:
                changed = False
            else: 
                self.db.create_index(table, column)
                changed = True
                
        return changed, drop, table, column

    def apply_index_change_flip(self, action):
        indexes = self.db.get_indexes()

        # Get table and column
        column = self.columns[action]

        for table_name in self.tables:
            if column in self.table_columns[table_name]:
                table = table_name
        
        # WORKAROUND FOR WEIRD COLUMN THAT DOES NOT WORK CREATING INDEX
        if column == 's_comment': return False, False, table, column

        # Apply change
        if indexes[column] == 1: 
            self.db.drop_index(table, column)
            drop = True
        else:
            self.db.create_index(table, column)
            drop = False
        
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
        # self.column_count.append(self.get_column_count(query))
        self.index_count.append(self.get_index_count(query))
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
        # self.db.execute(query)
        end = time.time()
        elapsed_time = end - start
        
        # Update window
        # self.column_count.append(self.get_column_count(query))
        self.index_count.append(self.get_index_count(query))
        self.cost_history.append(self.db.get_query_cost(query))
        self.workload_buffer.append(query)
        self.workload_buffer.pop(0)

        return query, elapsed_time

    """
        Count updates
        - update_column_count(query)
        - get_column_count_window()
    """

    def get_column_count(self, query):
        column_count = [0] * len(self.columns)
        where_columns = self.get_where_columns(query)
        for column in where_columns:
            column_count[self.columns.index(column)] = 1
        return column_count
    
    def get_column_count_window(self):
        total_count = [0] * len(self.columns)
        for count in self.column_count[-self.window_size:]:
            total_count = [sum(col) for col in zip(total_count, count)]
        return total_count

    def get_index_count(self, query):
        index_count = [0] * len(self.columns)
        current_indexes = self.db.get_indexes()
        where_columns = self.get_where_columns(query)
        for column in where_columns:
            for table in self.tables:
                if column in self.table_columns[table]:
                    if current_indexes[column] == 1: create = False
                    else: create = True
                    if create: self.db.create_index(table, column)
                    count = self.db.get_query_use(query, column)
                    index_count[self.columns.index(column)] = count
                    if create: self.db.drop_index(table, column)
        return index_count
    
    def get_index_count_window(self):
        total_count = [0] * len(self.columns)
        for count in self.index_count[-self.window_size:]:
            total_count = [sum(col) for col in zip(total_count, count)]
        return total_count

    def get_where_columns(self, query):
        columns = list()
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
                    for column in self.columns:
                        if str(column).upper() in str(where).upper():
                            columns.append(column)
        return list(set(columns))
    
    """
        Initialization methods
    """

    def initialize_window(self, window_size):
        workload_buffer = list()
        column_count = list()
        index_count = list()
        cost_history = list()

        for query in random.choices(self.workload):
            workload_buffer.append(query)
            cost_history.append(self.db.get_query_cost(query))
            # column_count.append(self.get_column_count(query))
            index_count.append(self.get_index_count(query))

        return workload_buffer, column_count, index_count, cost_history
    
    def get_state(self):
        if self.allow_columns:
            indexes = np.array(list(self.db.get_indexes().values()))
            # current_count = np.array(self.get_column_count_window())
            current_count = np.array(self.get_index_count_window())
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
    pprint(env.columns)