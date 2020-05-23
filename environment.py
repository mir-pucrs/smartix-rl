from database import Database
import numpy as np
import sqlparse

class Environment():
    def __init__(self, workload_path='workload/tpch.sql'):
        # Database instance
        self.db = Database()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0

        # State and reward
        self.column_order = list(self.db.get_indexes().keys())
        self.column_count = list()
        self.window_size = 10
        self.cost_history = list()

        # Environment
        self.n_features = len(self.column_order) * 2
        self.n_actions = self.n_features

    def step(self, action):
        # Apply action
        s_prime, reward = self.apply_transition(action)
        return s_prime, reward, False, dict()

    def apply_transition(self, action):
        # Apply index change
        print("Applying index...")
        self.apply_index_change(action)

        # Execute next_query
        print("Executing query...")
        next_query = self.step_workload()

        # Compute reward
        reward = self.compute_reward(next_query)

        # Get current state
        s_prime = self.get_state()

        return s_prime, reward

    def compute_reward(self, query):
        self.cost_history.append(self.db.get_query_cost(query))
        # Get avg of last 10 costs
        if len(self.cost_history) >= self.window_size:
            cost = sum(self.cost_history[-self.window_size:])
            cost_avg = cost/self.window_size
        else:
            cost = sum(self.cost_history)
            cost_avg = cost/len(self.cost_history)
        reward = (1/cost_avg) * 1000000
        return reward

    def get_state(self):
        indexes = np.array(list(self.db.get_indexes().values()))
        current_count = np.array(self.get_column_count())
        state = np.concatenate((indexes, current_count))
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
        if action < self.n_actions/2: create = True
        for idx, column in enumerate(indexes):
            if idx == action:
                # Get the table
                for table in self.db.tables.keys():
                    if column in self.db.tables[table]:
                        if create:
                            self.db.create_index(table, column)
                        else:
                            self.db.drop_index(table, column)
                        break
                break

    def step_workload(self):
        query = self.workload[self.workload_iterator]
        self.db.execute(query, verbose=False)
        self.update_column_count(query)

        if self.workload_iterator+1 == len(self.workload):
            print("--- RESETTING WORKLOAD ITERATOR!!!!!")
            self.workload_iterator = 0
        else:
            self.workload_iterator += 1
        return query
    
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

    def load_workload(self, path):
        with open(path, 'r') as f:
            data = f.read()
        workload = data.split('\n')
        return workload

    def reset(self):
        self.workload_iterator = 0
        self.db.reset_indexes()
        return self.get_state()
    
    def close(self):
        self.workload_iterator = 0
        self.db.reset_indexes()
        return self.db.close_connection()


if __name__ == "__main__":
    from pprint import pprint
    env = Environment()