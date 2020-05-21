from database import Database
import numpy as np

class Environment():
    def __init__(self, workload_path='workload/tpch.sql'):
        # Database instance
        self.db = Database()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0

        # Others
        self.n_features = len(self.get_state())
        self.n_actions = self.n_features * 2

    def step(self, action):
        """
        fazer recompensa como running mean das ultimas 10 queries
        fazer ultimas features do estado como sendo um contador das ultimas colunas q foram aparecendo
        """
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
        cost = self.db.get_query_cost(query)
        reward = (1/cost) * 1000000
        return reward

    def get_state(self):
        indexes = list(self.db.get_indexes().values())
        state = np.array(indexes)
        return state

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
        self.workload_iterator += 1
        return query

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

    pprint((env.get_state().shape))
