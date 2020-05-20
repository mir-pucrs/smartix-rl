from database import Database
import numpy as np
import json

class Environment():

    """
    step()

    apply_action() - create/drop index and return new state

    compute_reward() - fetch cost of executing last query?

    reset() - reset indexes in the database

    close() - close the database connection and reset indexes
    """

    def __init__(self, workload_path='workload/tpch.sql'):
        # Database instance
        self.db = Database()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0

    def step(self, action):
        """
            Parameters:
            action: int

            Returns:
            s_prime: np.array shape (n_actions, )
            reward: float
            done: boolean
            info: empty dict
        """
        # Apply action
        s_prime, reward = self.apply_transition(action)

        return s_prime, reward, False, dict()

    def apply_transition(self, action):
        # Apply index change
        self.apply_index_change(action)

        # Execute next_query
        next_query = self.step_workload()

        # Compute reward
        reward = self.compute_reward(next_query)

        # Get current state
        s_prime = self.get_state()

        return s_prime, reward

    def compute_reward(self, query):
        cost = self.db.get_query_cost(query)
        reward = 1/cost
        return reward

    def get_state(self):
        indexes = list(self.db.get_indexes().values())
        state = np.array(indexes)
        return state

    def apply_index_change(self, action):
        indexes = self.db.get_indexes()
        for idx, column in enumerate(indexes):
            if idx == action:
                # Get the table
                for table in self.db.tables.keys():
                    if column in self.db.tables[table]:
                        if indexes[column] == 1:
                            self.db.drop_index(table, column)
                        else:
                            self.db.create_index(table, column)
                        break
                break

    def step_workload(self):
        query = self.workload[self.workload_iterator]
        self.db.execute(query, verbose=False)
        return query

    def load_workload(self, path):
        with open(path, 'r') as f:
            data = f.read()
        workload = data.split('\n')
        return workload

    def reset(self):
        self.workload_iterator = 0
        return self.db.reset_indexes()
    
    def close(self):
        return self.db.close_connection()


if __name__ == "__main__":
    from pprint import pprint
    env = Environment()

    # pprint(state)
    # pprint(env.db.get_indexes())