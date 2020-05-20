from database import Database
import numpy as np

class Environment():

    """
    step()

    apply_action() - create/drop index and return new state

    compute_reward() - fetch cost of executing last query?

    reset() - reset indexes in the database

    close() - close the database connection and reset indexes
    """

    def __init__(self):
        # Database instance
        self.db = Database()

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
        pass
        # Apply action
        s_prime = self.apply_action(action)

        # Compute current state reward
        reward = self.compute_reward()

        return s_prime, reward, False, dict()

    def apply_action(self, action):
        # APPLY NEW QUERY TRANSITION TO NEXT STATE
        return np.array([0])

    def compute_reward(self):
        return 0.0

    def reset(self):
        return self.db.reset_indexes()
    
    def close(self):
        return self.db.close_connection()
