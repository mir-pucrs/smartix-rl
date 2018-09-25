import mysql.connector
from state import State

class Environment:

    def __init__(self):
        self.reward = {}
        self.state = State()

    def execute(self, action):
        # if the agent is at state bla do bla
        indexes = self.state.get_map_indexes()

        """
        TODO: implement function to return state and state_reward
        """

    def run_benchmark(self):
        """
        TODO: implement function to run benchmark
        """

    def comp_rewards(self):
        """
        TODO: implement function to compute rewards
        """