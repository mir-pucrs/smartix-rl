import mysql.connector
from state import State
from action_repr import Action_repr

class Environment:

    def __init__(self):
        self.reward = {}
        self.state = State()

    def execute(self, action):
        # if the agent is at state bla do bla
        indexes = self.state.get_map_indexes()
        if action.type == 'DROP':
            action.name
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

    def available_actions(self, state):
        list_actions = []
        for index in state.map_indexes.keys():
            a = Action_repr(index, 'DROP')
            a1 = Action_repr(index, 'CREATE')
            if state.map_indexes[index] == 0:
                list_actions.append(a1)
            else:
                list_actions.append(a)
        return list_actions