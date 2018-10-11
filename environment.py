import random
from state import State
from benchmark import Benchmark
from action import Action
from throughput import Throughput
from power_size import PowerSize


class Environment:

    def __init__(self):
        self.reward = []
        self.state = State()
        # self.action = Action()
        self.benchmark = Benchmark()
        self.map_indexes = self.state.get_map_indexes()
        self.tput = Throughput()
        self.powerSize = PowerSize()

    def calc_throughput(self):
        stream = 2
        global temp_val
        res_time_execution = self.tput.comp_time_execution()
        temp_val = sum(res_time_execution.values())
        print("Tempo total de execução: ", temp_val)
        total = ((stream * 2) / temp_val) * 3600 * 1
        print("Throughput ", total)
        return total

    def get_current_state(self):
        current_state = self.state.get_indexes()
        return current_state

    def calc_power_size(self):
        global geoMean
        global powerSize
        time_power_size = self.powerSize.comp_time_power_size()
        count = time_power_size[1]
        total_time = time_power_size[1]
        geoMean = (total_time**(1/count))
        powerSize = (3600/geoMean) * 1
        print("\nTotal: ", total_time, "\nGeometric mean: ", geoMean, "\nSize: ", count, "\nPower Size: ", powerSize)
        return powerSize

    # def available_actions(self, state):
    #     list_actions = []
    #     state = self.state.get_map_indexes()
    #     print('test state: ', state)
    #     for index in state:
    #         a = Action(index, 'DROP')
    #         a1 = Action(index, 'CREATE')
    #         if 1 in state:
    #             list_actions.append(a)
    #         else:
    #             list_actions.append(a1)
    #     print('list of actions: ', list_actions)
    #     return list_actions
    def create_index(self, column):
        Action('CREATE', column).add_index(column)

    def drop_index(self, index):
        Action('CREATE', index).add_index(index)

    def available_actions(self, state):
        available_actions = list()
        columns_list = self.state.get_columns_of_table()
        for column in columns_list:
            if column in state:
                available_actions.append(Action(column, 'DROP'))
            else:
                available_actions.append(Action(column, 'CREATE'))
        # print('list of states', states_list)
        # for column in state.indexes_map.keys():
        #     print(state.indexes_map[column])
        #     if state.indexes_map[column] == 0:
        #         available_actions.append(Action(column, 'CREATE'))
        #     else:
        #         available_actions.append(Action(column, 'DROP'))
        print("\n\nAvailable actions:", available_actions, "\n\n")
        return available_actions

    def execute(self, state, action):
        for column in state.indexes_map.keys():
            if action == 'CREATE':
                Action(column, 'CREATE').add_index(column)
            else:
                Action(column, 'DROP').drop_index(column)
        return action
        # column = random.choice(self.state.get_columns_of_queries())
        # print('random column', column)
        # if the agent is at state bla do bla
        # make something here
        # result = self.benchmark.run('TPCH')
        # self.reward.append(result)
        # print('result of benchmark', result)

    def run_benchmark(self):
        return self.benchmark.run()


    def comp_rewards(self):
        """
        TODO: implement function to compute rewards
        """

    # def run(self):
    #     self.execute()

# if __name__ == '__main__':
#     Environment().execute()
