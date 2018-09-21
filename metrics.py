from throughput import Throughput
from power_size import PowerSize
from environment import Environment
from state import State
from action import Action
from agent import Agent


class Metrics:

    def __init__(self):
        self.temp_val = 0
        self.geoMean = 0
        self.powerSize = 0
        self.state = State()
        self.env = Environment()
        self.tput = Throughput()
        self.powerSize = PowerSize()
        self.agent = Agent()
        self.action = Action()

    def calc_throughput(self):
        stream = 2
        global temp_val
        res_time_execution = self.tput.comp_time_execution()
        temp_val = sum(res_time_execution.values())
        print("Tempo total de execução: ", temp_val)
        total = ((stream * 2) / temp_val) * 3600 * 1
        print("Throughput ", total)

    def get_current_state(self):
        current_state = self.state.get_indexes()
        print(current_state)

    def calc_power_size(self):
        global geoMean
        global powerSize
        time_power_size = self.powerSize.comp_time_power_size()
        count = time_power_size[1]
        total_time = time_power_size[1]
        geoMean = (total_time**(1/count))
        powerSize = (3600/geoMean) * 1
        print("\nTotal: ", total_time, "\nGeometric mean: ", geoMean, "\nSize: ", count, "\nPower Size: ", powerSize)

    def run(self):
        # self.tput.run_functions()
        # self.powerSize.run_functions()
        # self.calc_throughput()
        # self.calc_power_size()
        # self.state.get_indexes()
        # self.action.add_index('l_comment')
        # self.action.drop_index('index1')
        # self.agent.reset()
        # self.agent.get_columns_of_queries()
        self.agent.reset_map_indexes()
        # self.agent.train()


if __name__ == '__main__':
    Metrics().run()
