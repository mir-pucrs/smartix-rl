import throughput
import power_size


class Metrics:

    def __init__(self):
        self.temp_val = 0
        self.geoMean = 0
        self.powerSize = 0


def calc_throughput():
    stream = 2
    global temp_val
    res_time_execution = throughput.comp_time_execution()
    temp_val = sum(res_time_execution.values())
    print("Tempo total de execução: ", temp_val)
    total = ((stream * 2) / temp_val) * 3600 * 1
    print("Throughput ", total)


def calc_power_size():
    global geoMean
    global powerSize
    time_power_size = power_size.comp_time_power_size()
    count = time_power_size[1]
    total_time = time_power_size[1]
    geoMean = (total_time**(1/count))
    powerSize = (3600/geoMean) * 1
    print("\nTotal: ", total_time, "\nGeometric mean: ", geoMean, "\nSize: ", count, "\nPower Size: ", powerSize)


def main():
    throughput.run_functions()
    power_size.run_functions()
    calc_throughput()
    calc_power_size()


if __name__ == '__main__':
    main()
