import query_execution


class Metrics:

    def __init__(self):
        self.temp_val = 0


def calc_throughput():
    stream = 2
    global temp_val
    res_time_execution = query_execution.comp_time_execution()
    temp_val = sum(res_time_execution.values())
    print("Tempo total de execução: ", temp_val)
    total = ((stream * 2) / temp_val) * 3600 * 1
    print("Throughput ", total)


def main():
    query_execution.run_functions()
    calc_throughput()


if __name__ == '__main__':
    main()
