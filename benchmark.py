from TPCH import TPCH


class Benchmark:


    def __init__(self, benchmark = "TPCH"):
        self.benchmark = benchmark

    def run(self):
        if (self.benchmark == "TPCH"):
            results = TPCH().run()
        else:
            raise "No corresponding benchmark"
        
        return results