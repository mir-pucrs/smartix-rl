from TPCH import TPCH


class Benchmark:

    benchmark = "TPCH"

    def __init__(self):
        pass

    def run(self, benchmark):

        if (benchmark == "TPCH"):
            results = TPCH().run()
        else:
            raise "No corresponding benchmark"
        
        return results


'''
Running TPCH and getting results, for example...

if __name__ == '__main__':
    results = Benchmark().run("TPCH")
    print("Results:", results)

'''