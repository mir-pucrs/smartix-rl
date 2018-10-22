from database import Database


class State:


    def __init__(self):
        # Database instance
        self.db = Database()

        # State attributes
        self.indexes_map = self.db.get_indexes_map()


    def __repr__(self):
        return str(self.indexes_map)
    
    
    def __hash__(self):
        return hash(str(self.indexes_map))

    
    def __eq__(self, other):
        return str(self) == str(other)



if __name__ == "__main__":
    state = State()
    print("\n\nIndexes map: \n", state.indexes_map)