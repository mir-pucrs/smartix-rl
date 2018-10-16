from database import Database


class Action:


    def __init__(self, column, type):
        # Database instance
        self.db = Database()

        # Action attributes
        self.table = "lineitem"
        self.column = column
        self.type = type


    def __repr__(self):
        return str(self.column) + ',' + str(self.type)


    def __hash__(self):
        return hash(str(self))


    def execute(self):
        if self.type == 'DROP':
            self.db.drop_index(self.column, self.table)
        elif self.type == 'CREATE':
            self.db.create_index(self.column, self.table)



if __name__ == "__main__":
    action = Action("l_shipmode", "CREATE")
    action.execute()