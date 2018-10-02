

class Action_repr():
    
    def __init__(self, name, type):
        self.name = name
        self.type = type
    
    def __repr__(self):
        return str(self.name) + ',' + str(self.type)
    
    def __hash__(self):
        return hash(str(self))