class Q_Value:
	
    def __init__ (self, state, action):
        self.state = state
        self.action = action

    def __hash__ (self):
        return hash(str(self.state) + str(self.action))
    
    def __repr__(self):
    	return str(self.state) + str(self.action)

    def __eq__(self, other):
        return str(self) == str(other)