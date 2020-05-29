class Simulator:
    def __init__(self, actions, observations, discount=1.0):
        self.NumActions = actions
        self.NumObservations = observations
        self.Discount = discount

        assert(self.Discount > 0 and self.Discount <= 1.0)

    def Validate(self, state):
        return True

#--- Accessors
    def GetNumActions(self):
        return self.NumActions

    def GetNumObservations(self):
        return self.NumObservations

    def GetDiscount(self):
        return self.Discount