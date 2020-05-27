class History:
    def __init__(self):
        self.HistoryVector = []

    def Add(self, action, observation=-1, state=None):
        self.HistoryVector.append(ENTRY(action, observation, state))

    def GetVisitedStates(self):
        states = []
        if self.HistoryVector:
            for history in self.HistoryVector:
                if history.State:
                    states.append(history.State)

        return states

    def Pop(self):
        self.HistoryVector = self.HistoryVector[:-1]

    def Truncate(self, t):
        self.HistoryVector = self.HistoryVector[:t]

    def Clear(self):
        self.HistoryVector[:] = []

    def Forget(self, t):
        self.HistoryVector = self.HistoryVector[t:]

    def Size(self):
        return len(self.HistoryVector)

    def Back(self):
        assert(self.Size() > 0)
        return self.HistoryVector[-1]

    def __eq__(self, other):
        if(other.Size() != self.Size()):
            return False

        for i,history in enumerate(other):
            if (history.Action != self.HistoryVector[i].Action) or \
                    (history.Observation != self.HistoryVector[i].Observation):
                return False
        return True

    def __getitem__(self, t):
        assert(t>=0 and t<self.Size())
        return self.HistoryVector[t]

class ENTRY:
    def __init__(self, action, observation, state):
        self.Action = action
        self.Observation = observation
        self.State = state

    def __str__(self):
        return "(" + str(self.Action) + " , " + str(self.Observation) + ")"

if __name__ == "__main__":
    entry = ENTRY(1, 1, None)
    history = History()
    history.Add(1, 1)
    assert(history.Size() == 1)
    history.Add(2, 2)
    print(history)
    assert(History().Add(1, 1) == History().Add(1, 1))