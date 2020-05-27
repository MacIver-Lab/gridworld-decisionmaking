from simulator import *

class TestState:
    def __init__(self):
        self.Depth = 0

class TestSimulator(Simulator):
    def __init__(self, actions, observations, maxDepth):
        Simulator.__init__(self, actions, observations, None, [])
        self.MaxDepth = maxDepth

    def Copy(self, state):
        newState = TestState()
        newState.Depth = state.Depth
        return newState

    def CreateStartState(self):
        return TestState()

    def CreateRandomStartState(self):
        return TestState()

    def FreeState(self, state):
        del state

    def Step(self, state, action):
        if action == 0 and state.Depth < self.MaxDepth:
            reward = 1.0
        else:
            reward = 0.0

        observation = Random(0, self.GetNumObservations())
        state.Depth += 1
        return False, state, observation, reward


    # def SimulatorStep(self, state, action):
    #     terminal, state, observation, reward = self.RealStep(state, action)
    #     return terminal, state, observation, reward

    def OptimalValue(self):
        discount = 1.0
        totalReward = 0.0
        for i in range(self.MaxDepth):
            totalReward += discount
            discount *= self.GetDiscount()

        return totalReward

    def MeanValue(self):
        discount = 1.0
        totalReward = 0.0
        for i in range(self.MaxDepth):
            totalReward += discount/self.GetNumActions()
            discount *= self.GetDiscount()
        return totalReward