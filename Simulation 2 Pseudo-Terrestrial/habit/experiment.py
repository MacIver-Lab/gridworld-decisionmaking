from coord import COORD
from PolicyReuseQ import PRQL
from astar import AStar

import numpy as np

class Experiment:
    def __init__(self, real):
        self.Real = real
        self.ASTAR = AStar(self.Real.XSize, self.Real.YSize, self.Real.Occlusions)

        self.PRQL = PRQL(self.Real)

    def Run(self, policy):
        undiscountedReturn = 0.0
        discountedReturn = 0.0
        discount = 1.0

        state = self.Real.CreateStartState()
        currentState = self.Real.Copy(state)

        t = 0
        while True:
            try:
                action = policy[t]
            except IndexError:
                self.ASTAR.__init__(self.Real.XSize, self.Real.YSize, self.Real.Occlusions)
                self.ASTAR.InitializeGrid((state.AgentPos).Copy(), (self.Real.GoalPos).Copy())
                path = self.ASTAR.Solve()
                pathPos = COORD(path[1][0], path[1][1])
                for action in range(self.Real.NumActions):
                    agentPos = (state.AgentPos).Copy()
                    nextAgentPos = self.Real.NextPos(agentPos, action)
                    if nextAgentPos == pathPos:
                        break

            terminal, state, observation, reward = self.Real.Step(state, action)
            currentState = self.Real.Copy(state)

            undiscountedReturn += reward
            discountedReturn += reward * discount
            discount *= self.Real.GetDiscount()

            if terminal:
                return reward

            t += 1

        return None


    def DiscountedReturn(self, policies, predatorHome, occlusions):

        if not policies:
            return 0.0

        self.Real.__init__(self.Real.XSize, self.Real.YSize, occlusions=occlusions)
        self.Real.PredatorHome = predatorHome
        self.PRQL.__init__(self.Real, policies=policies)
        newPolicyLibrary = self.PRQL.RemoveDuplicates(policies)
        self.PRQL.PolicyLibrary = newPolicyLibrary
        survivalrates = 0

        trajGraphDistances = []

        for j in range(2):
            terminalRewards = []
            successTrajectoryIndices = []
            for i in range(self.PRQL.N):
                self.PRQL.PolicyIndex = self.PRQL.ChoosePolicy()
                policy = self.PRQL.PolicyLibrary[self.PRQL.PolicyIndex]

                terminalReward = self.Run(policy)

                if not terminalReward:
                    i -= 1
                    continue
                terminalRewards.append(terminalReward)

                if terminalReward > 0:
                    successTrajectoryIndices.append(self.PRQL.PolicyIndex)


            survivalrates += float(sum(reward > 0 for reward in terminalRewards))/float(len(terminalRewards))

            trajGraphDistances.append(self.PRQL.gDistance(successTrajectoryIndices))

        return survivalrates/2., np.mean(trajGraphDistances)

