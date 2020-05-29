from collections import defaultdict
from gamePRQL import Game

import numpy as np

class PRQLearningAgent:
    def __init__(self, policies, numActions=4, visualrange=1):
        self.Actions = range(numActions)
        self.LearningRate = 0.01
        self.TauRate = 0.001
        self.Discount = 0.95

        self.PolicyLibrary = policies

        self.Tau = 0.0
        self.Weights = [0.0 for policy in self.PolicyLibrary]
        self.U = [0.0 for policy in self.PolicyLibrary]

        self.QTable = defaultdict(lambda: [0.0 for a in self.Actions])

        self.Verbose = 0
        self.N = 100
        self.VisualRange = visualrange

    def Clear(self):
        self.Tau = 0.0
        self.Weights = [0.0 for policy in self.PolicyLibrary]
        self.U = [0.0 for policy in self.PolicyLibrary]
        self.TauRate = 0.001 * (10 * int(self.VisualRange > 2))
        self.N = 100 + (10 * int(self.VisualRange > 3))

        self.QTable = defaultdict(lambda: [0.0 for a in self.Actions])

    def BoltzmanSelection(self, tau, x):
        numerator = np.exp(tau * np.asarray(x))
        return numerator / numerator.sum(axis=0)

    def ChoosePolicy(self):
        probs = self.BoltzmanSelection(self.Tau, self.Weights)
        return np.random.choice(np.arange(0, len(self.PolicyLibrary)), p=probs)

    def PolicyReuse(self, game, startState, policy):
        state = game.Copy(startState)
        discount = 1.0
        discountedReturn = 0.0
        if self.Verbose:
            print("Agent home location:%s"%state.AgentPos + " Predator home location:%s"%state.PredatorPos)

        currentState = game.Copy(startState)

        for t in range(200):
            if t >= len(policy):
                return discountedReturn, self.QTable, reward
            else:
                chosenAction = policy[t]

            terminal, nextState, observation, reward = game.Step(state, chosenAction)

            if self.Verbose:
                print("Agent chooses action: %d" % chosenAction)
                print("Agent moves to state:%s " % nextState.GetAgent() +
                      "and receives reward:%d " % reward +
                      "Predator moves to:%s" % nextState.GetPredator())

            self.Learn(game.Copy(state), chosenAction, reward, nextState)

            state = game.Copy(nextState)

            discountedReturn += discount * reward
            discount *= self.Discount

            if terminal:
                return discountedReturn, self.QTable, reward

            currentState = game.Copy(state)

    def Learn(self, state, action, reward, nextState):
        currentQ = self.QTable[state.GetState()][action]
        newQ = reward + self.Discount * max(self.QTable[nextState.GetState()])
        self.QTable[state.GetState()][action] += self.LearningRate * (newQ - currentQ)

    @staticmethod
    def argmax(state_action):
        maxIndexList = []
        maxValue = state_action[0]
        for index, value in enumerate(state_action):
            if value > maxValue:
                maxIndexList.clear()
                maxValue = value
                maxIndexList.append(index)
            elif value == maxValue:
                maxIndexList.append(index)
        return np.random.choice(maxIndexList)

    def habitSimulation(self, predatorPosition, xsize=15, ysize=15):
        habitGame = Game(xsize, ysize, visualrange=self.VisualRange)
        habitGame.PredatorHome = predatorPosition
        habitGame.State.PredatorPos = predatorPosition

        survivalRate = []
        weights = np.full(len(self.PolicyLibrary), np.nan)

        for trialInd in range(3):
            self.Clear()

            rewardList = []
            terminalRewards = []
            chosenPolicyIndices = []

            for k in range(self.N):

                policyIndex = self.ChoosePolicy()
                policy = self.PolicyLibrary[policyIndex]

                chosenPolicyIndices.append(policyIndex)
                try:
                    discounted, qTable, reward = self.PolicyReuse(habitGame, habitGame.State, policy)
                except TypeError:  # If an error occurs repeat trial
                    k -= 1
                    continue

                if k > self.N - 101:
                    rewardList.append(discounted)
                    terminalRewards.append(reward)

                if reward > 0:
                    self.Weights[policyIndex] = (self.Weights[policyIndex] * self.U[policyIndex] + discounted) \
                                                / (self.U[policyIndex] + 1.0)
                    self.U[policyIndex] += 1.0
                    self.Tau += self.TauRate

            weights = np.nanmean(np.vstack((weights, self.Weights)), axis=0)
            survivalRate.append(float(sum(reward > 0 for reward in terminalRewards)) / float(len(terminalRewards)))

        return np.max(survivalRate)




