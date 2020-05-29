from collections import defaultdict
from coord import Opposite
import gmatch4py as gm
from itertools import combinations
import networkx as nx

import numpy as np

class PRQL:
    def __init__(self, simulator, policies=None):
        self.Simulator = simulator
        self.Actions = self.Simulator.GetNumActions()
        self.Discount = self.Simulator.GetDiscount()

        self.PolicyLibrary = policies

        self.Tau = 0.0
        if policies:
            self.Weights = np.zeros(len(self.PolicyLibrary))
            self.U = np.zeros(len(self.PolicyLibrary))

        self.LearningRate = 0.01
        self.TauRate = 0.001

        self.QTable = defaultdict(lambda: [0.0 for a in self.Actions])

        self.Verbose = False
        self.N = 50

        self.PolicyIndex = None

    def Clear(self):
        self.Tau = 0.0
        self.Weights = np.zeros(len(self.PolicyLibrary))
        self.U = np.zeros(len(self.PolicyLibrary))

        self.QTable = defaultdict(lambda: [0.0 for a in self.Actions])

    def BoltzmanSelection(self, tau, x):
        numerator = np.exp(tau * np.asarray(x))
        return numerator / numerator.sum(axis=0)

    def ChoosePolicy(self):
        probs = self.BoltzmanSelection(self.Tau, self.Weights)
        return np.random.choice(np.arange(0, len(self.PolicyLibrary)), p=probs)

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

    def Update(self, discounted):
        self.Weights[self.PolicyIndex] = (self.Weights[self.PolicyIndex] * self.U[self.PolicyIndex] + discounted) \
                                    / (self.U[self.PolicyIndex] + 1.0)
        self.U[self.PolicyIndex] += 1.0
        self.Tau += self.TauRate

    def RemoveDuplicates(self, policyLibrary):
        newPolicyLibrary = []
        for policyIndex, policy in enumerate(policyLibrary):
            newpolicy = []
            ind = 0
            while ind < len(policy):
                action = policy[ind]
                if ind == 0 or ind == len(policy) - 1:
                    newpolicy.append(action)
                    ind += 1
                    continue

                prevAction = policy[ind - 1]
                nextAction = policy[ind + 1]
                if nextAction == Opposite(action) and prevAction == Opposite(action):
                    ind += 1
                    continue
                newpolicy.append(action)
                ind += 1
            newPolicyLibrary.append(newpolicy)
        return newPolicyLibrary

    def generateTrajectoryMap(self, policy, game, xsize=15, ysize=15):
        environment = np.zeros((xsize, ysize))
        agentPos = (game.AgentHome).Copy()
        environment[agentPos.Y, agentPos.X] = 1
        for action in policy:
            newAgentPos = game.NextPos(agentPos, action)
            if newAgentPos.Valid():
                environment[newAgentPos.Y, newAgentPos.X] = 1
                agentPos = newAgentPos
        return environment

    @staticmethod
    def getLatticeAdjacency(xsize, ysize):
        n = xsize * ysize
        adjacency = np.zeros((n, n))

        for r in range(ysize):
            for c in range(xsize):
                i = r * xsize + c
                if c > 0: adjacency[i - 1, i] = adjacency[i, i - 1] = 1
                if r > 0: adjacency[i - xsize, i] = adjacency[i, i - xsize] = 1

        return adjacency

    def policyGraph(self, policy, game, xsize=15, ysize=15):
        lattice_adj = self.getLatticeAdjacency(xsize, ysize)
        traj_adj = np.zeros((xsize * ysize, xsize * ysize))

        agentPos = (game.AgentHome).Copy()
        for action in policy:
            newAgentPos = game.NextPos(agentPos, action)
            if newAgentPos.Valid():
                node1 = agentPos.X + (agentPos.Y * xsize)
                node2 = newAgentPos.X + (newAgentPos.Y * xsize)
                traj_adj[node1, node2] = lattice_adj[node1, node2]
            agentPos = (newAgentPos).Copy()

        G = nx.from_numpy_matrix(traj_adj)

        return G

    def gDistance(self, successTrajectoryIndices):
        graphDistances = []
        if successTrajectoryIndices.any():
           if len(successTrajectoryIndices) == 1:
               return [0.0]
           else:
               successTrajectoryIndexPermutations = list(combinations(successTrajectoryIndices, 2))
               count = 0
               for index in successTrajectoryIndexPermutations:
                   G0 = self.policyGraph(self.PolicyLibrary[index[0]], self.Simulator)
                   G1 = self.policyGraph(self.PolicyLibrary[index[1]], self.Simulator)

                   minv = 0.0
                   ged = gm.GraphEditDistance(0.5, 0.5, 0.01, 0.01)
                   result = ged.compare([G0, G1], None)
                   minv = result[0, 1]

                   graphDistances.append(minv)
                   if count > 1000:
                       break
                   count += 1

               return graphDistances
