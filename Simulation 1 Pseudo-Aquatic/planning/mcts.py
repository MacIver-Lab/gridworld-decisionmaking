from statistics import STATISTICS
from history import History
from simulator import Status, PHASE
from utils import Random, Infinity, LargeInteger
from node import VNode, QNode
from beliefstate import BeliefState
from testsimulator import TestSimulator

import numpy as np

class SearchParams:
    Verbose = 0
    MaxDepth = 100
    NumSimulations = 1000
    NumStartState = 1000
    UseTransforms = True
    NumTransforms = 0
    MaxAttempts = 0
    ExpandCount = 1
    ExplorationConstant = 1
    UseRave = False
    RaveDiscount = 1.0
    RaveConstant = 0.01
    DisableTree = False
    Softmax = False

class MCTS:
    UCB_N = 10000
    UCB_n = 100
    UCB = [[0] * UCB_n] * UCB_N
    InitialisedFastUCB = False
    def __init__(self, simulator):
        self.Simulator = simulator
        self.History = History()
        self.Status = Status()

        self.TreeDepth = 0
        self.tau = 0
        self.PeakTreeDepth = 0

        self.StatTreeDepth = STATISTICS(0, 0)
        self.StatRolloutDepth = STATISTICS(0, 0)
        self.StatTotalReward = STATISTICS(0, 0)

        VNode.NumChildren = self.Simulator.GetNumActions()
        QNode.NumChildren = self.Simulator.GetNumObservations()

        self.Root = self.ExpandNode(self.Simulator.CreateRandomStartState())

        for i in range(0, SearchParams.NumStartState):
            self.Root.BeliefState.AddSample(self.Simulator.CreateRandomStartState())

 #----- Utility functions
    def BeliefState(self):
        return self.Root.Beliefs()

    def GetHistory(self):
        return self.History

    def GetStatus(self):
        return self.Status

    def ClearStatistics(self):
        self.StatTreeDepth.Clear()
        self.StatRolloutDepth.Clear()
        self.StatTotalReward.Clear()
#------

    def ExpandNode(self, state):
        vnode = VNode().Create()
        vnode.Value.Set(0, 0)

        vnode = self.Simulator.Prior(state, self.History, vnode, self.Status)
        return vnode

    def AddSample(self, node, state):
        sample = self.Simulator.Copy(state)
        node.BeliefState.AddSample(sample)
        return node

    @classmethod
    def InitFastUCB(self, exploration):
        if SearchParams.Verbose:
            print("Initialising fast UCB table...")
        for N in range(self.UCB_N):
            for n in range(self.UCB_n):
                if n == 0:
                    self.UCB[N][n] = Infinity
                else:
                    self.UCB[N][n] = exploration * np.sqrt(np.log(N + 1)/n)
        if SearchParams.Verbose:
            print("done")
        self.InitialisedFastUCB = True

    def Update(self, action, observation, reward):
        self.History.Add(action, observation)
        qnode = self.Root.Child(action)
        vnode = qnode.Child(observation)

        beliefs = BeliefState()

        if vnode:
            beliefs.Copy(vnode.BeliefState.Samples, self.Simulator)

        if SearchParams.UseTransforms:
            beliefs = self.AddTransforms(self.Root, beliefs)

        if vnode:
            if not beliefs.Samples and not vnode.BeliefState:
                return False, None

        if not vnode and not beliefs.Samples:
            return False, None

        if SearchParams.Verbose:
            self.Simulator.DisplayBeliefs(beliefs)

        state = 0
        if vnode and vnode.BeliefState.Samples:
            state = vnode.BeliefState.GetSample(0)
        else:
            state = beliefs.GetSample(0)

        newRoot = self.ExpandNode(state)
        newRoot.BeliefState = beliefs
        self.Root = newRoot

        return True, state

    def AddTransforms(self, root, beliefs):
        attempts = 0
        added = 0

        while added < SearchParams.NumTransforms and attempts < SearchParams.MaxAttempts:
            transform = self.CreateTransform()
            if transform:
                beliefs.AddSample(transform)
                added += 1
            attempts += 1

        if SearchParams.Verbose:
            print("Created ", added, " local transformations out of ", attempts, " attempts")

        return beliefs

    def CreateTransform(self):
        state = self.Root.BeliefState.CreateSample(self.Simulator)
        terminal, state, stepObs, stepReward = self.Simulator.Step(state, self.History.Back().Action)
        if self.Simulator.LocalMove(state, self.History, stepObs, self.Status):
            return state
        self.Simulator.FreeState(state)

    def FastUCB(self, N, n, logN):
        if self.InitialisedFastUCB and N < self.UCB_N and n < self.UCB_n:
            return self.UCB[int(N)][int(n)]

        if n == 0:
            return Infinity
        else:
            return SearchParams.ExplorationConstant * np.sqrt(logN / n)

    def SelectAction(self, state):
        self.UCTSearch()
        return self.GreedyUCB(self.Root, False, softmax=SearchParams.Softmax)

    def Rollout(self, state):
        self.Status.Phase = PHASE.ROLLOUT
        if SearchParams.Verbose:
            print("Starting rollout")

        totalReward = 0.0
        discount = 1.0
        #discount = self.Simulator.GetHyperbolicDiscount(0)
        terminal = False
        numSteps = 0

        while numSteps + self.TreeDepth < SearchParams.MaxDepth and not terminal:
            action = self.Simulator.SelectRandom(state, self.History, self.Status)
            terminal, state, observation, reward = self.Simulator.Step(state, action)

            if SearchParams.Verbose:
                self.Simulator.DisplayState(state)

            self.History.Add(action, observation, state=self.Simulator.Copy(state))

            totalReward += reward*discount
            discount *= self.Simulator.GetDiscount()
            #discount = self.Simulator.GetHyperbolicDiscount(numSteps + self.TreeDepth)
            numSteps += 1
            self.tau += numSteps

        self.StatRolloutDepth.Add(numSteps)
        if SearchParams.Verbose:
            print("Ending rollout after " + str(numSteps) + " steps, with total reward " + str(totalReward))

        return totalReward

    def UCTSearch(self):
        self.ClearStatistics()
        historyDepth = self.History.Size()
        for n in range(SearchParams.NumSimulations):
            state = self.Root.BeliefState.CreateSample(self.Simulator)
            self.Simulator.Validate(state)
            self.Status.Phase = PHASE.TREE

            if SearchParams.Verbose:
                print("Starting simulation")
                self.Simulator.DisplayState(state)

            self.TreeDepth = 0
            self.PeakTreeDepth = 0
            vnode = self.Root
            totalReward, vnode = self.SimulateV(state, vnode)
            self.Root = vnode

            self.StatTotalReward.Add(totalReward)
            self.StatTreeDepth.Add(self.PeakTreeDepth)

            if SearchParams.Verbose:
                print("Total Reward: ", self.StatTotalReward.Value)
                #self.DisplayValue()

            self.History.Truncate(historyDepth)

    def SimulateV(self, state, vnode):
        action = self.GreedyUCB(vnode, True)

        self.PeakTreeDepth = self.TreeDepth
        if (self.TreeDepth >= SearchParams.MaxDepth):
            return 0.0, vnode

        if self.TreeDepth == 1:
            vnode = self.AddSample(vnode, state)

        qnode = vnode.Child(action)

        totalReward, qnode = self.SimulateQ(state, qnode, action)
        vnode.Children[action] = qnode
        vnode.Value.Add(totalReward)
        vnode = self.AddRave(vnode, totalReward)

        return totalReward, vnode

    def SimulateQ(self, state, qnode, action):
        delayedReward = 0.0

        terminal, state, observation, immediateReward = \
            self.Simulator.Step(state, action)
        assert(observation >= 0 and observation < self.Simulator.GetNumObservations())
        self.History.Add(action, observation, state=self.Simulator.Copy(state))

        if SearchParams.Verbose:
            self.Simulator.DisplayState(state)

        vnode = qnode.Child(observation)
        if not vnode and not terminal and qnode.Value.GetCount() >= SearchParams.ExpandCount:
            vnode = self.ExpandNode(state)

        if not terminal:
            self.TreeDepth += 1
            self.tau += 1
            if vnode:
                delayedReward, vnode = self.SimulateV(state, vnode)
                qnode.Children[observation] = vnode
            else:
                delayedReward = self.Rollout(state)
            self.tau -= 1
            self.TreeDepth -= 1

        totalReward = immediateReward + self.Simulator.GetDiscount()*delayedReward
        #totalReward = immediateReward + self.Simulator.GetHyperbolicDiscount(self.tau + 1.0)
        qnode.Value.Add(totalReward)
        return totalReward, qnode

    def AddRave(self, vnode, totalReward):
        totalDiscount = 1.0
        for t in range(self.TreeDepth, self.History.Size()):
            qnode = vnode.Child(self.History[t].Action)
            if qnode:
                qnode.AMAF.Add(totalReward, totalDiscount)
                vnode.Children[self.History[t].Action] = qnode
                totalDiscount *= SearchParams.RaveDiscount

        return vnode

    def GreedyUCB(self, vnode, ucb, softmax=False):
        besta = []
        bestq = -Infinity
        beta = 1.0/3.0

        N = vnode.Value.GetCount()
        logN = np.log(N +1)

        qValues = []
        for action in range(self.Simulator.NumActions):
            qnode = vnode.Child(action)
            if qnode:
                q = qnode.Value.GetValue()
                n = qnode.Value.GetCount()

                if SearchParams.UseRave and qnode.AMAF.GetCount() > 0:
                    n2 = qnode.AMAF.GetCount()
                    beta = n2 / (n + n2 + SearchParams.RaveConstant*n*n2)
                    q = (1.0 - beta)*q + beta*qnode.AMAF.GetValue()

                if ucb:
                    q += self.FastUCB(N, n, logN)

                if q >= bestq:
                    if q > bestq:
                        besta = []
                    bestq = q
                    besta.append(action)

                qValues.append(q)
        assert(besta)

        if softmax:
            tempQ = []
            indices = []
            for i, qValue in enumerate(qValues):
                if qValue > -1*LargeInteger:
                    tempQ.append(qValue)
                    indices.append(i)

            qValues = np.array(tempQ, dtype=np.float64)
            logsoftmax = qValues - np.log(np.sum(np.exp(qValues * beta), axis=0))
            besta = [indices[np.argmax(logsoftmax, axis=0)]]

        return besta[Random(0, len(besta))]

# ----- Display Functions

    def DisplayStatistics(self):
        print("Tree Depth: ", self.StatTreeDepth)
        print("Rollout Depth: ", self.StatRolloutDepth)
        print("Total Reward: ", self.StatTotalReward)

        print("Policy after ", SearchParams.NumSimulations, " simulations")
        self.DisplayPolicy(6)
        print("Values after ", SearchParams.NumSimulations, " simulations")
        self.DisplayValue(6)

    def DisplayPolicy(self, depth):
        print("MCTS Policy: ")
        self.Root.VDisplayPolicy(self.History, depth)

    def DisplayValue(self, depth):
        print("MCTS Value: ")
        self.Root.VDisplayValue(self.History, depth)

# ---- Tests

def UnitTestMCTS():
    UnitTestGreedy()
    UnitTestUCB()
    UnitTestRollout()

    for depth in range(1, 4):
       UnitTestSearch(depth)

def UnitTestGreedy():
    testSimulator = TestSimulator(5, 5, 0)
    mcts = MCTS(testSimulator)
    numAct = testSimulator.GetNumActions()
    numObs = testSimulator.GetNumObservations()

    vnode = mcts.ExpandNode(testSimulator.CreateStartState())
    vnode.Value.Set(1, 0)
    vnode.Children[0].Value.Set(1, 1)

    for action in range(1, numAct):
        vnode.Child(action).Value.Set(0, 0)
    actionChosen = mcts.GreedyUCB(vnode, False)
    assert(actionChosen == 0)

def UnitTestUCB():
    testSimulator = TestSimulator(5, 5, 0)
    mcts = MCTS(testSimulator)

    numAct = testSimulator.GetNumActions()
    numObs = testSimulator.GetNumObservations()

    vnode1 = mcts.ExpandNode(testSimulator.CreateStartState())
    vnode1.Value.Set(1, 0)
    for action in range(0, numAct):
        if action == 3:
            vnode1.Child(action).Value.Set(99, 0)
        else:
            vnode1.Child(action).Value.Set(100+action, 0)
    actionChosen = mcts.GreedyUCB(vnode1, True)
    assert(actionChosen == 3)

    vnode2 = mcts.ExpandNode(testSimulator.CreateStartState())
    vnode2.Value.Set(1, 0)
    for action in range(numAct):
        if action == 3:
            vnode2.Child(action).Value.Set(99+numObs, 1)
        else:
            vnode2.Child(action).Value.Set(100+numAct - action, 0)
    actionChosen = mcts.GreedyUCB(vnode2, True)
    assert (actionChosen == 3)

    vnode3 = mcts.ExpandNode(testSimulator.CreateStartState())
    vnode3.Value.Set(1, 0)
    for action in range(numAct):
        if action == 3:
            vnode3.Child(action).Value.Set(1, 1)
        else:
            vnode3.Child(action).Value.Set(100+action, 1)
    actionChosen = mcts.GreedyUCB(vnode3, True)
    assert (actionChosen == 3)

    vnode4 = mcts.ExpandNode(testSimulator.CreateStartState())
    vnode4.Value.Set(1, 0)
    for action in range(numAct):
        if action == 3:
            vnode4.Child(action).Value.Set(0, 0)
        else:
            vnode4.Child(action).Value.Set(1, 1)
    actionChosen = mcts.GreedyUCB(vnode4, True)
    assert (actionChosen == 3)

def UnitTestRollout():
    testSimulator = TestSimulator(2, 2, 10)
    mcts = MCTS(testSimulator)

    SearchParams.NumSimulations = 1000
    SearchParams.MaxDepth = 10
    totalReward = 0.0

    for n in range(SearchParams.NumSimulations):
        state = testSimulator.CreateStartState()
        mcts.TreeDepth = 0
        totalReward += mcts.Rollout(state)

    rootValue = totalReward / SearchParams.NumSimulations
    meanValue = testSimulator.MeanValue()

    assert(abs(meanValue - rootValue) < 0.2)

def UnitTestSearch(depth):
    testSimulator = TestSimulator(3, 2, depth)
    mcts = MCTS(testSimulator)
    SearchParams.MaxDepth = depth + 1
    SearchParams.NumSimulations = 10**(depth+1)

    mcts.UCTSearch()
    rootValue = mcts.Root.Value.GetValue()
    optimalValue = testSimulator.OptimalValue()
    assert(abs(optimalValue - rootValue) < 0.5)


if __name__ == "__main__":
    UnitTestMCTS()