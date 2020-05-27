from utils import Random, LargeInteger, Infinity

import numpy as np


class MOVES:
    PURE, LEGAL, SMART, NUM_LEVELS = range(4)


class Knowledge():
    RolloutLevel = MOVES.LEGAL
    TreeLevel = MOVES.LEGAL
    def __init__(self):
        self.SmartTreeCount = 10
        self.SmartTreeValue = 1.0

    def Level(self, phase):
        assert(phase < PHASE.NUM_PHASES)
        if phase == PHASE.TREE:
            return self.TreeLevel
        else:
            return self.RolloutLevel


class PHASE:
    TREE, ROLLOUT, NUM_PHASES, MCC = range(4)


class PARTICLES:
    CONSISTENT, INCONSISTENT, RESAMPLED, OUT_OF_PARTICLES = range(4)


class Status:
    Phase = PHASE.TREE
    Particles = PARTICLES.CONSISTENT


class Simulator:
    def __init__(self, actions, observations, discount=1.0):
        self.NumActions = actions
        self.NumObservations = observations
        self.Discount = discount
        self.Knowledge = Knowledge()
        self.RewardRange = 100

        assert(self.Discount > 0 and self.Discount <= 1.0)

    def Validate(self, state):
        return True

    def LocalMove(self, state, history, stepObs, status):
        return 1

    def GenerateLegal(self, state, history, actions, status):
        for a in range(self.NumActions):
            actions.append(a)
        return actions

    def GeneratePreferred(self, state, history, actions, status):
        return actions

    def SelectRandom(self, state, history, status):
        if self.Knowledge.RolloutLevel >= MOVES.SMART:
            actions = []
            actions = self.GeneratePreferred(state, history, actions, status)
            if actions:
                return actions[Random(0, len(actions))]
        if self.Knowledge.RolloutLevel >= MOVES.LEGAL:
            actions = []
            actions = self.GenerateLegal(state, history, actions, status)
            if actions:
                return actions[Random(0, len(actions))]

        return Random(0, self.NumActions)

    def Prior(self, state, history, vnode, status):
        actions = []
        if self.Knowledge.TreeLevel == MOVES.PURE:
            vnode.SetChildren(0, 0)
        else:
            vnode.SetChildren(LargeInteger, -Infinity)

        if self.Knowledge.TreeLevel >= MOVES.LEGAL:
            actions = []
            actions = self.GenerateLegal(state, history, actions, status)

            for action in actions:
                qnode = vnode.Child(action)
                qnode.Value.Set(0, 0)
                qnode.AMAF.Set(0, 0)
                vnode.Children[action] = qnode

        if self.Knowledge.TreeLevel >= MOVES.SMART:
            actions = []
            actions = self.GeneratePreferred(state, history, actions, status)

            if actions:
                for action in actions:
                    qnode = vnode.Child(action)
                    qnode.Value.Set(self.Knowledge.SmartTreeCount, self.Knowledge.SmartTreeValue)
                    qnode.AMAF.Set(self.Knowledge.SmartTreeCount, self.Knowledge.SmartTreeValue)
                    vnode.Children[action] = qnode

        return vnode

#---- Displays
    def DisplayBeliefs(self, beliefState):
        return

    def DisplayState(self, state):
        return

    def DisplayAction(self, state, action):
        print("Action ", action)

    def DisplayObservation(self, state, observation):
        print("Observation ", observation)

    def DisplayReward(self, reward):
        print("Reward ", reward)

#--- Accessors
    def SetKnowledge(self, knowledge):
        self.Knowledge = knowledge

    def GetNumActions(self):
        return self.NumActions

    def GetNumObservations(self):
        return self.NumObservations

    def GetDiscount(self):
        return self.Discount

    def GetHyperbolicDiscount(self, t, kappa=0.277, sigma=1.0):
        return 1.0/np.power((1.0 + kappa*float(t)), sigma)

    def GetRewardRange(self):
        return self.RewardRange

    def GetHorizon(self, accuracy, undiscountedHorizon):
        if self.Discount == 1:
            return undiscountedHorizon
        return np.log(accuracy) / np.log(self.Discount)