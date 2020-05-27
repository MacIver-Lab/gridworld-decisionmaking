from simulator import Simulator
from grid import Grid
from coord import COORD, COMPASS, Compass, CompassString, Opposite, AggressiveDirectionalDistance
from utils import Random, Bernoulli, SetFlag

from math import floor
import numpy as np

class GameState:
    def __init__(self):
        self.AgentPos = COORD(5, 0)
        self.PredatorPos = COORD(0, 0)
        self.PredatorDir = -1
        self.PredatorSpeedMult = 2
        self.AgentObservationDirection = 0
        self.Depth = 0

    def __eq__(self, other):
        return self.AgentPos == other.AgentPos and \
               self.PredatorPos == other.PredatorPos

    def __str__(self):
        return "(" + str(self.AgentPos) + "; " + str(self.PredatorPos) + ")"

    def __hash__(self):
        return 0

class Game(Simulator):
    def __init__(self, xsize, ysize, visualRange=1, verbose=False):
        self.AgentHome = COORD(int(floor((xsize-1)/2)), 0)
        self.PredatorNum = 1
        self.PredatorHome = COORD(0, 0)
        self.ChaseProbability = 0.7
        self.MoveProbability = 0.5
        self.Discount = 0.95
        self.GoalPos = COORD(int(floor((xsize-1)/2)), ysize-1) #int(floor((xsize-1)/2))

        self.Grid = Grid(xsize, ysize)

        self.NumActions = 4
        self.VisualRange = visualRange
        self.VisionObservationBit = len(self.Grid.VisualArea(COORD(0, 0), 0, self.VisualRange))
        self.NumObservations = self.VisionObservationBit #1 << (self.VisionObservationBit)
        Simulator.__init__(self, self.NumActions, self.NumObservations, self.Discount)

        self.RewardClearLevel = 1000
        self.RewardDefault = -1
        self.RewardDie = -100
        self.RewardHitWall = -25
        self.RewardRange = 100

        self.State = GameState()
        self.State.AgentPos = self.AgentHome
        self.State.PredatorPos = self.PredatorHome

        visualArea = self.Grid.VisualArea(self.AgentHome, 0, self.VisualRange)
        invalidPredatorLocations = [self.GoalPos, self.AgentHome] + visualArea
        #invalidInsidePredatorLocations = [coord for coord in invalidPredatorLocations if self.Grid.Inside(coord)]

        allPredatorLocations = [COORD(x, y) for x in range(0, xsize) for y in range(0, ysize)]
        validPredatorLocations = list(set(allPredatorLocations) - set(invalidPredatorLocations))
        self.StartPredatorLocations = validPredatorLocations

        self.XSize = xsize
        self.YSize = ysize
        if verbose:
            #self.Display = Display(xsize, ysize, 'game')
            self.InitializeDisplay()

    def FreeState(self, state):
        del state

    def InitializeDisplay(self):
        state = self.CreateStartState()
        self.DisplayState(state)

    def Copy(self, state):
        newState = GameState()

        newState.AgentPos = state.AgentPos
        newState.PredatorPos = state.PredatorPos
        newState.PredatorDir = state.PredatorDir
        newState.PredatorSpeedMult = state.PredatorSpeedMult
        newState.Depth = state.Depth
        newState.AgentObservationDirection = state.AgentObservationDirection

        return newState

    def Validate(self, state):
        assert(self.Grid.Inside(state.AgentPos))
        assert(self.Grid.Inside(state.PredatorPos))

    def CreateStartState(self):
        state = GameState()
        state = self.NewLevel(state)
        return state

    def CreateRandomStartState(self):
        state = GameState()
        state = self.NewLevel(state)

        observation = np.zeros(4)
        for scan in range(0, 4):
            observation[scan] = self.MakeObservation(state, scan)
        if np.any(observation):
            return state

        state.PredatorPos = self.StartPredatorLocations[Random(0, len(self.StartPredatorLocations))]
        return state

    def NextPos(self, fromCoord, dir):
        nextPos = fromCoord + Compass[dir]
        if self.Grid.Inside(nextPos):
            return nextPos
        else:
            return Compass[COMPASS.NAA]

    def Step(self, state, action):
        if action < 4:
            state.AgentObservationDirection = action
        reward = self.RewardDefault

        newpos = self.NextPos(state.AgentPos, action)
        if newpos.Valid():
            state.AgentPos = newpos
        else:
            reward += self.RewardHitWall

        observation = 0
        hitPredator = 0
        if state.AgentPos == state.PredatorPos:
            hitPredator = 1

        previousPredatorLocation = state.PredatorPos
        state, hitPredator = self.MovePredator(state, Bernoulli(self.MoveProbability), previousPredatorLocation)

        observation = self.MakeObservation(state, state.AgentObservationDirection)

        if hitPredator:
            reward += self.RewardDie
            return True, state, observation, reward #Terminate death

        if state.AgentPos == self.GoalPos:
            reward += self.RewardClearLevel
            return True, state, observation, reward #Terminate goal state

        state.Depth += 1
        return False, state, observation, reward

    def MakeObservation(self, state, observationDirection):
        visualArea = self.Grid.VisualArea(state.AgentPos, observationDirection, self.VisualRange)

        observation = 0
        # Predator Observation
        for i, coord in enumerate(visualArea):
            if state.PredatorPos == coord:
                observation = i #SetFlag(observation, i)

        return observation

    def LocalMove(self, state, history, stepObs, status):
        state.PredatorPos = COORD(Random(0, self.Grid.GetXSize()),
                                  Random(0, self.Grid.GetYSize()))
        if history.Size() == 0:
            return True, state
        observation = self.MakeObservation(state, state.AgentObservationDirection)
        return history.Back().Observation == observation

    def MovePredator(self, state, move, previousPredatorLocation):
        if move:
            numberOfMoves = state.PredatorSpeedMult
        else:
            numberOfMoves = 1

        for i in range(0, numberOfMoves):
            if Bernoulli(self.ChaseProbability):
                state = self.MovePredatorAggressive(state, previousPredatorLocation)
            else:
                state = self.MovePredatorRandom(state, previousPredatorLocation)

            if state.AgentPos == state.PredatorPos:
                return state, (state.AgentPos == state.PredatorPos)

        return state, (state.AgentPos == state.PredatorPos)


    def MovePredatorRandom(self, state, previousPredatorLocation):
        copyState = self.Copy(state)
        predatorPos = copyState.PredatorPos
        while True:
            action = Random(0, 4)
            newpos = self.NextPos(predatorPos, action)
            if newpos.Valid():  # and action != Opposite(state.PredatorDir):
                break

        copyState.PredatorPos = newpos
        copyState.PredatorDir = action
        return copyState


    def MovePredatorAggressive(self, state, previousPredatorLocation):
        newState = self.Copy(state)
        bestDist = self.Grid.GetXSize() + self.Grid.GetYSize()
        bestPos = state.PredatorPos
        bestDir = -1

        agentPos = state.AgentPos
        predatorPos = state.PredatorPos

        for dir in range(0, 4):
            dist = AggressiveDirectionalDistance(agentPos, predatorPos, dir)
            newpos = self.NextPos(predatorPos, dir)
            if dist <= bestDist and newpos.Valid() and newpos != previousPredatorLocation:
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        newState.PredatorPos = bestPos
        newState.PredatorDir = bestDir

        return newState

    def NewLevel(self, state):
        state.AgentPos = self.AgentHome
        state.PredatorPos = self.PredatorHome
        state.PredatorDir = 2
        state.Depth = 0
        state.AgentObservationDirection = 0

        return state

    def GenerateLegal(self, state, history, legal, status):
        for a in range(self.NumActions):
            newpos = self.NextPos(state.AgentPos, a)
            if newpos.Valid() and newpos != state.PredatorPos:
                legal.append(a)

        return legal

    def GeneratePreferred(self, state, history, actions, status):
        if history.Size():
            for a in range(self.NumActions):
                newpos = self.NextPos(state.AgentPos, a)
                if newpos.Valid() and newpos != state.PredatorPos:
                    actions.append(a)

            if Opposite(history.Back().Action) in actions:
                actions.remove(Opposite(history.Back().Action))
            #actions = self.GenerateExplorationActions(state, history, actions, status)

            return actions

        else:
            return self.GenerateLegal(state, history, actions, status)

    def DisplayBeliefs(self, beliefState):
        counts = [0]*(self.Grid.GetYSize()*self.Grid.GetXSize())
        for i in range(beliefState.GetNumSamples()):
            state = beliefState.GetSample(i)
            predatorIndex = self.Grid.Index(state.PredatorPos.X, state.PredatorPos.Y)
            counts[predatorIndex] += 1

        counts = np.reshape(np.array(counts, dtype=np.float32), (self.Grid.GetXSize(), self.Grid.GetYSize()))
        for x in range(self.Grid.GetYSize()):
            for y in range(self.Grid.GetXSize()):
                counts[x, y] = float(counts[x, y]) / float(beliefState.GetNumSamples())

        print("Belief State: ", counts)

    def DisplayAction(self, state, action):
        print("Agent moves ", CompassString[action])


if __name__ == "__main__":
    state = GameState()
    game = Game(7, 7)
    game.Validate(state)
