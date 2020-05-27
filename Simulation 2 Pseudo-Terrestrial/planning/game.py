from simulator import Simulator
from grid import Grid
from coord import COORD, COMPASS, Compass, CompassString, Opposite, AggressiveDirectionalDistance
from utils import Random, Bernoulli, SetFlag

from math import floor
import numpy as np

class GameState:
    def __init__(self):
        self.AgentPos = COORD(5, 0)
        self.AgentDir = -1
        self.PredatorPos = COORD(0, 0)
        self.PredatorDir = -1
        self.PredatorSpeedMult = 2
        self.PredatorBeliefState = None
        self.Depth = 0

    def __eq__(self, other):
        return self.AgentPos == other.AgentPos and \
               self.PredatorPos == other.PredatorPos

    def __str__(self):
        return "(" + str(self.AgentPos) + "; " + str(self.PredatorPos) + ")"

    def __hash__(self):
        return 0

class Game(Simulator):
    def __init__(self, xsize, ysize, occlusions=[], visualrange=None, verbose=False):
        self.AgentHome = COORD(int(floor((xsize-1)/2)), 0)
        self.PredatorNum = 1
        self.PredatorHome = COORD(0, 0)
        self.ChaseProbability = 0.75
        self.MoveProbability = 0.5
        self.Discount = 0.95
        self.GoalPos = COORD(int(floor((xsize-1)/2)), ysize-1) #int(floor((xsize-1)/2))

        self.Grid = Grid(xsize, ysize)

        self.NumActions = 4
        self.VisualRange = visualrange
        self.Occlusions = occlusions

        if not visualrange:
            self.NumObservations = 2
        else:
            self.VisionObservationBit = len(self.Grid.VisualArea(COORD(0, 0), 0, self.VisualRange))
            self.NumObservations = self.VisionObservationBit #1 << (self.VisionObservationBit)
            visualArea = self.Grid.VisualArea(self.AgentHome, 0, self.VisualRange)

        Simulator.__init__(self, self.NumActions, self.NumObservations, self.GoalPos, occlusions, self.Discount)

        self.RewardClearLevel = 1000
        self.RewardDefault = -1
        self.RewardDie = -100
        self.RewardHitWall = -25
        self.RewardRange = 100

        self.State = GameState()
        self.State.AgentPos = self.AgentHome
        self.State.PredatorPos = self.PredatorHome

        self.XSize = xsize
        self.YSize = ysize
        if verbose:
            #self.Display = Display(xsize, ysize, 'game')
            self.InitializeDisplay()

    def GetValidPredatorLocations(self):
        allPredatorLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
        invalidPredatorLocations = [self.GoalPos, self.AgentHome] + self.Occlusions
        for predator in allPredatorLocations:
            if self.Grid.VisualRay((self.AgentHome).Copy(), predator.Copy(), self.Occlusions)[0]:
                invalidPredatorLocations.append(predator)
        self.StartPredatorLocations = list(set(allPredatorLocations) - set(invalidPredatorLocations))

    def FreeState(self, state):
        del state

    def InitializeDisplay(self):
        state = self.CreateStartState()
        self.DisplayState(state)

    def Copy(self, state):
        newState = GameState()

        newState.AgentPos = state.AgentPos
        newState.AgentDir = state.AgentDir
        newState.PredatorPos = state.PredatorPos
        newState.PredatorDir = state.PredatorDir
        newState.PredatorSpeedMult = state.PredatorSpeedMult
        newState.Depth = state.Depth
        newState.PredatorBeliefState = state.PredatorBeliefState

        return newState

    def Validate(self, state):
        assert(self.Grid.Inside(state.AgentPos))
        assert(self.Grid.Inside(state.PredatorPos))

    def CreateStartState(self):
        state = GameState()
        state = self.NewLevel(state)

        predLocation = (state.PredatorPos).Copy()
        if self.PredatorObservation(state):
            state.PredatorBeliefState = [state.AgentPos]
        else:
            allAgentLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
            validAgentLocations = list(set(allAgentLocations) - set(self.Occlusions))
            invisibleAgentLocations = [coord for coord in validAgentLocations if
                                       self.Grid.VisualRay(coord, predLocation, self.Occlusions)]
            state.PredatorBeliefState = invisibleAgentLocations

        return state

    def CreateRandomStartState(self):
        state = GameState()
        state = self.NewLevel(state)
        self.GetValidPredatorLocations()

        predLocation = (state.PredatorPos).Copy()
        if self.PredatorObservation(state):
            state.PredatorBeliefState = [state.AgentPos]
        else:
            allAgentLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
            validAgentLocations = list(set(allAgentLocations) - set(self.Occlusions))
            invisibleAgentLocations = [coord for coord in validAgentLocations if self.Grid.VisualRay(coord, predLocation, self.Occlusions)]
            state.PredatorBeliefState = invisibleAgentLocations

        agentObservation = np.zeros(self.NumActions)
        for scan in range(self.NumActions):
            agentObservation[scan] = self.MakeObservation(state, scan)
        if agentObservation.any():
            return state

        state.PredatorPos = self.StartPredatorLocations[Random(0, len(self.StartPredatorLocations))]
        return state

    def NewLevel(self, state):
        state.AgentPos = self.AgentHome
        state.AgentDir = -1
        state.PredatorPos = self.PredatorHome
        state.PredatorDir = -1
        state.PredatorSpeedMult = 2
        state.Depth = 0
        state.PredatorBeliefState = None

        return state

    def Valid(self, pos):
        if self.Grid.Inside(pos) and pos not in self.Occlusions:
            return True
        return False

    def NextPos(self, fromCoord, dir):
        nextPos = fromCoord + Compass[dir]
        if self.Valid(nextPos):
            return nextPos
        else:
            return Compass[COMPASS.NAA]

    def Step(self, state, action):
        reward = self.RewardDefault
        state.AgentDir = action

        newpos = self.NextPos(state.AgentPos, action)
        if self.Valid(newpos):
            state.AgentPos = newpos
        else:
            reward += self.RewardHitWall

        observation = 0
        hitPredator = 0
        if state.AgentPos == state.PredatorPos:
            reward += self.RewardDie
            return True, state, observation, reward  # Terminate death

        if self.PredatorObservation(state):
            state.PredatorBeliefState = [state.AgentPos]

        if not self.PredatorObservation(state):
            state.PredatorBeliefState = self.PredatorAgentPosPropogation(state)

        previousPredatorLocation = state.PredatorPos
        state, hitPredator = self.MovePredator(state, Bernoulli(self.MoveProbability), previousPredatorLocation)

        observation = self.MakeObservation(state, action)

        if hitPredator:
            reward += self.RewardDie
            return True, state, observation, reward #Terminate death

        if state.AgentPos == self.GoalPos:
            reward += self.RewardClearLevel
            return True, state, observation, reward #Terminate goal state

        state.Depth += 1
        return False, state, observation, reward

    def MakeObservation(self, state, action):
        observation = 0
        if self.VisualRange:
            visualArea = self.Grid.VisualArea(state.AgentPos, action, self.VisualRange)

            # Predator Observation
            for i, coord in enumerate(visualArea):
                if state.PredatorPos == coord:
                    if self.Grid.VisualRay((state.AgentPos).Copy(), (coord).Copy(), self.Occlusions)[0]:
                        observation = i #SetFlag(observation, i)
        else:
            if self.Grid.VisualRay((state.AgentPos).Copy(), (state.PredatorPos).Copy(), self.Occlusions)[0]:
                observation = 1

        return observation

    def PredatorObservation(self, state):
        predatorObservation, _ = self.Grid.VisualRay((state.AgentPos).Copy(), (state.PredatorPos).Copy(),
                                                     self.Occlusions)
        return predatorObservation

    def PredatorAgentPosPropogation(self, state):
        copyState = self.Copy(state)

        N2 = 15
        N1 = 15

        allPossibleAgentNewPositions = [copyState.PredatorBeliefState[0]]
        testedAgentPositions = []
        for n1 in range(N1):
            agentPosition = copyState.PredatorBeliefState[Random(0, len(copyState.PredatorBeliefState))]
            testedAgentPositions.append(agentPosition)
            propogateState = self.Copy(copyState)
            propogateState.AgentPos = agentPosition
            if self.PredatorObservation(propogateState):
                continue
            newAgentPositions = []
            for n2 in range(N2):
                for action in range(self.NumActions):
                    newAgentPosition = self.NextPos(propogateState.AgentPos, action)
                    if self.Valid(newAgentPosition) and newAgentPosition != copyState.PredatorPos:
                        newAgentPositions.append(newAgentPosition)
            if newAgentPositions:
                allPossibleAgentNewPositions.extend(newAgentPositions)

        for agentCoord in copyState.PredatorBeliefState:
            if agentCoord not in testedAgentPositions:
                allPossibleAgentNewPositions.append(agentCoord)

        return allPossibleAgentNewPositions

    def LocalMove(self, state, history, stepObs, status):
        allPredatorLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
        possiblePredatorLocations = list(set(allPredatorLocations) - set(self.Occlusions)) #Remove occlusions from possible predator location list
        state.PredatorPos = possiblePredatorLocations[Random(0, len(possiblePredatorLocations))]
        #state.PredatorPos = COORD(Random(0, self.Grid.GetXSize()),
        #                          Random(0, self.Grid.GetYSize()))
        if history.Size() == 0:
            return True, state
        observation = self.MakeObservation(state, state.AgentDir)
        return history.Back().Observation == observation

    def MovePredator(self, state, move, previousPredatorLocation):
        numberOfMoves = 1
        randomMove = True
        believedState = self.Copy(state)

        if move:
            numberOfMoves = state.PredatorSpeedMult

        if state.PredatorBeliefState:
            try:
                believedAgentPosition = state.PredatorBeliefState[0]
                randomMove = False
            except IndexError:
                numberOfMoves = 1
                pass

            if len(state.PredatorBeliefState) > 1:
                believedAgentPosition = state.PredatorBeliefState[Random(0, len(state.PredatorBeliefState))]
                numberOfMoves = 1
                randomMove = False
            believedState.AgentPos = believedAgentPosition

        for i in range(numberOfMoves):
            if Bernoulli(self.ChaseProbability) or i > 0 and not randomMove:
                believedState = self.MovePredatorAggressive(believedState, previousPredatorLocation)
            else:
                believedState = self.MovePredatorRandom(believedState)

            state.PredatorPos = believedState.PredatorPos
            if state.AgentPos == state.PredatorPos:
                return state, (state.AgentPos == state.PredatorPos)

        return state, (state.AgentPos == state.PredatorPos)


    def MovePredatorRandom(self, state):
        copyState = self.Copy(state)
        predatorPos = copyState.PredatorPos
        testedActions = []
        while True:
            action = Random(0, 4)
            testedActions.append(action)
            newpos = self.NextPos(predatorPos, action)
            if self.Valid(newpos):
                break

            if set(testedActions) == {0, 1, 2, 3}:
                newpos = copyState.PredatorPos
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
            if dist <= bestDist and self.Valid(newpos) and newpos != previousPredatorLocation:
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        newState.PredatorPos = bestPos
        newState.PredatorDir = bestDir

        return newState

    def GenerateLegal(self, state, history, legal, status):
        for a in range(self.NumActions):
            newpos = self.NextPos(state.AgentPos, a)
            if self.Valid(newpos) and newpos != state.PredatorPos:
                legal.append(a)

        return legal

    def GeneratePreferred(self, state, history, actions, status):
        if history.Size():
            for a in range(self.NumActions):
                newpos = self.NextPos(state.AgentPos, a)
                if self.Valid(newpos) and newpos != state.PredatorPos:
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
