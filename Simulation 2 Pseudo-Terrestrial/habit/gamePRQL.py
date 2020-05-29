from simulator import Simulator
from grid import Grid
from coord import COORD, COMPASS, Compass, AggressiveDirectionalDistance
from utils import Random, Bernoulli

from math import floor

class GameState:
    def __init__(self):
        self.AgentPos = COORD(5, 0)
        self.AgentDir = -1
        self.PredatorPos = COORD(0, 0)
        self.PredatorDir = -1
        self.PredatorSpeedMult = 2
        self.PredatorBeliefState = None

    def __eq__(self, other):
        return self.AgentPos == other.AgentPos and \
               self.PredatorPos == other.PredatorPos

    def __str__(self):
        return "(" + str(self.AgentPos) + "; " + str(self.PredatorPos) + ")"

    def __hash__(self):
        return 0

class Game(Simulator):
    def __init__(self, xsize, ysize, occlusions=[]):
        self.AgentHome = COORD(int(floor((xsize-1)/2)), 0)
        self.PredatorNum = 1
        self.PredatorHome = COORD(0, 0)
        self.ChaseProbability = 0.75
        self.MoveProbability = 0.5
        self.Discount = 0.95
        self.GoalPos = COORD(int(floor((xsize-1)/2)), ysize-1)

        self.Grid = Grid(xsize, ysize)

        self.NumActions = 4
        self.NumObservations = 2
        Simulator.__init__(self, self.NumActions, self.NumObservations, self.Discount)
        self.Occlusions = occlusions

        self.RewardClearLevel = 1000
        self.RewardDefault = -1
        self.RewardDie = -100
        self.RewardHitWall = -25
        self.RewardRange = 100

        self.State = GameState()
        self.State.AgentPos = self.AgentHome
        self.State.PredatorPos = self.PredatorHome

    def GetPossiblePredatorLocations(self):
        allPredatorLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
        invalidPredatorLocations = [self.GoalPos, self.AgentHome] + self.Occlusions
        return list(set(allPredatorLocations) - set(invalidPredatorLocations))

    def GetValidPredatorLocations(self):
        allPredatorLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
        invalidPredatorLocations = [self.GoalPos, self.AgentHome] + self.Occlusions
        for predator in allPredatorLocations:
            if self.Grid.VisualRay((self.AgentHome).Copy(), predator.Copy(), self.Occlusions)[0]:
                invalidPredatorLocations.append(predator)
        self.StartPredatorLocations = list(set(allPredatorLocations) - set(invalidPredatorLocations))

    def Copy(self, state):
        newState = GameState()

        newState.AgentPos = state.AgentPos
        newState.AgentDir = state.AgentDir
        newState.PredatorPos = state.PredatorPos
        newState.PredatorDir = state.PredatorDir
        newState.PredatorSpeedMult = state.PredatorSpeedMult
        newState.PredatorBeliefState = state.PredatorBeliefState

        return newState

    def Validate(self, state):
        assert(self.Grid.Inside(state.AgentPos))
        assert(self.Grid.Inside(state.PredatorPos))

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
        else:
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
        # Predator Observation
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
            try:
                agentPosition = copyState.PredatorBeliefState[Random(0, len(copyState.PredatorBeliefState))]
            except IndexError:
                break
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


    def MovePredator(self, state, move, previousPredatorLocation):
        numberOfMoves = 1

        observation = self.PredatorObservation(self.Copy(state))
        try:
            believedAgentPosition = state.PredatorBeliefState[0]
        except IndexError:
            if observation:
                believedAgentPosition = (state.AgentPos).Copy()
                state.PredatorBeliefState = [(state.AgentPos).Copy()]
            else:
                allAgentLocations = [COORD(x, y) for x in range(self.XSize) for y in range(self.YSize)]
                invisibleAgentLocations = [coord for coord in allAgentLocations if
                                           self.Grid.VisualRay(coord, (state.PredatorPos).Copy(), self.Occlusions)]
                validAgentLocations = list(set(invisibleAgentLocations) - set(self.Occlusions))
                state.PredatorBeliefState = validAgentLocations

        if len(state.PredatorBeliefState) > 1:
            believedAgentPosition = state.PredatorBeliefState[Random(0, len(state.PredatorBeliefState))]
            numberOfMoves = 1

        if move:
            numberOfMoves = state.PredatorSpeedMult

        believedState = self.Copy(state)
        believedState.AgentPos = believedAgentPosition
        believedState.AgentPos = (state.AgentPos).Copy()

        for i in range(numberOfMoves):
            if Bernoulli(self.ChaseProbability) or (i > 0 and len(self.Occlusions) > 15):
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