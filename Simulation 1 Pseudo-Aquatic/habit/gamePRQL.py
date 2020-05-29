from coord import COORD, COMPASS, Compass, AggressiveDirectionalDistance
from utils import Random, Bernoulli

from math import floor
from grid import Grid

class GameState:
    def __init__(self):
        self.AgentPos = COORD(7, 0)
        self.PredatorPos = COORD(0, 0)
        self.PredatorDir = -1
        self.PredatorSpeedMult = 2

    def GetState(self):
        return self.AgentPos.X + (self.AgentPos.Y*14)
    
    def GetAgent(self):
        return str(self.AgentPos)
    
    def GetPredator(self):
        return str(self.PredatorPos)
        

class Game():
    def __init__(self, xsize, ysize, visualrange=1, visualcone=None):
        self.AgentHome = COORD(int(floor((xsize-1)/2)), 0)
        self.PredatorNum = 1
        self.PredatorHome = COORD(0, 0)
        self.ChaseProbability = 0.75

        self.MoveProbability = 0.5
        self.GoalPos = COORD(int(floor((xsize-1)/2)), ysize-1)

        self.NumActions = 4
        self.VisualRange = visualrange

        self.State = GameState()
        self.State.AgentPos = self.AgentHome
        self.State.PredatorPos = self.PredatorHome

        self.XSize = xsize
        self.YSize = ysize

        self.RewardDefault = -1
        self.RewardHitWall = -25
        self.RewardClearLevel = 1000
        self.RewardDie = -100

        self.Grid = Grid(xsize, ysize)
    
    def CreateStartState(self):
        state = GameState()
        state = self.NewLevel(state)
        return state

    def Copy(self, state):
        newState = GameState()

        newState.AgentPos = state.AgentPos
        newState.PredatorPos = state.PredatorPos
        newState.PredatorDir = state.PredatorDir
        newState.PredatorSpeedMult = state.PredatorSpeedMult

        return newState

    def Inside(self, position):
        return position.X > -1 and position.X < self.XSize and position.Y > -1 and position.Y < self.YSize


    def NextPos(self, fromCoord, dir):
        nextPos = COORD(fromCoord.X, fromCoord.Y) + Compass[dir]
        if self.Inside(nextPos):
            return nextPos
        else:
            return Compass[COMPASS.NAA]

    def Step(self, state, action):
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
        move = Bernoulli(self.MoveProbability)
        state, hitPredator = self.MovePredator(state, move, previousPredatorLocation, hitPredator)

        observation = self.MakeObservation(state, action)

        if state.AgentPos == self.GoalPos:
            reward += self.RewardClearLevel
            return True, state, observation, reward  # Terminate goal state

        if hitPredator:
            reward += self.RewardDie
            return True, state, observation, reward  # Terminate death

        return False, state, observation, reward

    def MakeObservation(self, state, action):
        copyState = self.Copy(state)
        visualCone = self.Grid.VisualArea(copyState.AgentPos, action, self.VisualRange)
        if copyState.PredatorPos in visualCone:
            return 1
        return 0

    def MovePredator(self, state, move, previousPredatorLocation, hitPredator):
        if hitPredator:
            return state, (state.AgentPos == state.PredatorPos)

        copyState = self.Copy(state)

        if move:
            numberOfMoves = state.PredatorSpeedMult
        else:
            numberOfMoves = 1

        for i in range(numberOfMoves):
            if Bernoulli(self.ChaseProbability) or (i > 0 and self.VisualRange == 1):
                copyState = self.MovePredatorAggressive(copyState)
            else:
                copyState = self.MovePredatorRandom(copyState, previousPredatorLocation)

            if copyState.AgentPos == copyState.PredatorPos:
                return copyState, (copyState.AgentPos == copyState.PredatorPos)

        return copyState, (copyState.AgentPos == copyState.PredatorPos)


    def MovePredatorRandom(self, state, previousPredatorLocation):
        copyState = self.Copy(state)
        numActions = 4
        predatorPos = copyState.PredatorPos
        testedActions = []
        while True:
            action = Random(0, numActions)
            testedActions.append(action)
            newpos = self.NextPos(predatorPos, action)
            if newpos.Valid() and newpos != previousPredatorLocation:
                break
            if set(testedActions) == set(range(self.NumActions)):
                newpos = predatorPos
                break

        copyState.PredatorPos = newpos
        copyState.PredatorDir = action
        return copyState

    def MovePredatorAggressive(self, state, previousPredatorLocation=None):
        copyState = self.Copy(state)

        bestDist = self.Grid.GetXSize() + self.Grid.GetYSize()
        bestPos = state.PredatorPos
        bestDir = -1

        agentPos = copyState.AgentPos
        predatorPos = copyState.PredatorPos

        for dir in range(0, self.NumActions):
            dist = AggressiveDirectionalDistance(agentPos, predatorPos, dir)
            newpos = self.NextPos(predatorPos, dir)
            if previousPredatorLocation is  None:
                if dist <= bestDist and newpos.Valid():
                    bestDist = dist
                    bestPos = newpos
                    bestDir = dir
            else:
                if dist <= bestDist and newpos.Valid() and newpos != previousPredatorLocation:
                    bestDist = dist
                    bestPos = newpos
                    bestDir = dir


        copyState.PredatorPos = bestPos
        copyState.PredatorDir = bestDir

        return copyState

    def NewLevel(self, state):
        state.AgentPos = self.AgentHome
        state.PredatorPos = self.PredatorHome
        state.PredatorDir = 2

        return state