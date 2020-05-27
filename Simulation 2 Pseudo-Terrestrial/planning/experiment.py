from mcts import MCTS, SearchParams
from episode import Episode
from timeit import default_timer as timer
from statistics import STATISTICS

import csv, pickle, os

from pathlib2 import Path
import pandas as pd

class Results:
    def __init__(self):
        self.Time = STATISTICS(0., 0.)
        self.Reward = STATISTICS(0., 0.)
        self.DiscountedReturn = STATISTICS(0., 0.)
        self.UndiscountedReturn = STATISTICS(0., 0.)
        self.Steps = STATISTICS(0., 0.)

    def Clear(self):
        self.Time.Clear()
        self.Reward.Clear()
        self.DiscountedReturn.Clear()
        self.UndiscountedReturn.Clear()
        self.Steps.Clear()


class ExperimentParams:
    SpawnArea = 4
    NumRuns = 20
    NumPredators = 5
    NumSteps = 200
    SimSteps = 1000
    TimeOut = 36000
    MinDoubles = 0
    MaxDoubles = 50
    TransformDoubles = -1
    TransformAttempts = 1000
    Accuracy = 0.01
    UndiscountedHorizon = 100
    AutoExploration = True
    EntropyLevels = [float(i)/10. for i in range(0, 10)]
    Depth = [100, 1000, 5000]


class Experiment:
    def __init__(self, real, simulator):
        self.Real = real
        self.Simulator = simulator
        self.Episode = Episode()

        if ExperimentParams.AutoExploration:
            if SearchParams.UseRave:
                SearchParams.ExplorationConstant = 0
            else:
                SearchParams.ExplorationConstant = self.Simulator.GetRewardRange()

        self.Results = Results()
        MCTS.InitFastUCB(SearchParams.ExplorationConstant)

    def Run(self):
        notOutOfParticles = True
        undiscountedReturn = 0.0
        discountedReturn = 0.0
        discount = 1.0

        state = self.Real.CreateStartState()
        currentState = self.Real.Copy(state)
        self.Episode.Add(-1, -1, currentState, 0)

        self.MCTS = MCTS(self.Simulator)

        start = timer()
        t = 0
        observation = 0
        self.NumObservation = 0


        while t < ExperimentParams.NumSteps:
            action = self.MCTS.SelectAction(state)

            terminal, state, observation, reward = self.Real.Step(state, action)
            currentState = self.Real.Copy(state)

            self.NumObservation += 1*(observation > 0)
            undiscountedReturn += reward
            discountedReturn += reward * discount
            discount *= self.Real.GetDiscount()

            self.Episode.Add(action, observation, currentState, reward)

            if SearchParams.Verbose:
                self.Real.DisplayState(state)

            if terminal:
                self.Episode.Add(action, observation, currentState, reward)
                self.Episode.Complete()
                return reward

            notOutOfParticles, beliefState = self.MCTS.Update(action, observation, currentState)
            if not notOutOfParticles:
                print("random action selection")
                self.Episode.Add(action, observation, currentState, reward)
                break

#            if (timer() - start) > ExperimentParams.TimeOut:
#                break

            t += 1

        if not notOutOfParticles:
            if SearchParams.Verbose:
                print("Out of particles, finishing episode with SelectRandom")
            history = self.MCTS.GetHistory()

            while t <= ExperimentParams.NumSteps:
                t += 1

                action = self.Simulator.SelectRandom(state, history, self.MCTS.GetStatus())
                terminal, state, observation, reward = self.Real.Step(state, action)

                self.Results.Reward.Add(reward)
                undiscountedReturn += reward
                discountedReturn += reward * discount
                discount *= self.Real.GetDiscount()

                if SearchParams.Verbose:
                    self.Real.DisplayState(state)

                if terminal:
                    self.Episode.Add(action, observation, state, reward)
                    self.Episode.Complete()
                    return reward

                self.MCTS.History.Add(action, observation)
                self.Episode.Add(action, observation, state, reward)

    def DiscountedReturn(self, occlusions, predatorHome, knowledge,
                                    occlusionInd, predatorInd, simulationDirectory, visualRange=None):

        occlusionDirectory = simulationDirectory + '/Occlusion_%d'%(occlusionInd)
        Path(occlusionDirectory).mkdir(parents=True, exist_ok=True)
        occlusionFile = occlusionDirectory + '/OcclusionCoordinates.csv'
        if not Path(occlusionFile).is_file():
            self.OcclusionCoords2CSV(occlusions, occlusionFile)

        if visualRange is None:
            summaryFile = simulationDirectory + '/Occlusion_%d/Predator_%d/Summary.csv' % (occlusionInd, predatorInd)
        else:
            summaryFile = simulationDirectory + '/Occlusion_%d/Predator_%d/VisualRange_%d/Summary.csv' % \
                          (occlusionInd, predatorInd, visualRange)

        summary = {'Depth 100': [],
                   'Depth 1000': []}

        if Path(summaryFile).is_file():
            ExperimentParams.Depth = [5000]

        for depth in ExperimentParams.Depth:
            if depth == 5000:
                if visualRange is None:
                    directory = simulationDirectory + '/Occlusion_%d/Predator_%d/Depth_%d' % (occlusionInd, predatorInd, depth)
                    Path(directory).mkdir(parents=True, exist_ok=True)
                else:
                    directory = simulationDirectory + '/Occlusion_%d/Predator_%d/VisualRange_%d/Depth_%d' % (
                    occlusionInd, predatorInd, visualRange, depth)
                    Path(directory).mkdir(parents=True, exist_ok=True)
            else:
                df = pd.DataFrame({key: pd.Series(value) for key, value in summary.items()})
                df.to_csv(summaryFile, encoding='utf-8', index=False)

            for trial in range(ExperimentParams.MinDoubles, ExperimentParams.MaxDoubles):
                if depth == 5000:
                    episodeFile = directory + '/Episode_%d.csv' % (trial)

                    if Path(episodeFile).is_file():
                        episode = pd.read_csv(episodeFile, header=0)
                        if episode['Reward'].iloc[-1] != -1:
                            return 1

                self.Results.Clear()

                SearchParams.NumSimulations = depth
                SearchParams.NumStartState = depth
                SearchParams.Softmax = False
                if depth > 10:
                    SearchParams.Softmax = True

                if int(depth * (10 ** ExperimentParams.TransformDoubles)) > 0:
                    SearchParams.NumTransforms = int(depth * (10 ** ExperimentParams.TransformDoubles))
                else:
                    SearchParams.NumTransforms = 1

                SearchParams.Softmax = False
                SearchParams.MaxAttempts = SearchParams.NumTransforms * ExperimentParams.TransformAttempts

                if visualRange:
                    if visualRange <= 3:
                        SearchParams.MaxDepth = 50
                    if visualRange == 4:
                        SearchParams.MaxDepth = 100
                    if visualRange > 4:
                        SearchParams.MaxDepth = 200

                self.Real.__init__(self.Real.XSize, self.Real.YSize, visualrange=visualRange,
                                   occlusions=occlusions)
                self.Real.PredatorHome = predatorHome
                self.Real.SetKnowledge(knowledge)

                self.Simulator.__init__(self.Simulator.XSize, self.Simulator.YSize, visualrange=visualRange,
                                        occlusions=occlusions)
                self.Simulator.PredatorHome = predatorHome
                self.Simulator.SetKnowledge(knowledge)

                if SearchParams.Verbose:
                    self.Real.InitializeDisplay()

                terminalReward = self.Run()

                if depth < 5000:
                    summary['Depth %d' % depth].append(terminalReward)
                else:
                    self.Episode.Episode2CSV(episodeFile)

                self.Episode.Clear()


    def Dictionary2CSV(self, dictTable, filename):
        columns = sorted(dictTable)
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(zip(*[dictTable[col] for col in columns]))

    def OcclusionCoords2CSV(self, occlusions, occlusionFile):
        occlusionDict = {}
        occlusionDict['X'] = []; occlusionDict['Y'] = []
        for coord in occlusions:
            occlusionDict['X'].append(coord.X); occlusionDict['Y'].append(coord.Y)

        self.Dictionary2CSV(occlusionDict, occlusionFile)
