from mcts import MCTS, SearchParams
from episode import Episode
from timeit import default_timer as timer
from statistics import STATISTICS

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
    NumSteps = 100
    SimSteps = 1000
    TimeOut = 36000
    Forget = 100
    MinDoubles = 0
    MaxDoubles = 100
    NumDepth = 14
    NumPredatorLocations = 1000
    TransformDoubles = -1
    TransformAttempts = 1000
    Accuracy = 0.01
    UndiscountedHorizon = 100
    AutoExploration = True
    Depth = [1, 10, 100]


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
            self.Results.Reward.Add(reward)
            undiscountedReturn += reward
            discountedReturn += reward * discount
            discount *= self.Real.GetDiscount()

            if SearchParams.Verbose:
                self.Real.DisplayState(state)

            if terminal:
                self.Episode.Add(action, observation, currentState, reward)
                self.Episode.Complete()
                return reward

            notOutOfParticles, beliefState = self.MCTS.Update(action, observation, reward)
            if not notOutOfParticles:
                break

            self.Episode.Add(action, observation, currentState, reward)
            #if (timer() - start) > ExperimentParams.TimeOut:
            #    break

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


    def DiscountedReturn(self, predatorHome, simulationDirectory, knowledge):

        for visualRange in range(1, 2):

            summary = {'Depth 1': [],
                       'Depth 10': [],
                       'Depth 100': [],
                       'Depth 1000': []}
            summaryFile = simulationDirectory + '/VisualRange_%d/Summary.csv' % \
                          (visualRange)

            directory = simulationDirectory + '/VisualRange_%d'%(visualRange)
            Path(directory).mkdir(parents=True, exist_ok=True)

            for depth in ExperimentParams.Depth:
                if depth == 5000:
                    directory = simulationDirectory + '/VisualRange_%d/Depth_%d' % (visualRange, depth)
                    Path(directory).mkdir(parents=True, exist_ok=True)


                for trial in range(ExperimentParams.MinDoubles, 1): #ExperimentParams.MaxDoubles
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

                    SearchParams.MaxAttempts = SearchParams.NumTransforms * ExperimentParams.TransformAttempts

                    if visualRange <= 3:
                        SearchParams.MaxDepth = 50
                    elif visualRange == 4:
                        SearchParams.MaxDepth = 100
                    elif visualRange > 4:
                        SearchParams.MaxDepth = 200

                    self.Real.__init__(self.Real.XSize, self.Real.YSize, visualRange)
                    self.Real.PredatorHome = predatorHome
                    self.Real.SetKnowledge(knowledge)

                    self.Simulator.__init__(self.Simulator.XSize, self.Simulator.YSize, visualRange)
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

            df = pd.DataFrame({key: pd.Series(value) for key, value in summary.items()})
            df.to_csv(summaryFile, encoding='utf-8', index=False)
