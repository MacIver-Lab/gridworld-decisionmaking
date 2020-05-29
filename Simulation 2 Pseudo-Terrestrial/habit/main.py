from gamePRQL import Game
from coord import COORD
from experiment import Experiment

import os, sys, glob, pickle
import numpy as np
import pandas as pd


XSize = 15
YSize = 15
NumSimulations = 20
NumEntropy = 10
NumPredators = 5


def getEnvironmentPolicies(directory):
    parentDirectory = os.path.dirname(directory)

    print("Importing successful trajectories...")
    environmentPolicies = [[[[] for predator in range(NumPredators)] for occlusion in range(NumEntropy)]
                            for simulation in range(NumSimulations)]

    predatorHomes = np.full((NumPredators, NumSimulations, NumEntropy), np.nan, dtype='object')

    for simulationInd in range(NumSimulations):
        for occlusionInd in range(NumEntropy):
            sys.stdout.write('\r' + "Simulation #: %d, Entropy: .%d" % (simulationInd, occlusionInd))

            for predatorInd in range(NumPredators):
                if not os.path.exists(parentDirectory+'/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/'%(simulationInd,occlusionInd, predatorInd)):
                    continue
                episodeFolder = parentDirectory+'/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/Depth_5000/Episode_*.csv'%(simulationInd,
                                                                                                                           occlusionInd, predatorInd)
                episodeFiles = glob.glob(episodeFolder)
                loadPredator = True
                for i, episodeFile in enumerate(episodeFiles):
                    episode = pd.read_csv(episodeFile, header=0)
                    if i == 0:
                        predatorHomes[predatorInd, simulationInd, occlusionInd] = COORD(episode['Predator X'].iloc[0],
                                                                                        episode['Predator Y'].iloc[0])
                    if episode['Reward'].iloc[-1] < 0:
                        continue

                    policies = environmentPolicies[simulationInd][occlusionInd][predatorInd]
                    policy = (episode[['Action']].values).flatten()
                    policies.append(policy[1:])
                    environmentPolicies[simulationInd][occlusionInd][predatorInd] = policies

    return environmentPolicies, predatorHomes

def getOcclusions(directory):
    parentDirectory = os.path.dirname(directory)

    occlusionList = [[[] for occlusion in range(NumEntropy)] for simulation in range(NumSimulations)]
    for simulationInd in range(NumSimulations):
        for occlusionInd in range(NumEntropy):
            occlusionFile = parentDirectory + "/planning/Data/Simulation_%d/Occlusion_%d/OcclusionCoordinates.csv" % (
                     simulationInd, occlusionInd)

            occlusion = pd.read_csv(occlusionFile, header=0)
            occlusions = [COORD(x[0], x[1]) for x in occlusion[['X', 'Y']].values]

            occlusionList[simulationInd][occlusionInd] = occlusions
    return occlusionList

if __name__ == "__main__":

    print("Starting simulation...")
    directory = os.getcwd()

    importOther = False

    if not importOther:
        environmentPolicies, predatorHomes = getEnvironmentPolicies(directory)
        occlusionList = getOcclusions(directory)
    else: #If importing policies from another place format:
        #environmentPolicies: nested list of list with indexing [simulation][entropy][predatorIndex]
        #predatorHomes: numpy list of objects with indexing [predatorIndex, simulation, entropy]
        #occlusionList: nested list of list with indexing [simulation][entropy]
        #save all of it to a pickle file with pickle.dump([environmentPolicies, predatorHomes, occlusionList], f)
        with open("policies_occlusions_predators.pkl", "rb") as f:
            environmentPolicies, predatorHomes, occlusionList = pickle.load(f)


    habitSurvivalRate = np.zeros((NumPredators, NumSimulations, NumEntropy))
    habitGDist = np.zeros((NumPredators, NumSimulations, NumEntropy))

    for occlusionInd in range(NumEntropy):
        for simulationInd in range(NumSimulations):

            occlusions = occlusionList[simulationInd][occlusionInd]
            real = Game(XSize, YSize, occlusions=occlusionList[simulationInd][occlusionInd])
            experiment = Experiment(real)

            for predatorInd in range(NumPredators):
                sys.stdout.write('\r' + "Simulation #: %d, Entropy: .%d, Predator: %d" % (simulationInd, occlusionInd, predatorInd))
                predatorHome = predatorHomes[predatorInd, simulationInd, occlusionInd]

                try:
                    if np.isnan(predatorHome):
                        habitSurvivalRate[predatorInd, simulationInd, occlusionInd] = np.nan
                        continue
                except:
                    pass

                policies = environmentPolicies[simulationInd][occlusionInd][predatorInd]
                if not real.Grid.VisualRay((real.AgentHome).Copy(), (predatorHome).Copy(), occlusions)[0]:
                    aggregatePolicyLibrary = []
                    for predatorInd2 in range(5):

                        try:
                            if np.isnan(predatorHomes[predatorInd2, simulationInd, occlusionInd]):
                                continue
                        except:
                            pass

                        if not real.Grid.VisualRay((real.AgentHome).Copy(),
                                                   (predatorHomes[predatorInd2, simulationInd, occlusionInd]).Copy(),
                                                   occlusions)[0]:
                            aggregatePolicyLibrary.extend(environmentPolicies[simulationInd][occlusionInd][predatorInd2])

                    policies = aggregatePolicyLibrary
                sr, gDist = experiment.DiscountedReturn(policies, predatorHome, occlusions)

                habitSurvivalRate[predatorInd, simulationInd, occlusionInd] = sr
                habitGDist[predatorInd, simulationInd, occlusionInd] = gDist




