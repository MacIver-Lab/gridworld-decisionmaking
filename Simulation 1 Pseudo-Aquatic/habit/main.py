import pandas as pd
import os, sys, glob
import numpy as np

from PolicyReuseQ import PRQLearningAgent
from coord import COORD, Opposite
from grid import Grid

import warnings
warnings.filterwarnings("ignore")

XSize = 15
YSize = 15
NumSimulations = 20
NumVisualRange = 5

AgentHome = COORD(7, 0)

def getEnvironmentPolicies(directory):
    parentDirectory = os.path.dirname(directory)
    print("Importing successful trajectories...")
    environmentPolicies = [[[] for vision in range(NumSimulations)]
                            for simulation in range(NumVisualRange)]

    for simulationInd in range(NumSimulations):
        for visrange in range(NumVisualRange):
            sys.stdout.write('\r' + "Simulation #: %d, Visual range: .%d" % (simulationInd, visrange+1))

            episodeFolder = parentDirectory + '/planning/Data/Simulation_%d/Vision_%d/Depth_5000/Episode_*.csv'%(simulationInd, visrange+1)
            episodeFiles = glob.glob(episodeFolder)

            for episodeFile in episodeFiles:
                episode = pd.read_csv(episodeFile, header=0)
                if episode['Reward'].iloc[-1] < 0:
                    continue

                policies = environmentPolicies[simulationInd][visrange]
                policy = (episode[['Action']].values).flatten()
                policies.append(policy[1:])
                environmentPolicies[simulationInd][visrange] = policies

    return environmentPolicies

def getPredatorLocations(directory):
    parentDirectory = os.path.dirname(directory)

    predatorLocations = [None for simulation in range(NumSimulations)]

    for simulationInd in range(NumSimulations):
        episodeFolder = parentDirectory + '/planning/Data/Simulation_%d/Vision_1/Depth_5000/Episode_*.csv'%(simulationInd)
        episodeFiles = glob.glob(episodeFolder)
        episode = pd.read_csv(episodeFiles[0], header=0)

        predatorLocations[simulationInd] = COORD(episode['Predator X'].iloc[0], episode['Predator Y'].iloc[0])

    return predatorLocations

def removeDuplicates(policyLibrary):
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

if __name__ == "__main__":

    directory = os.getcwd()
    environmentPolicies = getEnvironmentPolicies(directory)
    predatorLocations = getPredatorLocations(directory)

    print("Starting habit simulations")
    habitSurvivalRate = np.full((NumSimulations, NumVisualRange), np.nan)

    for visualrange in range(1, NumVisualRange+1):
        for simulation in range():
            sys.stdout.write('\r' + "Simulation #: %d, Visual range: %d" % (simulation, visualrange))

            try:
                predatorHome = predatorLocations[simulation]
            except IndexError:
                continue

            inVisualCone=False
            for action in range(4):
                if predatorHome in Grid(XSize, YSize).VisualArea(AgentHome, action, visualrange):
                    inVisualCone = True

            aggregatePolicyLibrary = []
            if not inVisualCone:
                for predator in range(0, NumSimulations):
                    try:
                        tempPredatorHome = predatorLocations[simulation]
                    except IndexError:
                        continue

                    tempPredInCone = False
                    for action in range(4):
                        if tempPredatorHome in Grid(XSize, YSize).VisualArea(AgentHome, action, visualrange):
                            tempPredInCone = True

                    if not tempPredInCone:
                        aggregatePolicyLibrary.extend(removeDuplicates(environmentPolicies[predator][visualrange-1]))


            else:
                aggregatePolicyLibrary = removeDuplicates(environmentPolicies[simulation][visualrange-1])

            policyLibrary = aggregatePolicyLibrary

            if not policyLibrary:
                continue

            habitSurvivalRate[simulation, visualrange - 1] = PRQLearningAgent(policyLibrary, visualrange=visualrange).habitSimulation(predatorHome)
