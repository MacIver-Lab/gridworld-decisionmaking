from game import Game
from simulator import Knowledge
from mcts import SearchParams, UnitTestMCTS
from experiment import ExperimentParams, Experiment
from entropy import UnitTestENTROPY
from utils import UnitTestUTILS, Random
from coord import COORD, UnitTestCOORD
from grid import Grid
from entropy import Entropy
from astar import AStar
from math import floor

import os, sys, pickle, itertools
from pathlib2 import Path
import numpy as np

from mpi4py import MPI

SearchParams.Verbose = 0

XSize = 15
YSize = 15
AgentHome = COORD(7, 0)
GoalPos = COORD(floor(XSize/2), YSize-1)
visualRange = None


treeknowlege = 2 # 0 = pure, 1 = legal, 2 = smart
rolloutknowledge = 2 # 0 = pure, 1 = legal, 2 = smart
smarttreecount = 1.0 # prior count for preferred actions in smart tree search
smarttreevalue = 1.0 # prior value for preferred actions during smart tree search

def GetPath(occlusions):
    ASTAR = AStar(XSize, YSize, occlusions)
    ASTAR.InitializeGrid(AgentHome, GoalPos)
    path = ASTAR.Solve()

    return path

def GetEntropy(occlusions):
    I = np.zeros((XSize, YSize))
    for coord in occlusions:
        I[coord.X, coord.Y] = 1
    entropy = Entropy(I)
    outputMatrix = entropy.MovingWindowFilter(entropy.MovingAverage, 1)
    filteredMatrices = [outputMatrix]
    profile = entropy.Profile(filteredMatrices)

    return profile

def UnitTests():
    print("Testing UTILS")
    UnitTestUTILS()
    print("Testing COORD")
    UnitTestCOORD()
    print("Testing MCTS")
    UnitTestMCTS()
    print("Testing ENTROPY")
    UnitTestENTROPY()
    print("Testing complete!")

def SafeMultiExperiment(args):
    try:
        #print("Starting Simulation %d, Entropy 0.%d, Predator %d, Visual Range %s"
        #      % (args[3], args[0][0], args[1][0], str(args[2])))
        MultiExperiment(args)
    except (ValueError, IndexError) as e:
        print("Error in Simulation %d, Entropy 0.%d, Predator %d, Visual Range %s"
              %(args[3], args[0][0], args[1][0], str(args[2])))

def MultiExperiment(args):
    occlusionInd = args[0][0]
    occlusions = args[0][1]
    predatorInd = args[1][0]
    predatorHome = args[1][1]
    visualRange = args[2]
    simulationInd = args[3]
    directory = args[4]

    real = Game(XSize, YSize, occlusions=occlusions)
    simulator = Game(XSize, YSize, occlusions=occlusions)

    knowledge = Knowledge()
    knowledge.TreeLevel = treeknowlege
    knowledge.RolloutLevel = rolloutknowledge
    knowledge.SmartTreeCount = smarttreecount
    knowledge.SmartTreeValue = smarttreevalue

    experiment = Experiment(real, simulator)

    simulationDirectory = directory + '/Data/Simulation_%d'%(simulationInd)
    Path(simulationDirectory).mkdir(parents=True, exist_ok=True)

    _ = experiment.DiscountedReturn(occlusions, predatorHome, knowledge,
                                    occlusionInd, predatorInd, simulationDirectory, visualRange=visualRange)


def MPITest(args):
    print("Testing Simulation %d, Entropy 0.%d, Predator %d"
              %(args[3], args[0][0], args[1][0]))
    Path((args[0])+'/Data').mkdir(parents=True, exist_ok=True)
    filename = args[0] + "/Data/Results_" + str(args[5]) + ".txt"
    print(filename)
    with open(filename, 'w') as f:
        f.write("Hello world!")

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('READY', 'DONE', 'EXIT', 'START')


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    #rank = 0
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    status = MPI.Status()

    if rank == 0:
        print("Starting simulation...")
        planningDir = os.getcwd()

        #UnitTests()

        if not os.path.exists('init_vars.pkl'):
            occlusionList = [[[] for occlusionInd in ExperimentParams.EntropyLevels]
                                    for simulationInd in range(ExperimentParams.NumRuns)]
            predatorList = [[[] for occlusionInd in ExperimentParams.EntropyLevels]
                             for simulationInd in range(ExperimentParams.NumRuns)]
            for simulationInd in range(ExperimentParams.NumRuns):
                for i, entropyValue in enumerate(ExperimentParams.EntropyLevels):
                    while True and entropyValue < .8:
                        numOcclusions = 0
                        occlusions = Grid(XSize, YSize).CreateOcclusions(numOcclusions, AgentHome, GoalPos)
                        while GetEntropy(occlusions)[0] < entropyValue:
                            #entropyMatrix = GetEntropy(occlusions)[0]
                            #if entropyMatrix >= entropyValue:
                            #    break
                            occlusions = Grid(XSize, YSize).CreateOcclusions(numOcclusions, AgentHome, GoalPos)
                            numOcclusions += 1
                        path = GetPath(occlusions)
                        if path:
                            occlusionList[simulationInd][i] = occlusions
                            break

                    while True and entropyValue >= .8:
                        numOcclusions = 20
                        occlusions = Grid(XSize, YSize).CreateRandomOcclusions(numOcclusions, AgentHome, GoalPos)
                        while GetEntropy(occlusions)[0] < entropyValue:
                            occlusions = Grid(XSize, YSize).CreateOcclusions(numOcclusions, AgentHome, GoalPos)
                            numOcclusions += 1

                        path = GetPath(occlusions)
                        if path:
                            occlusionList[simulationInd][i] = occlusions
                            break

                    allPredatorLocations = Grid(XSize, YSize).CreatePredatorLocations(ExperimentParams.SpawnArea, AgentHome,
                                                                              GoalPos,occlusions)

                    temp_predator = [allPredatorLocations[Random(0, len(allPredatorLocations))]
                                         for p in range(ExperimentParams.NumPredators)]
                    predatorList[simulationInd][i] = temp_predator


            with open('init_vars.pkl', 'wb') as f:
                pickle.dump([occlusionList, predatorList], f)

        with open('init_vars.pkl', 'rb') as f:
            occlusionList, predatorList = pickle.load(f)

        tasks = []
        for simulationInd in range(ExperimentParams.NumRuns):
            for occlusionInd in range(len(ExperimentParams.EntropyLevels)):
                for predatorInd in range(ExperimentParams.NumPredators):
                    if visualRange:
                        for visrange in visualRange:
                            tasks.append([(occlusionInd, occlusionList[simulationInd][occlusionInd]),
                                          (predatorInd, predatorList[simulationInd][occlusionInd][predatorInd]), visrange,
                                          simulationInd, planningDir])
                    tasks.append([(occlusionInd, occlusionList[simulationInd][occlusionInd]),
                            (predatorInd, predatorList[simulationInd][occlusionInd][predatorInd]), None, simulationInd, planningDir])

        task_index = 0
        numWorkers = size - 1
        closedWorkers = 0
        print("Master starting with %d workers" % numWorkers)
        print("Total number of tasks %d" % len(tasks))

        while closedWorkers < numWorkers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                if task_index < len(tasks):
                    currentTask = tasks[task_index]
                    print(" Starting Simulation %d, Entropy 0.%d, Predator %d, Visual Range %s"
                            %(currentTask[3], currentTask[0][0], currentTask[1][0], str(currentTask[2])))
                    print("Sending task %d:%d to worker %d" % (task_index, len(tasks), source))
                    comm.send(currentTask, dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:
                print("Finished processing worker %d" % source)

            elif tag == tags.EXIT:
                print("Worker %d exited." % source)
                closedWorkers += 1

        print("Master finishing...")
        sys.exit(1)

    else:
        name = MPI.Get_processor_name()
        print("I am a worker with rank %d on %s." % (rank, name))

        while True:
            comm.send(None, dest=0, tag=tags.READY)
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:
                SafeMultiExperiment(task)
                #MPITest(task)
                comm.send(None, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
