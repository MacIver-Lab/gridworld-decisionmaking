from game import Game
from simulator import Knowledge
from mcts import SearchParams, UnitTestMCTS
from experiment import ExperimentParams, Experiment
from utils import UnitTestUTILS
from coord import COORD, UnitTestCOORD, ManhattanDistance

import os, sys, pickle, random
from pathlib2 import Path

from mpi4py import MPI

SearchParams.Verbose = 0

XSize = 15
YSize = 15

treeknowlege = 2 # 0 = pure, 1 = legal, 2 = smart
rolloutknowledge = 2 # 0 = pure, 1 = legal, 2 = smart
smarttreecount = 10.0 # prior count for preferred actions in smart tree search
smarttreevalue = 1.0 # prior value for preferred actions during smart tree search

def UnitTests():
    print("Testing UTILS")
    UnitTestUTILS()
    print("Testing COORD")
    UnitTestCOORD()
    print("Testing MCTS")
    UnitTestMCTS()
    print("Testing complete!")

def SafeMultiExperiment(args):
    try:
        MultiExperiment(args)
    except (ValueError, IndexError) as e:
        print("Error in Simulation %d"%(args[0][0]))

def MultiExperiment(args):
    simulationInd = args[0][0]
    predatorHome = args[0][1]
    directory = args[1]

    real = Game(XSize, YSize)
    simulator = Game(XSize, YSize)

    knowledge = Knowledge()
    knowledge.TreeLevel = treeknowlege
    knowledge.RolloutLevel = rolloutknowledge
    knowledge.SmartTreeCount = smarttreecount
    knowledge.SmartTreeValue = smarttreevalue

    experiment = Experiment(real, simulator)

    simulationDirectory = directory + '/Data/Simulation_%d' % (simulationInd)
    Path(simulationDirectory).mkdir(parents=True, exist_ok=True)

    _ = experiment.DiscountedReturn(predatorHome, simulationDirectory, knowledge)


def MPITest(args):
    print("Testing simulation %d, occlusion index: %d, visual range: %d, depth: %d, predator index: %d"
                           %(args[5], args[1][0], args[2], args[3], args[4][0]))
    predatorInd = args[4][0]
    filename = args[0] + "/Results_" + str(predatorInd) + ".txt"
    with open(filename, 'w') as f:
        f.write("Hello world!")


def GetPredatorLocations():
    #np.random.seed(1)

    tempGame = Game(XSize, YSize)
    agentSurroundCoordinates = []
    for x in range(tempGame.AgentHome.X - ExperimentParams.SpawnArea,
                    tempGame.AgentHome.X + ExperimentParams.SpawnArea + 1):
        for y in range(tempGame.AgentHome.Y - ExperimentParams.SpawnArea,
                        tempGame.AgentHome.Y + ExperimentParams.SpawnArea + 1):
            if (ManhattanDistance(COORD(x, y), tempGame.AgentHome) <= 2 * ExperimentParams.SpawnArea):
                agentSurroundCoordinates.append(COORD(x, y))
    agentSurroundCoordinates.append(tempGame.GoalPos)

    allPredatorLocations = [COORD(x, y) for x in range(0, XSize) for y in range(0, YSize)]
    validSpawnLocations = list(set(allPredatorLocations) - set(agentSurroundCoordinates))

    predatorIndices = random.sample(range(0, len(validSpawnLocations)), ExperimentParams.NumRuns)
    predatorLocations = [validSpawnLocations[ind] for ind in predatorIndices]

    return predatorLocations


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
        planningDirectory = os.getcwd()

        UnitTests()

        if not os.path.exists('init_vars.pkl'):
            predatorLocations = GetPredatorLocations()

            with open('init_vars.pkl', 'wb') as f:
                pickle.dump(predatorLocations, f)

        with open('init_vars.pkl', 'rb') as f:
            predatorLocations = pickle.load(f)

        tasks = []
        for simulationInd in range(ExperimentParams.NumRuns):
            tasks.append([(simulationInd, predatorLocations[simulationInd]), planningDirectory])

        MultiExperiment(tasks[0])

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
                    print(
                        "Starting simulation %d"% (currentTask[0][0]))
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
                _ = MultiExperiment(task)
                # MPITest(task)
                comm.send(None, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
