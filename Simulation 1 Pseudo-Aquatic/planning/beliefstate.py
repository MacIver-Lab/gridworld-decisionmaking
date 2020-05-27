from utils import Random

class BeliefState():
    def __init__(self):
        self.Samples = []

    def Empty(self):
        self.Samples = []

    def GetNumSamples(self):
        return len(self.Samples)

    def GetSample(self, index):
        return self.Samples[index]

    def Copy(self, beliefs, simulator):
        for state in beliefs:
            self.AddSample(simulator.Copy(state))

    def CreateSample(self, simulator):
        index = Random(0, len(self.Samples))
        return simulator.Copy(self.Samples[index])

    def AddSample(self, state):
        self.Samples.append(state)

    def Move(self, beliefs):
        for state in beliefs.Samples:
            self.AddSample(state)
        beliefs.Samples = []

    def Free(self, simulator):
        for state in self.Samples:
            simulator.FreeState(state)
        self.Samples = []

    def __getitem__(self, item):
        return self.Samples[item]