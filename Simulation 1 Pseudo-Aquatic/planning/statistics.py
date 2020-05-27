import numpy as np
from utils import *

class STATISTICS:
    def __init__(self, val, count):
        self.Value = val
        self.Count = count
        self.Mean = val
        self.Variance = 0.
        self.Min = 0.
        self.Max = 0.

    def SetValue(self, val):
        self.Value = val

    def SetCount(self, count):
        self.Count = count

    def Add(self, val):
        meanOld = float(self.Mean)
        countOld = float(self.Count)

        self.Count += 1.0
        assert(self.Count > 0)
        self.Mean += float(float((val - self.Mean))/float(self.Count))
        self.Variance = float(float((countOld*(self.Variance + meanOld**2) + val**2))/float(self.Count) - self.Mean**2)

        if val > self.Max:
            self.Max = val
        if val < self.Min:
            self.Min = val

    def Clear(self):
        self.Count = 0
        self.Mean = 0.0
        self.Variance = 0.0
        self.Min = Infinity
        self.Max = -Infinity

    def Initialise(self, val, count):
        self.Mean = val
        self.Count = count

    def GetValue(self):
        return self.Value

    def GetTotal(self):
        return self.Mean * self.Count

    def GetStdDev(self):
        return np.sqrt(self.Variance)

    def GetStdError(self):
        return np.sqrt(self.Variance/float(self.Count))

    def GetMean(self):
        return self.Mean

    def GetCount(self):
        return self.Count

    def __str__(self):
        return "[ " + str(self.Mean) + " , " + str(self.Variance) + " ]"