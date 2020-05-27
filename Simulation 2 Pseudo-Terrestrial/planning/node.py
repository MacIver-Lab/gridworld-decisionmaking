from beliefstate import BeliefState
from utils import Infinity

class Value:
    def __init__(self):
        self.Count = 0.0
        self.Total = 0.0

    def Set(self, count, value):
        self.Count = float(count)
        self.Total = float(value) * float(count)

    def Add(self, totalReward, weight=1):
        self.Count += 1.0
        self.Total += float(totalReward) * float(weight)

    def GetValue(self):
        if self.Count == 0:
            return self.Total
        else:
            return float(self.Total) / float(self.Count)

    def GetTrueValue(self):
        return float(self.Total)

    def GetCount(self):
        return self.Count

    def __str__(self):
        return "(" + str(self.Count) + " , " + str(self.Total) + ")"

class QNode:
    NumChildren = 0
    def __init__(self):
        self.Children = [] #children of vnodes
        self.Value = Value()
        self.AMAF = Value()

    def Child(self, c):
        return self.Children[c]

    def QInitialise(self, full):
        assert(QNode.NumChildren)
        numChildren = QNode.NumChildren
        if full:
            numChildren = 1
        self.Children = [None for action in range(numChildren)]

        #for observation in xrange(QNode.NumChildren):
        #    self.Children[observation].Value.Count = 0
        #    self.Children[observation].Value.Count = 0

    def QDisplayValue(self, history, maxDepth):
        history.Display()
        print(": " + str(self.Value.GetValue()) + "(" + str(self.Value.GetCount()) + ")")
        if history.Size() >= maxDepth:
            return

        for observation in range(QNode.NumChildren):
            if self.Children[observation]:
                history.Back().Observation = observation
                self.Children[observation].VDisplayValue(history, maxDepth)

    def QDisplayPolicy(self, history, maxDepth):
        history.Display()
        print(": " + str(self.Value.GetValue()) + " (" + str(self.Value.GetCount()) + ")")
        if history.Size() >= maxDepth:
            return
        for observation in range(QNode.NumChildren):
            if self.Children[observation]:
                history.Back().Observation = observation
                self.Children[observation].VDisplayPolicy(history, maxDepth)

    def __str__(self):
        return "QN( " + str(QNode.NumChildren) + " ; " + str(self.Value) + " ; " + str(self.AMAF) + " )"

class VNode:
    NumChildren = 0
    def __init__(self):
        self.Value = Value()
        self.BeliefState = BeliefState()
        self.Children = [] #children of qnodes

    @classmethod
    def Free(cls, vnode, simulator):
        vnode.BeliefState.Free(simulator)
        del vnode

    def VInitialise(self, full):
        assert(VNode.NumChildren)
        numChildren = VNode.NumChildren
        if full:
            numChildren = 5
        self.Children = [QNode() for action in range(numChildren)]
        for action in range(numChildren):
            self.Children[action].QInitialise(full)

    def Create(self, full=False):
        self.VInitialise(full)
        vnode = VNode()
        vnode.Children = self.Children
        vnode.BeliefState = self.BeliefState
        vnode.Value = self.Value
        return vnode

    def GetNumChildren(self):
        return len(self.Children)

    def Child(self, c):
        return self.Children[c]

    def Beliefs(self):
        return self.BeliefState

    def SetChildren(self, count, value):
        for a in range(self.GetNumChildren()):
            qnode = self.Children[a]
            qnode.Value.Set(count, value)
            qnode.AMAF.Set(count, value)

    def VDisplayValue(self, history, maxDepth):
        if history.Size() >= maxDepth:
            return

        for action in range(VNode.NumChildren):
            history.Add(action)
            self.Children[action].QDisplayValue(history, maxDepth)
            history.Pop()

    def VDisplayPolicy(self, history, maxDepth):
        if history.Size() >= maxDepth:
            return

        bestq = -Infinity
        besta = -1

        for action in range(self.NumChildren):
            if self.Children[action].Value.GetValue() > bestq:
                besta = action
                bestq = self.Children[action].Value.GetValue()

        if besta != -1:
            history.Add(besta)
            self.Children[besta].QDisplayPolicy(history, maxDepth)
            history.Pop()

    def __str__(self):
        return "VN( " + str(VNode.NumChildren) + " ; " + str(self.Value) + " )"


if __name__ == "__main__":
    vnode = VNode()
    VNode.NumChildren = 4
    QNode.NumChildren = 5
    v = vnode.Create()