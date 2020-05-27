from collections import Counter
import numpy as np


class Entropy(object):
    def __init__(self, I):
        self.I = I
        self.YSize, self.XSize = I.shape

    def MovingWindowFilter(self, f, radius):
        output = np.zeros([self.YSize - radius + 1, self.XSize - radius + 1])

        for (row, col), value in np.ndenumerate(self.I):
            if not (row > (self.YSize - radius) or col > (self.XSize - radius)):
                componentMatrix = np.zeros([radius, radius])

                for rowOffset in range(0, radius):
                    for colOffset in range(0, radius):
                        componentMatrix[rowOffset][colOffset] = self.I[row + rowOffset][col + colOffset]

                output[row][col] = f(componentMatrix)

        return output

    def MovingAverage(self, componentMatrix):
        runningTotal = 0
        numComponents = 0

        for (row, col), value in np.ndenumerate(componentMatrix):
            runningTotal += value
            numComponents += 1

        output = runningTotal / numComponents

        return output

    def CalculateEntropy(self, prob):
        runningTotal = 0
        for item in prob:
            runningTotal += item * np.log2(item)

        if runningTotal != 0:
            runningTotal *= -1

        return runningTotal

    def BinaryEntropy(self, p0, p1):
        return self.CalculateEntropy([p0, p1])

    def MatrixEntropy(self, matrix):
        counts = Counter(matrix.flatten())
        totalCount = sum(counts.values())
        if len(counts) == 1:
            discreteDist = [counts[0.0] / totalCount]
        else:
            discreteDist = [counts[0.0] / totalCount, counts[1.0] / totalCount]

        return self.CalculateEntropy(discreteDist)

    def Profile(self, filteredMatrices):
        #temp = []
        #for filt in filteredMatrices:
        #    ent = self.MatrixEntropy(filt)
         #   temp.append(ent)
        #return temp
        return [self.MatrixEntropy(filt) for filt in filteredMatrices]

def UnitTestENTROPY():
    inputMatrices = []

    inputMatrices.append(np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0]]))

    inputMatrices.append(np.array([
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0]]))

    inputMatrices.append(np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]))

    inputMatrices.append(np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]]))

    inputMatrices.append(np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]]))

    for m in range(0, len(inputMatrices)):
        activeMatrix = inputMatrices[m]
        entropy = Entropy(activeMatrix)
        #print("---------\nMatrix #{0}\n---------\n".format(m))

        # Produce the filtered matrices at varying scales and the associated
        # entropy "profiles"
        matrices = []
        for n in range(1, min(activeMatrix.shape)):
            outputMatrix = entropy.MovingWindowFilter(f=entropy.MovingAverage, radius=n)

            #subplot = pyplot.subplot(5, 4, m * 4 + n)
            #pyplot.axis('off')
            #pyplot.imshow(outputMatrix,
            #              interpolation='nearest',
            #              cmap='Greys_r',
            #              vmin=0,
            #              vmax=1)
            #matrices.append(outputMatrix)

            #print("Neighborhood size = {0}\n{1}\n".format(n, outputMatrix))
        profile = entropy.Profile(matrices)
        #print("Profile:\n{0}\n".format(entropy.Profile(matrices)))

    #pyplot.show()

if __name__=="__main__":
    UnitTestENTROPY()



