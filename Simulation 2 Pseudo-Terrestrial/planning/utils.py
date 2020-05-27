import numpy as np

LargeInteger = 1e6
Infinity = 1e10
Tiny = 1e-10

def Sign(x):
    return (x > 0) - (x < 0)

def RandomDouble(min, max):
    return np.random.uniform(low=min, high=max)

def Random(min, max):
    if min != max:
        return np.random.randint(low=min, high=max)
    else:
        return min

def RandomSeed(seed):
    np.random.seed(seed)

def Bernoulli(p):
    return np.random.binomial(1, p)

def Near(x, y, tol):
    return abs(x-y) <= tol

def SetFlag(flags, bit):
    return flags | (1 << bit)

def CheckFlag(flags, bit):
    return (flags & (1 << bit)) != 0

def UnitTestUTILS():
    assert (Sign(10) == 1)
    assert (Sign(-10) == -1)
    assert (Sign(0) == 0)

    n = [0] * 6
    for i in range(10000):
        for j in range(1, 6):
            n[j] += (Random(0, j) == 0)
    assert (Near(n[1], 10000, 0))
    assert (Near(n[2], 5000, 250))
    assert (Near(n[3], 3333, 250))
    assert (Near(n[4], 2500, 250))
    assert (Near(n[5], 2000, 250))

    c = 0
    for i in range(10000):
        c += Bernoulli(0.5)
    assert (Near(c, 5000, 250))


if __name__ == '__main__':
    UnitTestUTILS()
