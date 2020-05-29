import numpy as np

LargeInteger = 1e6
Infinity = 1e10
Tiny = 1e-10

def Softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def Sign(x):
    return (x > 0) - (x < 0)

def Entropy(I):
    I = np.array(I)
    m, n = I.shape
    h = [0.0] *256
    h = np.array(h)
    for i in range(m):
        for j in range(n):
            h[int(I[i,j])] += 1

    h /= np.sum(h)
    h = h[h != 0]

    return -np.sum(h*np.log2(h))

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


def UnitTestUTILS():
    assert (Sign(10) == 1)
    assert (Sign(-10) == -1)
    assert (Sign(0) == 0)

    n = [0] * 6
    for i in xrange(10000):
        for j in xrange(1, 6):
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

    I = np.zeros((20, 20))
    assert(Entropy(I) == 0)

    I[3, 0:4] = 1
    I[5, 9:19] = 1
    assert(0.21 < Entropy(I) and Entropy(I) < 0.22)

if __name__ == '__main__':
    UnitTestUTILS()
