import numpy as np

class COORD:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def Valid(self):
        return self.X >= 0 and self.Y >= 0

    def Copy(self):
        return COORD(self.X, self.Y)

    def __eq__(self, other):
        return self.X == other.X and self.Y == other.Y

    def __ne__(self, other):
        return self.X != other.X or self.Y != other.Y

    def __add__(self, other):
        return COORD(self.X+other.X, self.Y+other.Y)

    def __iadd__(self, offset):
        return COORD(self.X+offset.X, self.Y+offset.Y)

    def __mul__(self, mult):
        return COORD(self.X * mult, self.Y * mult)

    def __str__(self):
        return "(" + str(self.X) + ";" + str(self.Y) + ")"

    def __hash__(self):
        return 0

class COMPASS:
    NAA, NORTH, EAST, SOUTH, WEST, NOMOVE = range(-1, 5)

def Clockwise(dir):
    if dir < 4:
        return (dir + 1) % 4
    else:
        return None

def Anticlockwise(dir):
    if dir < 4:
        return (dir + 3) % 4
    else:
        return None

def Opposite(dir):
    if dir < 4:
        return (dir + 2) % 4
    else:
        return None

def EuclideanDistance(cord1, cord2):
    return np.sqrt((cord1.X - cord2.X)**2 + (cord1.Y - cord2.Y)**2)

def ManhattanDistance(cord1, cord2):
    return abs(cord1.X - cord2.X) + abs(cord1.Y - cord2.Y)

def DirectionalDistance(cord1, cord2, direction):
    return{
        0: (COORD(cord2.X - cord1.X, cord2.Y - cord1.Y) == COORD(0, cord2.Y - cord1.Y))*(cord2.Y - cord1.Y),
        1: (COORD(cord2.X - cord1.X, cord2.Y - cord1.Y) == COORD(cord2.X - cord1.X, 0))*(cord2.X - cord1.X),
        2: (COORD(cord1.X - cord2.X, cord1.Y - cord2.Y) == COORD(0, cord1.Y - cord2.Y)) * (cord1.Y - cord2.Y),
        3: (COORD(cord1.X - cord2.X, cord1.Y - cord2.Y) == COORD(cord1.X - cord2.X, 0)) * (cord1.X - cord2.X),
        #4: Infinity
    }.get(direction, False)

def AggressiveDirectionalDistance(lhs, rhs, direction):
    return{
        0: rhs.Y - lhs.Y,
        1: rhs.X - lhs.X,
        2: lhs.Y - rhs.Y,
        3: lhs.X - rhs.X,
    }.get(direction, False)

def LineCoordinates(coord1, coord2):
    a = coord1.Copy(); b = coord2.Copy()

    dx = b.X - a.X; dy = b.Y - a.Y

    isSteep = abs(dy) > abs(dx)
    if isSteep:
        a.X, a.Y = a.Y, a.X; b.X, b.Y = b.Y, b.X

    swapped = False
    if a.X > b.X:
        a.X, b.X = b.X, a.X; a.Y, b.Y = b.Y, a.Y; swapped = True

    dx = b.X - a.X; dy = b.Y - a.Y
    error = int(dx/2.0); ystep = 1 if a.Y < b.Y else -1

    y = a.Y
    points = []
    for x in range(a.X, b.X+1):
        coord = COORD(y, x) if isSteep else COORD(x, y)
        points.append(coord)
        error -= abs(dy)

        if error < 0:
            y += ystep; error += dx
    if swapped: points.reverse()

    return points

def Norm(cord1):
    return max(cord1.X, cord1.Y)

def CompassDistance(cord1, cord2, direction):
    return {
        0: cord2.Y - cord1.Y,
        1: cord2.X - cord1.X,
        2: cord1.Y - cord2.Y,
        3: cord1.X - cord2.X,
        #4: Infinity
    }.get(direction, False)

NaA = COORD(-1, -1)
North = COORD(0, 1)
East = COORD(1, 0)
West = COORD(-1, 0)
South = COORD(0, -1)
NoMove = COORD(0, 0)

Compass = [North,
           East,
           South,
           West,
           NoMove,
           NaA]

CompassString = ["N", "E", "S", "W", "NM", "NaA"]

def UnitTestCOORD():
    assert (COORD(3, 3) + COORD(2, 2) == COORD(5, 5))
    coord = COORD(5, 2)
    coord += COORD(2, 5)
    assert (coord == COORD(7, 7))
    assert (COORD(2, 2) + North == COORD(2, 3))
    assert (COORD(2, 2) + East == COORD(3, 2))
    assert (COORD(2, 2) + South == COORD(2, 1))
    assert (COORD(2, 2) + West == COORD(1, 2))
    assert (CompassString[COMPASS.NORTH] == "N")
    assert (CompassString[-1] == "NaA")
    assert (ManhattanDistance(COORD(3, 2), COORD(-4, -7)) == 16)
    assert (DirectionalDistance(COORD(3, 2), COORD(-4, -7), COMPASS.NORTH) == 0)
    assert (DirectionalDistance(COORD(3, 2), COORD(3, -7), COMPASS.SOUTH) == 9)

if __name__ == "__main__":
    UnitTestCOORD()