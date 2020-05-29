from coord import COORD, COMPASS, ManhattanDistance, Compass
import numpy as np

class Grid:
    def __init__(self, xsize, ysize):
        self.XSize = xsize
        self.YSize = ysize
        self.Grid = [0]*(xsize * ysize)

    def GetXSize(self):
        return self.XSize

    def GetYSize(self):
        return self.YSize

    def Resize(self, xsize, ysize):
        self.Grid.append([0]*abs((self.XSize *  self.YSize) - (xsize * ysize)))
        self.XSize = xsize
        self.YSize = ysize

    def Inside(self, coord):
        return coord.X >= 0 and coord.Y >= 0 and coord.X < self.XSize and coord.Y < self.YSize

    def Index(self, x, y):
        assert (self.Inside(COORD(x, y)))
        return ((self.XSize) * (self.YSize - 1 - y)) + x

    def Coordinate(self, index):
        assert (index < self.XSize * self.YSize)
        return COORD(divmod(index, self.XSize)[0], divmod(index, self.XSize)[1])

    def VisualArea(self, coord, observationDirection, visualRange, pureVision=False):
        RadiusCoordinates = []
        for x in range(coord.X - visualRange, coord.X + visualRange + 1):
            for y in range(coord.Y - visualRange, coord.Y + visualRange + 1):
                if (ManhattanDistance(COORD(x, y), coord) <= 2 * visualRange):
                    RadiusCoordinates.append(COORD(x, y))
        
        RangeCoordinates = []
        RadiusCoordinates = np.flipud(
            np.reshape(np.array(RadiusCoordinates), (2 * visualRange + 1, 2 * visualRange + 1)).transpose())
        if observationDirection == COMPASS.NORTH:
            RangeCoordinates = self.VisualCone(RadiusCoordinates, visualRange)

        elif observationDirection == COMPASS.EAST:
            RadiusCoordinates = np.flipud(RadiusCoordinates.transpose())
            RangeCoordinates = self.VisualCone(RadiusCoordinates, visualRange)

        elif observationDirection == COMPASS.WEST:
            RadiusCoordinates = RadiusCoordinates.transpose()
            RangeCoordinates = self.VisualCone(RadiusCoordinates, visualRange)

        elif observationDirection == COMPASS.SOUTH:
            RadiusCoordinates = np.flipud(RadiusCoordinates)
            RangeCoordinates = self.VisualCone(RadiusCoordinates, visualRange)

        assert (RangeCoordinates)

        if pureVision:
            return RangeCoordinates

        for a in range(4):
            sidePos = coord + Compass[a]
            if sidePos not in RangeCoordinates:
                RangeCoordinates.append(sidePos)

        return RangeCoordinates

    def VisualCone(self, coord, visualRange):
        temp = []
        for dec in range(visualRange):
            temp.append(coord[dec, dec:(2 * visualRange - dec) + 1])
        return [y for x in temp for y in x]  # if self.Inside(y)]

    def DistToEdge(self, coord, direction):
        assert(self.Inside(coord))
        return {
            1: self.YSize - 1 - coord.Y,
            2: self.XSize - 1 - coord.X,
            3: coord.Y,
            4: coord.X
        }.get(direction, False)

    def __getitem__(self, item):
        assert(item >= 0 and item < self.XSize * self.YSize)
        return self.Grid[item]