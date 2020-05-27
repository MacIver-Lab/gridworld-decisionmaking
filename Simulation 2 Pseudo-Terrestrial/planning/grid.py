from coord import COORD, COMPASS, ManhattanDistance, Compass, LineCoordinates
import numpy as np
from utils import Random

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

    def VisualRay(self, coord1, coord2, occlusions):
        points = LineCoordinates(coord1, coord2)
        intersections = set(occlusions).intersection(set(points))
        return (len(intersections) == 0), points

    def DistToEdge(self, coord, direction):
        assert(self.Inside(coord))
        return {
            1: self.YSize - 1 - coord.Y,
            2: self.XSize - 1 - coord.X,
            3: coord.Y,
            4: coord.X
        }.get(direction, False)

    def ValidOcclusion(self, coord, agentCoord, goalCoord, occlusionCoords):
        aroundOrigin = [COORD(0, 0) + COORD(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        aroundAgent = [agentCoord + COORD(i, j) for i in range(-1, 2) for j in
                       range(-1, 2)]  # [agent + COORD(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        aroundGoal = [goalCoord + COORD(i, j) for i in range(-1, 2) for j in range(-1, 2)]

        return coord not in aroundOrigin and \
               coord not in aroundAgent and \
               coord not in aroundGoal and \
               coord not in occlusionCoords

    def CreateRandomOcclusions(self, value, agentCoord, goalCoord):
        allOcclusionCoords = []
        for i in range(value):
            while True:
                coord = COORD(Random(0, self.XSize), Random(0, self.YSize))
                if self.ValidOcclusion(coord, agentCoord, goalCoord,
                                       allOcclusionCoords):
                    break
            allOcclusionCoords.append(coord)

        return allOcclusionCoords

    def CreateOcclusions(self, value, agentCoord, goalCoord):
        occlusionCoords = []
        if value == 0:
            return occlusionCoords

        areaSizes = []
        while True:
            limit = value
            if value > 10:
                limit = 10
            size = Random(1, limit)
            if size + np.sum(np.asarray(areaSizes)) <= value:
                areaSizes.append(size)
            if np.sum(np.asanyarray(areaSizes)) == value:
                break

        allOcclusionCoords = []
        for area in areaSizes:
            triedCoordinates = []
            while True:
                areaOcclusionCoords = []
                while True:
                    startCoord = COORD(Random(0, self.XSize), Random(0, self.YSize))
                    if self.ValidOcclusion(startCoord, agentCoord, goalCoord, allOcclusionCoords) and startCoord not in triedCoordinates:
                        break

                areaOcclusionCoords.append(startCoord)
                n = 0
                for i in range(area - 1):
                    if n > 100:
                        break
                    n = 0
                    while True:
                        if n > 100:
                            break
                        adjacentCell = areaOcclusionCoords[-1] + Compass[Random(0, 4)]
                        if self.Inside(adjacentCell) and self.ValidOcclusion(adjacentCell, agentCoord, goalCoord, allOcclusionCoords):
                            areaOcclusionCoords.append(adjacentCell)
                            break
                        n += 1
                if n > 100:
                    triedCoordinates.append(startCoord)
                    continue
                allOcclusionCoords.extend(areaOcclusionCoords)
                break

        return allOcclusionCoords

    def CreatePredatorLocations(self, spawnArea, agentCoord, goalCoord, occlusions=[]):
        agentSurroundCoordinates = []
        for x in range(agentCoord.X - spawnArea, agentCoord.X + spawnArea + 1):
            for y in range(agentCoord.Y - spawnArea, agentCoord.Y + spawnArea + 1):
                if (ManhattanDistance(COORD(x, y), agentCoord) <= 2 * spawnArea):
                    agentSurroundCoordinates.append(COORD(x, y))

        agentSurroundCoordinates.append(goalCoord)
        agentSurroundCoordinates.extend(occlusions)

        predatorLocations = []
        for y in range(self.YSize):
            for x in range(self.XSize):
                newLocation = COORD(x, y)
                if newLocation in agentSurroundCoordinates:
                    continue

                predatorLocations.append(newLocation)

        return predatorLocations

    def __getitem__(self, item):
        assert(item >= 0 and item < self.XSize * self.YSize)
        return self.Grid[item]