import heapq
from coord import *

class Cell(object):
    def __init__(self, coord, reachable):
        self.Reachable = reachable
        self.Coordinate = coord
        self.Parent = None
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

class AStar(object):
    def __init__(self, xsize, ysize, occlusions=[]):
        self.Opened = []
        heapq.heapify(self.Opened)
        self.Closed = set()

        self.Cells = []
        self.XSize = xsize
        self.YSize = ysize

        self.Occlusions = occlusions

    def InitializeGrid(self, startCoordinate, endCoordinate):
        for y in range(self.YSize):
            for x in range(self.XSize):
                reachable = True
                if COORD(x, y) in self.Occlusions:
                    reachable = False

                self.Cells.append(Cell(COORD(x,y), reachable))

        self.StartCell = self.GetCell(startCoordinate)
        self.EndCell = self.GetCell(endCoordinate)

    def Heuristic(self, cell):
        return 10 * (abs(cell.Coordinate.X - self.EndCell.Coordinate.X) + abs(cell.Coordinate.Y - self.EndCell.Coordinate.Y))

    def GetCell(self, coord):
        return self.Cells[coord.X + coord.Y *(self.YSize)]

    def GetNeighbors(self, cell):
        cells = []

        if cell.Coordinate.X < self.XSize - 1:
            cells.append(self.GetCell(COORD(cell.Coordinate.X + 1, cell.Coordinate.Y)))

        if cell.Coordinate.Y > 0:
            cells.append(self.GetCell(COORD(cell.Coordinate.X, cell.Coordinate.Y - 1)))

        if cell.Coordinate.X > 0:
            cells.append(self.GetCell(COORD(cell.Coordinate.X - 1, cell.Coordinate.Y)))

        if cell.Coordinate.Y < self.YSize - 1:
            cells.append(self.GetCell(COORD(cell.Coordinate.X, cell.Coordinate.Y + 1)))

        return cells

    def UpdateCell(self, neighbor, cell):
        neighbor.g = cell.g + 10
        neighbor.h = self.Heuristic(neighbor)
        neighbor.Parent = cell
        neighbor.f = neighbor.h + neighbor.g

        return neighbor

    def DisplayPath(self):
        cell = self.EndCell
        path = [(cell.Coordinate.X, cell.Coordinate.Y)]
        while cell.Parent is not self.StartCell:
            cell = cell.Parent
            path.append((cell.Coordinate.X, cell.Coordinate.Y))

        path.append((self.StartCell.Coordinate.X, self.StartCell.Coordinate.Y))
        path.reverse()
        return path

    def Solve(self):

        heapq.heappush(self.Opened, (self.StartCell.f, self.StartCell))
        while len(self.Opened):
            f, cell = heapq.heappop(self.Opened)

            self.Closed.add(cell)
            if cell is self.EndCell:
                path = self.DisplayPath()
                return path

            neighboringCells = self.GetNeighbors(cell)
            for neighboringCell in neighboringCells:
                if neighboringCell.Reachable and neighboringCell not in self.Closed:
                    if (neighboringCell.f, neighboringCell) in self.Opened:
                        if neighboringCell.g > cell.g + 10:
                            self.UpdateCell(neighboringCell, cell)

                    else:
                        self.UpdateCell(neighboringCell, cell)
                        heapq.heappush(self.Opened, (neighboringCell.f, neighboringCell))

        return None