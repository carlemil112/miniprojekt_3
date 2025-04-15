import cv2
from cropped_outer_rim import TileTypeDetection

class NeighbourDetection:
    def __init__(self, tile_grid):
        """
        tile_grid: 2D list of classified tile types, e.g. [['forest', 'water', ...], [...], ...]
        """
        self.tile_grid = tile_grid
        self.rows = len(tile_grid)
        self.cols = len(tile_grid[0])

    def get_neighbours(self, i, j):
        """
        Return the coordinates of valid 4-connected neighbours.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        neighbours = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                neighbours.append((ni, nj))
        return neighbours

    def get_neighbour_types(self, i, j):
        """
        Return the types of valid 4-connected neighbours.
        """
        neighbour_coords = self.get_neighbours(i, j)
        neighbour_types = [self.tile_grid[ni][nj] for ni, nj in neighbour_coords]
        return neighbour_types

    def count_matching_neighbours(self, i, j):
        """
        Return the number of neighbours that match the tile type at (i, j).
        """
        center_type = self.tile_grid[i][j]
        neighbour_types = self.get_neighbour_types(i, j)
        return sum(1 for t in neighbour_types if t == center_type)

    def get_matching_neighbour_coords(self, i, j):
        """
        Return the coordinates of neighbours that match the tile type at (i, j).
        """
        center_type = self.tile_grid[i][j]
        neighbour_coords = self.get_neighbours(i, j)
        matching_coords = [(ni, nj) for ni, nj in neighbour_coords if self.tile_grid[ni][nj] == center_type]
        return matching_coords
