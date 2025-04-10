import cv2


# NeighbourDetection class to detect neighbours of a given tile in a board game image.

class NeighbourDetection:
    def __init__(self, tile_grid):
        self.tile_grid = tile_grid
        self.rows = len(tile_grid)
        self.cols = len(tile_grid[0])


    def get_neighbours(self, i, j):

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbours = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                neighbours.append((ni, nj))
        return neighbours
    
    



        
        