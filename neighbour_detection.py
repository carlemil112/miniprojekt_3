import cv2

class NeighbourDetection:
    def __init__(self, tile_grid, crown_grid):
        """
        tile_grid: 2D list of classified tile types, e.g. [['forest', 'water', ...], [...], ...]
        """
        self.tile_grid = tile_grid
        self.crown_grid = crown_grid
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
    
    def calculate_region_score(self, start_i, start_j):
        '''
        Calculate the score for a connected region starting at (start_i, start_j).
        '''
        visited = set()
        stack = [(start_i, start_j)]
        total_crowns = 0
        tile_type = self.tile_grid[start_i][start_j]

        while stack:
            i, j = stack.pop()
            if (i, j) in visited:
                continue

            visited.add((i, j))
            total_crowns += self.crown_grid[i][j]

            # Get all matching neighbors
            for ni, nj in self.get_matching_neighbour_coords(i, j):
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

        return len(visited) * total_crowns


    def find_all_regions(self):
        visited = set()
        regions = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in visited and self.tile_grid[i][j] != 'empty':
                    region = []
                    stack = [(i, j)]
                    tile_type = self.tile_grid[i][j]
                    
                    while stack:
                        x, y = stack.pop()
                        if (x, y) in visited:
                            continue
                            
                        visited.add((x, y))
                        region.append((x, y))
                        
                        # Add matching neighbors
                        for nx, ny in self.get_matching_neighbour_coords(x, y):
                            if (nx, ny) not in visited:
                                stack.append((nx, ny))
                                
                    if region:
                        regions.append({
                            'tile_type': tile_type,
                            'tiles': region,
                            'score': self.calculate_region_score(i, j)
                        })
        return regions