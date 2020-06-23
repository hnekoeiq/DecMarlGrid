from ..base import MultiGridEnv, MultiGrid
from ..objects import *
import numpy as np


class CheckersMultiGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        for i in range(int(np.floor(width/2))):
            for j in range(3):
                if (j%2) + 1 + 2*i < width - 1:
                    self.put_obj(Goal(color="red", reward=-100), (j%2) + 1 + 2*i, j + 1)

                if ((j+1)%2) + 1 + 2*i < width - 1:
                    self.put_obj(Goal(color="green", reward=100), ((j+1)%2) + 1 + 2*i, j + 1)


        self.place_agents()
