from ..base import MultiGridEnv, MultiGrid
from ..objects import *
import numpy as np


class PursuitEvasionMultiGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        self.place_agents()
