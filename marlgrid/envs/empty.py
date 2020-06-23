from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class EmptyMultiGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(color="green", reward=100), 1, 1)
        self.put_obj(Goal(color="green", reward=100), width - 2, height - 2)
        self.put_obj(Goal(color="red", reward=-200), 1, height - 2)
        # self.put_obj(Goal(color="red", reward=100), width - 2, 1)
        # self.put_obj(Goal(color="red", reward=-1), width - 5, height - 3)
        # self.put_obj(Goal(color="red", reward=-1), width - 5, height - 2)
        # self.put_obj(Goal(color="red", reward=-1), width - 2, height - 5)
        self.place_agents()
